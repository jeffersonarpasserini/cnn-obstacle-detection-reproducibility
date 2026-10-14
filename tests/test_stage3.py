"""Independent statistical/visibility checks for the exploratory stage."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score,cohen_kappa_score

from src.stage3_statistics import (auc_draws,bootstrap_weights,metric_draws,oracle_threshold,
    oracle_metric_draws,partial_auc,partial_auc_draws,specificity_draws,delong_contrast,cluster_logistic,intervals)
from src.stage3_images import human_review_templates,cosine_nearest
from scripts.analyze_blind_relabel import agreement,analyze
from scripts.build_stage3_report import ROOT as PROJECT_ROOT
from src.stage3_review import (load_error_categories,validate_author_review,validate_blind_analysis,
                              build_review_tables)


def test_scene_weights_and_tied_auc_match_literal_resampling():
    y=np.array([0,1,0,1,1,0]); groups=np.array(['mixed','mixed','b','c','d','e'])
    score=np.array([.2,.2,.8,.9,.7,.6]); w=bootstrap_weights(y,groups,60)
    assert np.array_equal(w[:,0],w[:,1])
    actual=auc_draws(y,score,w)
    for value,draw in zip(actual,w):
        indices=np.repeat(np.arange(len(y)),draw)
        assert value==pytest.approx(roc_auc_score(y[indices],score[indices]))


def test_mean_seed_metric_is_not_an_ensemble_metric():
    y=np.array([0,0,1,1]); decisions=np.array([[0,1,1,1],[0,0,0,1],[1,0,1,1]])
    actual=metric_draws(y,decisions,np.ones((1,4)))
    assert actual['accuracy'][0]==pytest.approx(np.mean(decisions==y))
    assert actual['obstructed_recall'][0]+actual['obstructed_false_negative_rate'][0]==1


def test_all_metrics_reproduce_existing_article_metrics_per_seed():
    from src.artifact import prediction_metrics
    y=np.array([0,0,1,1]); d=np.array([[0,1,1,1],[0,0,0,1],[1,0,1,1]])
    s=np.array([[.1,.6,.8,.9],[.1,.2,.4,.9],[.6,.1,.7,.8]])
    actual=metric_draws(y,d,np.ones((1,4)),s)
    expected=[prediction_metrics(y,decision,score) for decision,score in zip(d,s)]
    for key in expected[0]:
        assert actual[key][0]==pytest.approx(np.mean([row[key] for row in expected]))


def test_subgroup_recall_absent_class_is_unestimable_and_draws_are_counted():
    y=np.array([0,1]); d=y.copy(); w=np.array([[1,1],[2,0],[0,2],[0,0]])
    actual=intervals(y,d,w)
    assert actual['accuracy']['bootstrap_valid']==3
    assert actual['clear_recall']['bootstrap_valid']==2
    assert actual['obstructed_recall']['bootstrap_valid']==2
    assert actual['clear_recall']['ci_low']==1
    absent=intervals(np.zeros(3,int),np.array([0,1,0]),np.ones((2,3)))
    assert np.isnan(absent['clear_recall']['estimate'])
    assert np.isnan(absent['clear_recall']['ci_low'])
    assert absent['clear_recall']['bootstrap_valid']==0


@pytest.mark.parametrize('score',[np.array([.2,.2,.8,.8]),np.ones(4),np.array([0,0,1,1])])
def test_external_oracle_exhaustive_ties(score):
    y=np.array([0,1,0,1]); threshold,value=oracle_threshold(y,score)
    candidates=np.r_[np.unique(score),np.nextafter(score.max(),np.inf)]
    expected=max(np.mean((score>=t)==y) for t in candidates)
    assert value==expected and np.mean((score>=threshold)==y)==expected
    draws=oracle_metric_draws(y,score,np.ones((1,len(y))))
    assert draws['accuracy'][0]==expected


def test_partial_auc_focus_is_sensitivity_not_low_false_positive_rate():
    y=np.array([0,0,1,1]); perfect=np.array([0.,0.,1.,1.]); tied=np.ones(4)
    assert partial_auc(y,perfect)==pytest.approx(1.)
    assert partial_auc(y,tied)==pytest.approx(.5)
    assert partial_auc_draws(y,tied,np.ones((1,4)))[0]==pytest.approx(.5)
    assert partial_auc(y,1-perfect)<.5


def test_batched_partial_auc_and_oracle_reselect_in_each_draw():
    y=np.array([0,0,0,1,1,1]); s=np.array([.1,.4,.5,.4,.7,.9]); w=bootstrap_weights(y,None,50)
    p=partial_auc_draws(y,s,w); oracle=oracle_metric_draws(y,s,w)
    for i,draw in enumerate(w):
        index=np.repeat(np.arange(len(y)),draw)
        assert p[i]==pytest.approx(partial_auc(y[index],s[index]))
        assert oracle['accuracy'][i]==pytest.approx(oracle_threshold(y[index],s[index])[1])


def test_recall_matched_specificity_handles_score_ties():
    y=np.array([0,0,1,1]); score=np.array([.2,.2,.2,.9])
    value=specificity_draws(y,score,np.ones((1,4)))
    assert np.all(value==.5)


def test_delong_correlated_identical_scores_have_zero_difference():
    y=np.array([0,0,0,1,1,1]); s=np.array([.1,.4,.6,.2,.8,.9])
    result=delong_contrast(y,np.vstack([s,s,s,s]),[1,-1/3,-1/3,-1/3])
    assert result['difference']==pytest.approx(0,abs=1e-15)
    assert result['standard_error']<1e-8
    assert result['p_value']==1


def test_cosine_via_excludes_its_own_image():
    vectors=np.eye(3)
    assert np.all(cosine_nearest(vectors,vectors)==0)
    assert np.all(cosine_nearest(vectors,vectors,exclude_self=True)==1)


def test_manual_fields_are_empty_and_existing_annotations_preserved(tmp_path):
    human_review_templates(tmp_path,range(12),[{'filename':'a.jpg','error_type':'clear_to_nonclear'}])
    assert pd.read_csv(tmp_path/'cluster_names.csv').name.isna().all()
    assert pd.read_csv(tmp_path/'error_annotations.csv').category.isna().all()
    frame=pd.read_csv(tmp_path/'cluster_names.csv',keep_default_na=False); frame.loc[0,'name']='Author label'
    frame.to_csv(tmp_path/'cluster_names.csv',index=False)
    with pytest.raises(ValueError,match='preserve'):
        human_review_templates(tmp_path,range(12),[{'filename':'a.jpg','error_type':'clear_to_nonclear'}])
    assert pd.read_csv(tmp_path/'cluster_names.csv').loc[0,'name']=='Author label'


def test_cluster_logistic_is_descriptive_and_robust():
    rng=np.random.default_rng(1980); x=rng.normal(size=(400,2)); groups=np.repeat(np.arange(100),4)
    p=1/(1+np.exp(-(.2+.6*x[:,0]-.3*x[:,1])))
    beta,se,status=cluster_logistic(x,rng.binomial(1,p),groups)
    assert status['converged'] and status['clusters']==100
    assert beta[1]>0 and np.isfinite(se).all() and (se>0).all()


def test_blind_agreement_script_with_synthetic_replies(tmp_path):
    y=np.tile([0,1],50); labels=y.copy(); labels[:10]=1-labels[:10]
    key=pd.DataFrame({'item':[f'item_{i:03d}.jpg' for i in range(100)],'original_label':y,
        'filename':[f'image_{i}' for i in range(100)],'group_id':np.arange(100)})
    response=pd.DataFrame({'item':key.item,'label':np.where(labels,'clear','nonclear'),'confidence(1-3)':2,'notes':''})
    key.to_csv(tmp_path/'key.csv',index=False); response.to_csv(tmp_path/'responses.csv',index=False)
    analyze(tmp_path/'responses.csv',tmp_path/'key.csv',tmp_path/'output',iterations=60,synthetic=True)
    actual=pd.read_csv(tmp_path/'output/manuscript_tables/agreement.csv').iloc[0]
    assert actual.agreement==.9 and actual.cohen_kappa==pytest.approx(cohen_kappa_score(y,labels))
    assert actual.synthetic and len(pd.read_csv(tmp_path/'output/manuscript_tables/discordances.csv'))==10


def test_protected_hashes_detect_content_changes(tmp_path,monkeypatch):
    import src.stage3_io as io
    source=tmp_path/'frozen.csv'; source.write_text('unchanged')
    monkeypatch.setattr(io,'ROOT',tmp_path); monkeypatch.setattr(io,'protected_files',lambda:[source])
    before=io.snapshot(); (tmp_path/'protected_before.json').write_text(json.dumps(before))
    source.write_text('changed')
    with pytest.raises(ValueError,match='Protected'):
        io.check_protected(tmp_path)


def test_native_external_author_regression_is_read_only():
    from src.stage3_io import inputs
    frame,manifest,truth,groups=inputs()
    expected={'a':(.809010152284264,61),'b':(.8128172588832487,12),
              'c':(.807741116751269,34),'d':(.8153553299492385,39)}
    for name,panel in frame[frame.family.eq('fixed')].groupby('model'):
        panel=panel.sort_values('filename'); native=(panel.native_score.to_numpy()>0).astype(int)
        accuracy,discordance=expected[name[0]]
        assert np.mean(native==truth)==pytest.approx(accuracy)
        assert np.sum(native!=panel.predicted_label_default.to_numpy())==discordance


def test_historical_fold_table_missing_fnr_uses_complement_of_recall():
    from scripts.analyze_stage3 import ROOT,fold_endpoint_values
    historical=pd.read_csv(ROOT/'results/full_search/per_fold_metrics.csv')
    values=fold_endpoint_values(historical,'obstructed_false_negative_rate')
    assert np.allclose(values,1-historical.obstructed_recall)
    assert values.std(ddof=1)==pytest.approx(historical.obstructed_recall.std(ddof=1))


@pytest.fixture
def final_review_root(tmp_path):
    """Exercise the real author evidence using disposable copies only."""
    paths = [
        'configs/stage3.json',
        'results/stage3/error_analysis/cluster_names.csv',
        'results/stage3/error_analysis/error_annotations.csv',
        'results/stage3/error_analysis/CATEGORY_ADDENDUM.md',
        'results/stage3/error_analysis/manuscript_tables/consensus_errors.csv',
        *[f'results/stage3/error_analysis/manuscript_tables/{name}.csv'
          for name in ['cluster_assignments', 'cluster_counts', 'cluster_metrics']],
        'results/stage3/blind_relabel/status.json',
        'external-data/blind_relabel/responses_breve.csv',
        *[f'results/stage3/blind_relabel/manuscript_tables/{name}.csv'
          for name in ['agreement', 'confusion_matrix', 'discordances']],
    ]
    for name in paths:
        destination = tmp_path/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT/name, destination)
    return tmp_path


def test_category_addendum_is_read_dynamically_and_stops_at_next_heading(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'error_categories': ['original']}), encoding='utf-8')
    addendum = tmp_path/'addendum.md'
    addendum.write_text(
        '# Adendo\n\n## Categorias acrescentadas\nTexto fora da lista.\n\n'
        '## Lista completa usada na anotação\n\noriginal, categoria dinâmica,\n'
        'outra categoria.\n\n## Regras de precedência\nNão é uma categoria.\n',
        encoding='utf-8',
    )
    assert load_error_categories(config, addendum) == [
        'original', 'categoria dinâmica', 'outra categoria',
    ]
    addendum.write_text('## Outra seção\noriginal, categoria dinâmica.', encoding='utf-8')
    with pytest.raises(ValueError, match='complete annotation list'):
        load_error_categories(config, addendum)


@pytest.mark.parametrize('filename,column', [
    ('cluster_names.csv', 'name'), ('cluster_names.csv', 'reviewer'),
    ('cluster_names.csv', 'date'), ('error_annotations.csv', 'category'),
    ('error_annotations.csv', 'reviewer'), ('error_annotations.csv', 'date'),
])
def test_final_review_rejects_incomplete_sheet(final_review_root, filename, column):
    path = final_review_root/'results/stage3/error_analysis'/filename
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[0, column] = '   '
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='incomplete'):
        validate_author_review(final_review_root)


def test_final_review_rejects_category_outside_addendum(final_review_root):
    path = final_review_root/'results/stage3/error_analysis/error_annotations.csv'
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[0, 'category'] = 'categoria não autorizada'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='outside the configured/addendum list'):
        validate_author_review(final_review_root)


@pytest.mark.parametrize('change', ['order', 'counts', 'cluster'])
def test_final_review_rejects_order_counts_or_cluster_changes(final_review_root, change):
    folder = final_review_root/'results/stage3/error_analysis'
    if change == 'cluster':
        path = folder/'cluster_names.csv'
        frame = pd.read_csv(path, keep_default_na=False)
        frame.loc[0, 'cluster'] = 1
    else:
        path = folder/'error_annotations.csv'
        frame = pd.read_csv(path, keep_default_na=False)
        if change == 'order':
            frame = frame.iloc[[1, 0, *range(2, len(frame))]]
        else:
            frame.loc[0, 'category'] = 'sombra'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        validate_author_review(final_review_root)


def test_final_review_and_blind_analysis_accept_valid_evidence_read_only(final_review_root):
    from src.study_io import sha256
    files = [path for path in final_review_root.rglob('*') if path.is_file()]
    before = {path: sha256(path) for path in files}
    clusters, errors = validate_author_review(final_review_root)
    status = validate_blind_analysis(final_review_root)
    assert len(clusters) == 12 and len(errors) == 148
    assert errors.error_type.value_counts().to_dict() == {
        'clear_to_nonclear': 137, 'nonclear_to_clear': 11,
    }
    assert status['rater'] == 'Fabricio Breve'
    assert {path: sha256(path) for path in files} == before


def test_handoff_mode_accepts_empty_fields_and_rejects_final_review(final_review_root):
    with pytest.raises(ValueError, match='stay empty'):
        validate_author_review(final_review_root, mode='handoff')
    with pytest.raises(ValueError, match='no real responses yet'):
        validate_blind_analysis(final_review_root, mode='handoff')
    folder = final_review_root/'results/stage3/error_analysis'
    for filename, column in [('cluster_names.csv', 'name'), ('error_annotations.csv', 'category')]:
        frame = pd.read_csv(folder/filename, keep_default_na=False)
        frame[column] = ''
        frame.to_csv(folder/filename, index=False)
    status_path = final_review_root/'results/stage3/blind_relabel/status.json'
    status = json.loads(status_path.read_text())
    status.update(responses_received=False, analysis_executed=False)
    status_path.write_text(json.dumps(status))
    clusters, errors = validate_author_review(final_review_root, mode='handoff')
    assert clusters.name.eq('').all() and errors.category.eq('').all()
    assert not validate_blind_analysis(final_review_root, mode='handoff')['analysis_executed']


@pytest.mark.parametrize('change', ['pending', 'hash', 'synthetic', 'numeric'])
def test_final_mode_rejects_unanalysed_or_stale_blind_evidence(final_review_root, change):
    from src.study_io import sha256
    folder = final_review_root/'results/stage3/blind_relabel'
    status = json.loads((folder/'status.json').read_text())
    if change == 'pending':
        status['analysis_executed'] = False
    elif change == 'hash':
        status['responses_sha256'] = '0'*64
    else:
        path = folder/'manuscript_tables/agreement.csv'
        frame = pd.read_csv(path)
        frame.loc[0, 'synthetic' if change == 'synthetic' else 'agreement'] = (
            True if change == 'synthetic' else .97)
        frame.to_csv(path, index=False)
        status['agreement_sha256'] = sha256(path)
    (folder/'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError):
        validate_blind_analysis(final_review_root)


def test_review_tables_preserve_metrics_and_join_external_assignments(final_review_root):
    from src.study_io import sha256
    tables = final_review_root/'results/stage3/error_analysis/manuscript_tables'
    assignment_path = tables/'cluster_assignments.csv'
    assignments = pd.read_csv(assignment_path)
    # VIA filenames may overlap external identifiers; their clusters must be excluded.
    overlapping_via = assignments[assignments.domain.eq('external')].copy()
    overlapping_via['domain'] = 'via'
    overlapping_via['cluster'] = (overlapping_via.cluster + 1) % 12
    pd.concat([assignments, overlapping_via]).to_csv(assignment_path, index=False)
    frozen = [tables/f'{name}.csv' for name in ['cluster_assignments', 'cluster_counts', 'cluster_metrics']]
    before = {path: sha256(path) for path in frozen}
    clusters, errors = validate_author_review(final_review_root)
    generated = build_review_tables(final_review_root, clusters, errors)
    counts = generated['error_category_counts']
    assert counts.groupby('error_type').n.sum().to_dict() == {
        'clear_to_nonclear': 137, 'nonclear_to_clear': 11,
    }
    assert counts.groupby('error_type').percent_within_type.sum().to_numpy() == pytest.approx([100, 100])
    assert counts.loc[counts.category.eq('outro'), 'n'].sum() == 0
    assert generated['error_category_by_cluster'].n.sum() == 148
    named = generated['cluster_summary_named'].set_index('cluster')
    assert named.loc[5, 'external_images'] == 51 and named.loc[5, 'via_images'] == 1
    original = pd.read_csv(tables/'cluster_metrics.csv')
    recorded = original.loc[(original.cluster.eq(5)) &
        original.model.eq('ft_EfficientNetB3') & original.endpoint.eq('accuracy'), 'estimate'].item()
    assert named.loc[5, 'ft_efficientnetb3_accuracy'] == recorded
    assert {path: sha256(path) for path in frozen} == before
    for name, frame in generated.items():
        assert frame.analysis_scope.eq('exploratory/post hoc').all()
        assert (tables/f'{name}.csv').exists() and (tables/f'{name}.tex').exists()


@pytest.mark.parametrize('mode', ['final', 'handoff'])
def test_report_modes_build_and_reproduce_without_touching_source(final_review_root, monkeypatch, mode):
    import scripts.build_stage3_report as report
    from src.study_io import sha256
    root = final_review_root
    for name in [
        'results/stage3/progress.json', 'results/stage3/protected_verification.json',
        'results/stage3/registration.json', 'results/stage3/dry_run.json',
        'results/stage3/verification/pytest.log', 'results/stage3/verification/smoke.json',
        'results/external_test/predictions.csv', 'results/external_test/final_models.json',
    ]:
        destination = root/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT/name, destination)
    for job in range(40):
        marker = root/f'results/stage3/grouped_cv/jobs/{job}/complete.json'
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text('{}')
    mosaics = root/'results/stage3/error_analysis/mosaics'
    mosaics.mkdir()
    for cluster in range(12):
        (mosaics/f'cluster_{cluster:02d}.jpg').touch()
    (root/'docs').mkdir()
    if mode == 'handoff':
        folder = root/'results/stage3/error_analysis'
        for filename, column in [('cluster_names.csv', 'name'), ('error_annotations.csv', 'category')]:
            frame = pd.read_csv(folder/filename, keep_default_na=False)
            frame[column] = ''
            frame.to_csv(folder/filename, index=False)
        path = root/'results/stage3/blind_relabel/status.json'
        status = json.loads(path.read_text())
        status.update(responses_received=False, analysis_executed=False)
        path.write_text(json.dumps(status))
    monkeypatch.setattr(report, 'ROOT', root)
    monkeypatch.setattr(report, 'read', lambda folder, name: pd.read_csv(
        PROJECT_ROOT/'results/stage3'/folder/'manuscript_tables'/f'{name}.csv'))
    kwargs = {} if mode == 'final' else {'mode': mode}
    report.build(**kwargs)  # Default mode must be final.
    destination = root/'docs/STAGE3_REPORT.md'
    contents = destination.read_text()
    metadata = json.loads((root/'results/stage3/verification/completed.json').read_text())
    assert metadata['report_mode'] == mode
    assert metadata['real_blind_responses_analysed'] == (mode == 'final')
    assert metadata['cluster_name_fields_filled'] == (12 if mode == 'final' else 0)
    assert metadata['error_categories_filled'] == (148 if mode == 'final' else 0)
    if mode == 'final':
        assert 'Regra de rotulagem a explicitar no artigo' in contents
        assert '98/100' in contents and '0.959612' in contents
        assert 'aguardando Fabricio Breve' not in contents
    else:
        assert 'aguardando Fabricio Breve' in contents
        assert 'nome em branco' in contents and 'categoria em branco' in contents
    before = sha256(destination)
    report.build(**kwargs)
    assert sha256(destination) == before
