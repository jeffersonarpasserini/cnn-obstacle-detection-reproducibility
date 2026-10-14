#!/usr/bin/env python3
"""Post-test threshold, image, scene and error descriptors; human fields stay blank."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ks_2samp,norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.stage3_io import SCOPE,inputs,panels,table,via_inputs,image_paths,embeddings
from src.stage3_images import attributes,cosine_nearest,mosaic,human_review_templates
from src.stage3_statistics import oracle_threshold,oracle_metric_draws,cluster_logistic,intervals
from src.study_features import CanonicalFeatureStore
from src.study_io import write_json,sha256
from scripts.analyze_stage3 import weights,metric_rows


def image_descriptors(output,smoke,external_names):
    folder=output/'error_analysis'; folder.mkdir(parents=True,exist_ok=True)
    path=folder/'image_attributes.csv'
    if path.exists(): return pd.read_csv(path)
    via,labels,external,manifest=image_paths()
    external=[p for p in external if p.name in set(external_names)]
    if smoke: via_for_images=via[:16]
    else: via_for_images=via
    v=pd.DataFrame([attributes(p) for p in via_for_images]).assign(domain='via')
    e=pd.DataFrame([attributes(p) for p in external]).assign(domain='external')
    # Threshold was fixed from ALL VIA before any stage-three error analysis.
    registration=json.loads((ROOT/'results/stage3/registration.json').read_text())
    threshold=registration['low_light_threshold']
    all_names,dino=embeddings(); positions={n:i for i,n in enumerate(all_names)}
    dv=dino[[positions[p.name] for p in via]]; de=dino[[positions[p.name] for p in external]]
    v['distance_dinov2']=cosine_nearest(dv,dv,exclude_self=True)[:len(v)]
    e['distance_dinov2']=cosine_nearest(de,dv)
    store=CanonicalFeatureStore(via,device='cpu',threads=2)
    ordered=pd.read_csv(ROOT/'results/learning_curve/split.csv').sort_values('filename').filename.tolist()
    external_positions={name:i for i,name in enumerate(ordered)}
    provenance=[]
    for architecture in ('EfficientNetB3','MobileNet','ResNet50','ResNet101V2'):
        base=store.get(architecture)
        cache=ROOT/'results/learning_curve/feature_cache'/architecture
        identity=json.loads((cache/'identity.json').read_text())
        recorded_names=[item['filename'] for item in identity['ordered_pixels']]
        if recorded_names!=ordered: raise ValueError('Frozen external feature order changed')
        complete=json.loads((cache/'complete.json').read_text())
        # finish_job stores output digests in the completed marker.
        array=np.load(cache/'features.npy',mmap_mode='r')
        if sha256(cache/'features.npy')!=complete['artifacts']['features.npy'] or complete['identity']!=identity:
            raise ValueError('Frozen external feature cache digest/identity changed')
        if array.shape[0]!=len(ordered) or array.shape[1]!=base.shape[1]: raise ValueError('Frozen feature shape changed')
        selected=array[[external_positions[p.name] for p in external]]
        v[f'distance_{architecture}']=cosine_nearest(base,base,exclude_self=True)[:len(v)]
        e[f'distance_{architecture}']=cosine_nearest(selected,base)
        provenance.append({'architecture':architecture,'via_features_sha256':store.provenance[architecture]['features_sha256'],
            'external_features_sha256':sha256(cache/'features.npy'),'encoder_sha256':identity['encoder_sha256'],
            'new_inference_jobs':0})
    result=pd.concat([v,e],ignore_index=True)
    result['low_light']=result.luminance_mean<threshold; result['analysis_scope']=SCOPE
    result.to_csv(path,index=False); table(result,folder,'image_attributes')
    write_json(folder/'attribute_provenance.json',{'analysis_scope':SCOPE,'low_light_threshold':threshold,
        'DINOv2_sha256':sha256(ROOT/'external-data/scene_union/embeddings.npz'),'features':provenance,
        'computed_without_labels':True,'via_nearest_neighbor':'leave self out','no_new_deep_forwards':True})
    return result


def analyze(output,iterations=10000,smoke=False):
    folder=output/'error_analysis'; folder.mkdir(parents=True,exist_ok=True)
    frame,manifest,truth,groups=inputs(smoke); names=manifest.index.tolist()
    via=via_inputs(); w=weights(output,truth,groups,iterations)
    threshold_rows=[]; shift_rows=[]; score_rows=[]; ks_rows=[]; calibration=[]
    descriptors=image_descriptors(output,smoke,names)
    current=descriptors[descriptors.domain.eq('external')].set_index('filename').loc[names]
    feature_columns=[column for column in current if column not in ('domain','low_light','analysis_scope')]
    quintile_rows=[]; quintile_bounds=[]; regression=[]; diagnostics=[]; low_light_rows=[]
    for name,panel,d,s in panels(frame):
        source=via[via.model.eq(name)]
        selected_oracles=[]; oracle_boots=[]; platt_scores=[]
        for seed,score,decision in zip(sorted(panel.seed.unique()),s,d):
            threshold,accuracy=oracle_threshold(truth,score)
            oracle=(score>=threshold).astype(int)
            selected_oracles.append(oracle); oracle_boots.append(oracle_metric_draws(truth,score,w))
            point_rows=metric_rows(name,truth,decision,score,w,seed=int(seed),decision='frozen default')
            oracle_rows=metric_rows(name,truth,oracle,score,w,seed=int(seed),decision='external accuracy oracle, not deployable')
            for row in oracle_rows:
                if row['endpoint'] in oracle_boots[-1]:
                    row['ci_low'],row['ci_high']=np.quantile(oracle_boots[-1][row['endpoint']],[.025,.975])
            threshold_rows+=point_rows+oracle_rows
            threshold_rows.append({'model':name,'seed':int(seed),'endpoint':'threshold_oracle',
                'estimate':threshold,'oracle_accuracy':accuracy,'default_accuracy':float(np.mean(decision==truth)),
                'oracle_minus_default_accuracy':accuracy-float(np.mean(decision==truth)),
                    'notice':'Oracle thresholds are reselected in scene bootstrap; not deployable'})
            original=source[source.seed.eq(seed)].sort_values('filename')
            for domain,t,pred in [('via_oof',original.true_label.to_numpy(),original.predicted_label_default.to_numpy()),
                                  ('external',truth,decision)]:
                shift_rows.append({'model':name,'seed':int(seed),'domain':domain,'predicted_obstructed_fraction':float(np.mean(pred==0)),
                    'observed_obstructed_fraction':float(np.mean(t==0)),'images':len(t)})
            for label in (0,1):
                a=original[original.true_label.eq(label)].score_clear.to_numpy(); b=score[truth==label]
                for domain,values in [('via_oof',a),('external',b)]:
                    score_rows.append({'model':name,'seed':int(seed),'class':label,'domain':domain,
                        'count':len(values),'mean':float(values.mean()),'p01':float(np.quantile(values,.01)),
                        'p10':float(np.quantile(values,.10)),'p25':float(np.quantile(values,.25)),
                        'p50':float(np.quantile(values,.50)),'p75':float(np.quantile(values,.75)),
                        'p90':float(np.quantile(values,.90)),'p99':float(np.quantile(values,.99))})
                result=ks_2samp(a,b)
                ks_rows.append({'model':name,'seed':int(seed),'class':label,'ks_statistic':result.statistic,
                    'p_value':result.pvalue,'notice':'Descriptive two-sample KS; images within scenes are correlated'})
            if name.startswith('ft_'):
                adapter=LogisticRegression(C=1.,max_iter=500,random_state=1980)
                train_score=logit(np.clip(original.score_clear.to_numpy(),1e-7,1-1e-7))[:,None]
                adapter.fit(train_score,original.true_label)
                calibrated=adapter.predict_proba(logit(np.clip(score,1e-7,1-1e-7))[:,None])[:,1]
                calibration+=metric_rows(name,truth,(calibrated>=.5).astype(int),calibrated,w,
                    seed=int(seed),analysis='FT VIA-only Platt sensitivity')
                calibration.append({'model':name,'seed':int(seed),'endpoint':'brier','estimate':brier_score_loss(truth,calibrated),
                    'unrecalibrated_brier':brier_score_loss(truth,score),'fit_source':'342 VIA OOF only',
                    'platt_intercept':float(adapter.intercept_[0]),'platt_slope':float(adapter.coef_[0,0])})
                platt_scores.append(calibrated)
            else:
                calibration+=metric_rows(name,truth,decision,score,w,seed=int(seed),analysis='existing frozen VIA adapter')
                platt_scores.append(score)
            z=(current[feature_columns]-current[feature_columns].mean())/current[feature_columns].std(ddof=0)
            x=np.column_stack([z.to_numpy(),truth])
            beta,se,diagnostic=cluster_logistic(x,(decision==truth).astype(int),groups)
            diagnostics.append({'model':name,'seed':int(seed),**diagnostic})
            for term,value,error in zip(['intercept',*feature_columns,'class_clear'],beta,se):
                regression.append({'model':name,'seed':int(seed),'term':term,'coefficient':float(value),
                    'robust_se':float(error),'ci_low':float(value-1.95996398454*error),'ci_high':float(value+1.95996398454*error),
                    'odds_ratio':float(np.exp(np.clip(value,-700,700))),
                    'p_value':float(2*norm.sf(abs(value)/error)) if error else np.nan,
                    'notice':'Descriptive association; attributes standardised externally, class unscaled; not causal'})
        platt_scores=np.asarray(platt_scores)
        calibration+=metric_rows(name,truth,(platt_scores>=.5).astype(int),platt_scores,w,
            seed='mean',analysis='individual-seed metric mean; VIA-only calibration sensitivity')
        brier_draws=np.mean([w@((p-truth)**2)/w.sum(axis=1) for p in platt_scores],axis=0)
        low,high=np.quantile(brier_draws,[.025,.975])
        calibration.append({'model':name,'seed':'mean','endpoint':'brier',
            'estimate':float(np.mean((platt_scores-truth)**2)),'ci_low':low,'ci_high':high,
            'analysis':'individual-seed metric mean; VIA-only calibration sensitivity'})
        for low_light in (False,True):
            selected=current.low_light.to_numpy()==low_light
            if not selected.any(): continue
            for endpoint,value in intervals(truth[selected],d[:,selected],w[:,selected]).items():
                if endpoint in ('accuracy','clear_recall','obstructed_recall','obstructed_false_negative_rate'):
                    low_light_rows.append({'model':name,'low_light':low_light,'images':int(selected.sum()),
                        'endpoint':endpoint,**value,'threshold_source':'Fixed VIA luminance P10 before error analysis'})
        threshold_rows+=metric_rows(name,truth,d,s,w,seed='mean',analysis='individual-seed metric mean',decision='frozen default')
        selected_oracles=np.asarray(selected_oracles)
        points=intervals(truth,selected_oracles,np.ones((1,len(truth))),s)
        for endpoint in oracle_boots[0]:
            draws=np.mean([r[endpoint] for r in oracle_boots],axis=0); low,high=np.quantile(draws,[.025,.975])
            threshold_rows.append({'model':name,'seed':'mean','analysis':'individual-seed metric mean',
                'decision':'external accuracy oracle, not deployable','endpoint':endpoint,
                'estimate':points[endpoint]['estimate'],'ci_low':low,'ci_high':high,
                'notice':'External threshold reselected per bootstrap draw; oracle training accuracy'})
        if len(s)>1:
            mean_score=s.mean(axis=0)
            threshold_rows+=metric_rows(name,truth,(mean_score>=.5).astype(int),mean_score,w,seed='ensemble',
                analysis='three-seed ensemble sensitivity',decision='frozen default')
        for column in feature_columns:
            edges=np.unique(np.quantile(current[column],[0,.2,.4,.6,.8,1]))
            bins=np.digitize(current[column],edges[1:-1],right=True)
            if name==sorted(frame.model.unique())[0]:
                quintile_bounds.extend({'attribute':column,'quantile_index':j,'boundary':float(value),
                    'source':'external descriptive quintiles, held fixed in resampling'} for j,value in enumerate(edges))
            for q in sorted(set(bins)):
                selected=bins==q
                for endpoint,value in intervals(truth[selected],d[:,selected],w[:,selected]).items():
                    if endpoint not in ('accuracy','clear_recall','obstructed_recall','obstructed_false_negative_rate'): continue
                    quintile_rows.append({'model':name,'attribute':column,'quintile':int(q+1),'images':int(selected.sum()),
                        'clear':int(np.sum(truth[selected]==1)),'nonclear':int(np.sum(truth[selected]==0)),
                        'endpoint':endpoint,**value})
    table(threshold_rows,folder,'default_oracle_metrics'); table(shift_rows,folder,'predicted_obstructed_fraction')
    table(score_rows,folder,'score_distributions'); table(ks_rows,folder,'kolmogorov_smirnov')
    table(calibration,folder,'via_only_platt'); table(quintile_rows,folder,'attribute_quintiles')
    table(quintile_bounds,folder,'quintile_boundaries'); table(regression,folder,'cluster_robust_logistic')
    table(diagnostics,folder,'logistic_diagnostics')
    table(low_light_rows,folder,'low_light_metrics')
    q=pd.DataFrame(quintile_rows); ranking=q[q.endpoint.eq('accuracy')].groupby(['attribute','model']).estimate.agg(['min','max'])
    ranking=(ranking['max']-ranking['min']).groupby('attribute').mean().sort_values(ascending=False)
    table(ranking.rename('mean_accuracy_range').reset_index(),folder,'descriptive_attribute_ranking')
    table(q[q.attribute.isin(ranking.head(3).index)],folder,'most_informative_quintiles')
    # Visual clusters use no classes/errors during fitting.
    embedding_names,embedding=embeddings(); positions={n:i for i,n in enumerate(embedding_names)}
    external_embedding=embedding[[positions[n] for n in names]]
    via_paths,via_truth,_,_=image_paths(); via_embedding=embedding[[positions[p.name] for p in via_paths]]
    kmeans=KMeans(n_clusters=12,random_state=1980,n_init=20).fit(external_embedding)
    clusters=kmeans.labels_; via_clusters=kmeans.predict(via_embedding)
    assignments=pd.DataFrame({'filename':names,'cluster':clusters,'domain':'external'})
    assignments=pd.concat([assignments,pd.DataFrame({'filename':[p.name for p in via_paths],'cluster':via_clusters,'domain':'via'})])
    table(assignments,folder,'cluster_assignments'); counts=[]; cluster_metrics=[]; rng=np.random.default_rng(1980); mosaic_index=[]
    for cluster in range(12):
        selected=clusters==cluster; vv=via_clusters==cluster
        counts.append({'cluster':cluster,'external_images':int(selected.sum()),'via_images':int(vv.sum()),
            'external_clear_fraction':float(np.mean(truth[selected]==1)),
            'via_clear_fraction':float(np.mean(via_truth[vv]==1)) if vv.any() else np.nan})
        for name,panel,d,s in panels(frame):
            for endpoint,value in intervals(truth[selected],d[:,selected],w[:,selected]).items():
                if endpoint in ('accuracy','clear_recall','obstructed_recall'):
                    cluster_metrics.append({'model':name,'cluster':cluster,'endpoint':endpoint,**value})
        images=np.flatnonzero(selected); chosen=rng.choice(images,min(16,len(images)),replace=False)
        rows=[{'path':ROOT/'external-data/via-dataset-extended'/names[i],'identifier':manifest.iloc[i].sha256[:12]} for i in chosen]
        mosaic(rows,folder/'mosaics'/f'cluster_{cluster:02d}.jpg')
        mosaic_index.extend({'cluster':cluster,'filename':names[i],'identifier':manifest.iloc[i].sha256[:12]} for i in chosen)
    table(counts,folder,'cluster_counts'); table(cluster_metrics,folder,'cluster_metrics'); table(mosaic_index,folder,'cluster_mosaic_index')
    fixed=[d[0] for name,panel,d,s in panels(frame) if not name.startswith('ft_')]
    ft=frame[frame.model.eq('ft_EfficientNetB3')]
    ft_decisions=ft.pivot(index='filename',columns='seed',values='predicted_label_default').loc[names].to_numpy().T
    ft_scores=ft.pivot(index='filename',columns='seed',values='score_clear').loc[names].to_numpy().T
    majority=(ft_decisions.mean(axis=0)>=.5).astype(int)
    consensus=np.all(np.asarray(fixed)!=truth,axis=0)&(majority!=truth)
    fixed_scores=[s[0] for name,panel,d,s in panels(frame) if not name.startswith('ft_')]
    error_confidence=np.mean(np.vstack([fixed_scores,np.median(ft_scores,axis=0)]),axis=0)
    error_confidence=np.where(truth==0,error_confidence,1-error_confidence)
    errors=[]; error_details=[]
    for label,error_type in [(1,'clear_to_nonclear'),(0,'nonclear_to_clear')]:
        selected=np.flatnonzero(consensus&(truth==label)); selected=selected[np.argsort(-error_confidence[selected],kind='stable')]
        for i in selected:
            errors.append({'filename':names[i],'error_type':error_type})
            error_details.append({'filename':names[i],'error_type':error_type,'confidence_in_error':float(error_confidence[i]),
                                  'true_label':int(truth[i]),'prediction':int(1-truth[i]),'identifier':manifest.iloc[i].sha256[:12]})
        top=selected[:60]
        rows=[{'path':ROOT/'external-data/via-dataset-extended'/names[i]} for i in top]
        titles=[manifest.iloc[i].sha256[:12]+f'\ntruth={"clear" if truth[i] else "nonclear"}; pred={"nonclear" if truth[i] else "clear"}' for i in top]
        mosaic(rows,folder/'mosaics'/f'consensus_{error_type}.jpg',titles=titles,columns=6)
    table(error_details,folder,'consensus_errors')
    human_review_templates(folder,range(12),errors)
    categories=json.loads((ROOT/'configs/stage3.json').read_text())['error_categories']
    (folder/'AUTHOR_REVIEW.md').write_text('# exploratory/post hoc — revisão do autor\n\n'
        'Preencha cluster_names.csv e error_annotations.csv. Nomes de clusters e categorias estão vazios. '
        'Nenhum rótulo foi alterado. Os mosaicos de clusters não mostram classes ou nomes originais; '
        'mosaicos de consenso mostram verdade/previsão e identificador por hash.\n\n'
        'Categorias permitidas: '+', '.join(categories)+'.\n\n'
        'Consenso usa decisões padrão congeladas dos quatro fixos e maioria das seeds de EfficientNetB3. '
        'Confiança no erro: média dos quatro escores direcionados ao erro e da mediana das três seeds FT. '
        'Mostram-se no máximo 60 por tipo; o CSV inclui todos os erros de consenso. '
        'Acurácia-oráculo e atributos mais informativos são escolhas externas pós-teste, sem uso implantável.\n')
