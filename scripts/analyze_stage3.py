#!/usr/bin/env python3
"""Ordered post-test analyses on frozen data; no principal inference or retuning."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from src.stage3_io import SCOPE,configuration,inputs,panels,table,via_inputs,image_paths,check_protected
from src.stage3_statistics import (bootstrap_weights,metric_draws,intervals,auc_draws,
    specificity_draws,partial_auc,partial_auc_draws,roc,delong_contrast)
from src.study_io import sha256,write_json,finish_job,completed_job,checked_identity
from src.study_statistics import substitution_status,wilson
from src.artifact import prediction_metrics,continuous_scores
from scripts.analyze_fine_tuning import mcnemar_exact
from scripts.build_manuscript_tables import holm_adjust


def weights(output,truth,groups,iterations,key='external'):
    folder=output/'_cache'; folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{key}_{iterations}.npy'
    if not path.exists():
        np.save(path,bootstrap_weights(truth,groups,iterations))
    matrix=np.load(path,mmap_mode='r')
    if matrix.shape!=(iterations,len(truth)): raise ValueError('Post-test bootstrap cache shape changed')
    return matrix


def metric_rows(name,truth,decisions,scores,w,**extra):
    return [{'model':name,'endpoint':endpoint,**value,**extra}
            for endpoint,value in intervals(truth,decisions,w,scores).items()]


def paired_rows(first,second,truth,fd,sd,w,**extra):
    a,b=metric_draws(truth,fd,w),metric_draws(truth,sd,w)
    pa,pb=metric_draws(truth,fd,np.ones((1,len(truth)))),metric_draws(truth,sd,np.ones((1,len(truth))))
    out=[]
    for key in ('accuracy','obstructed_recall','obstructed_false_negative_rate'):
        low,high=np.quantile(a[key]-b[key],[.025,.975])
        out.append({'fixed':first,'fine_tuning':second,'endpoint':key,'difference':float(pa[key][0]-pb[key][0]),
            'ci_low':float(low),'ci_high':float(high),
            'substitution':substitution_status(low,high,lower_is_better=key.endswith('negative_rate')),**extra})
    return out


def fold_endpoint_values(frame,endpoint):
    # The historical full-search table predates its explicitly named FNR column.
    if endpoint=='obstructed_false_negative_rate' and endpoint not in frame:
        return 1-frame['obstructed_recall']
    return frame[endpoint]


def native_sensitivity(output,iterations,smoke):
    folder=output/'native_sensitivity'; frame,manifest,truth,groups=inputs(smoke)
    w=weights(output,truth,groups,iterations); rows=[]; comparison=[]; native={}; neural={}
    expected={'a':(.80901,61),'b':(.8128173,12),'c':(.8077411,34),'d':(.8153553,39)}
    for name,panel,d,s in panels(frame):
        if panel.family.iloc[0]!='fixed': neural[name]=d; continue
        panel=panel.sort_values('filename'); nd=(panel.native_score.to_numpy()>0).astype(int)[None,:]
        native[name]=nd
        for kind,decision,ranking in [('native',nd,panel.native_score.to_numpy()[None,:]),('frozen_adapter',d,s)]:
            rows+=metric_rows(name,truth,decision,ranking,w,decision=kind,
                auc_score_source='native_score' if kind=='native' else 'score_clear')
            for endpoint,successes in [('obstructed_recall',int(np.sum((truth==0)&(decision[0]==0)))),
                    ('obstructed_false_negative_rate',int(np.sum((truth==0)&(decision[0]==1))))]:
                lo,hi=wilson(successes,int(np.sum(truth==0)))
                rows.append({'model':name,'endpoint':endpoint,'estimate':successes/np.sum(truth==0),
                             'ci_low':lo,'ci_high':hi,'decision':kind,'method':'Wilson'})
        a,b,p=mcnemar_exact(truth,nd[0],d[0]); difference=int(np.sum(nd!=d))
        accuracy=float(np.mean(nd[0]==truth)); reference,count=expected[name[0]]
        if not smoke and (difference!=count or abs(accuracy-reference)>1e-5):
            raise ValueError('Native sensitivity differs from the author verification values')
        comparison.append({'model':name,'native_accuracy':accuracy,'adapter_accuracy':float(np.mean(d[0]==truth)),
            'disagreements':difference,'native_only_correct':a,'adapter_only_correct':b,'mcnemar_p':p,
            'expected_native_accuracy':reference,'expected_disagreements':count})
    table(rows,folder,'native_adapter_metrics'); table(comparison,folder,'native_adapter_comparison')
    pairs=[]
    for first,fd in native.items():
        for second,sd in neural.items(): pairs+=paired_rows(first,second,truth,fd,sd,w,decision='native sensitivity')
    table(pairs,folder,'native_substitution')


def grouped_cv(output,iterations,smoke):
    import joblib
    from src.study_features import CanonicalFeatureStore
    from src.fine_tuning import fit_fixed_pipeline,transform_fixed
    paths,truth,_,_=image_paths(); names=np.asarray([p.name for p in paths])
    group_table=pd.read_csv(ROOT/'data/groups.csv').set_index('filename').loc[names]
    groups=group_table.group_id.to_numpy()
    if sha256(ROOT/'data/groups.csv')!='30537852c34f3d94e9dfeac436a62991d69561f2c28a3d4a3cceb4fe56919c26':
        raise ValueError('VIA groups changed')
    folds=list(StratifiedGroupKFold(10,shuffle=True,random_state=1980).split(np.zeros(len(truth)),truth,groups))
    expected=pd.read_csv(ROOT/'results/scene_groups/fold_diagnostics.csv')
    diagnostics=[]
    for fold,(_,test) in enumerate(folds,1):
        row={'fold':fold,'test_image_count':len(test),'clear':int(np.sum(truth[test]==1)),
             'nonclear':int(np.sum(truth[test]==0)),'groups_disjoint':True}
        reference=expected[expected.fold.eq(fold)].iloc[0]
        if any(row[key]!=reference[key] for key in ('test_image_count','clear','nonclear')):
            raise ValueError('Grouped folds do not reproduce frozen scene diagnostics')
        diagnostics.append(row)
    folder=output/'grouped_cv'; folder.mkdir(parents=True,exist_ok=True)
    table(diagnostics,folder,'fold_diagnostics')
    assignments=np.empty(len(truth),int)
    for fold,(_,test) in enumerate(folds,1): assignments[test]=fold
    table(pd.DataFrame({'filename':names,'group_id':groups,'test_fold':assignments}),folder,'fold_assignments')
    experiments=json.loads((ROOT/'configs/article_selected_models_10fold.json').read_text())['experiments']
    store=CanonicalFeatureStore(paths,device='cpu',threads=2)
    predictions=[]; metrics=[]; costs=[]
    for experiment in experiments:
        matrices=[store.get(a) for a in experiment['extractors']]
        for fold,(train,test) in enumerate(folds[:1] if smoke else folds,1):
            spec=dict(experiment)
            if smoke and spec['reduction']=='pca': spec['components']=4
            if set(groups[train])&set(groups[test]): raise ValueError('Grouped refit scene leakage')
            directory=folder/'jobs'/f'{spec["name"]}__fold_{fold:02d}'
            identity={'experiment':spec,'seed':1980,'train':train.tolist(),'test':test.tolist(),
                'groups_sha256':sha256(ROOT/'data/groups.csv'),'canonical_source':sha256(ROOT/'src/fixed_refit.py'),
                'pipeline_source':sha256(ROOT/'src/fine_tuning.py'),'smoke':smoke}
            checked_identity(directory,identity)
            if not completed_job(directory,identity):
                started=perf_counter()
                pipeline,_,reduction,training,_=fit_fixed_pipeline(spec,[x[train] for x in matrices],truth[train],1980)
                prepared=transform_fixed(pipeline,[x[test] for x in matrices])
                decision=pipeline['classifier'].predict(prepared); score=continuous_scores(pipeline['classifier'],prepared)
                joblib.dump(pipeline,directory/'pipeline.joblib')
                restored=joblib.load(directory/'pipeline.joblib')
                if not np.array_equal(decision,restored['classifier'].predict(transform_fixed(restored,[x[test] for x in matrices]))):
                    raise ValueError('Grouped pipeline native reload changed decisions')
                pd.DataFrame({'filename':names[test],'true_label':truth[test],'group_id':groups[test],
                    'model':spec['name'],'fold':fold,'predicted_label':decision,'score_clear':score,
                    'analysis_scope':SCOPE}).to_csv(directory/'predictions.csv',index=False)
                write_json(directory/'cost.json',{'model':spec['name'],'fold':fold,'reduction_seconds':reduction,
                    'training_seconds':training,'elapsed_seconds':perf_counter()-started,'analysis_scope':SCOPE,
                    'native_reload_identical':True,'new_backbone_forwards':0})
                finish_job(directory,identity,['pipeline.joblib','predictions.csv','cost.json'])
            panel=pd.read_csv(directory/'predictions.csv',float_precision='round_trip')
            predictions.append(panel); costs.append(json.loads((directory/'cost.json').read_text()))
            metrics.append({'model':spec['name'],'fold':fold,**prediction_metrics(panel.true_label,panel.predicted_label,panel.score_clear)})
            print(f'Grouped {spec["name"]}/fold{fold} completed; frozen canonical features',flush=True)
    table(metrics,folder,'per_fold_metrics'); table(costs,folder,'costs')
    pred=pd.concat(predictions); pred.to_csv(folder/'predictions.csv',index=False)
    if smoke: return
    if len(pred)!=4*342 or pred.duplicated(['model','filename']).any(): raise ValueError('Grouped OOF coverage mismatch')
    original=pd.read_csv(ROOT/'results/fine_tuning_recovered/fixed_per_image_predictions.csv',float_precision='round_trip')
    w=weights(output,truth,None,iterations,'via_images'); summary=[]; paired=[]; pair_rows=[]
    original_folds=pd.read_csv(ROOT/'results/full_search/fold_assignments.csv').set_index('filename').test_fold
    paired_groups=[g for g,rows in group_table.reset_index().groupby('group_id') if len(rows)>1]
    for name,panel in pred.groupby('model'):
        panel=panel.set_index('filename').loc[names]; old=original[original.experiment.eq(name)].set_index('filename').loc[names]
        ds,oscores=panel.predicted_label.to_numpy(),panel.score_clear.to_numpy()
        od,oldscore=old.predicted_label.to_numpy(),old.score_clear.to_numpy()
        newdraws=metric_draws(truth,ds,w,oscores); olddraws=metric_draws(truth,od,w,oldscore)
        for kind,d,s in [('grouped',ds,oscores),('original',od,oldscore)]:
            original_fold_values=pd.DataFrame(metrics).query('model==@name') if kind=='grouped' else pd.read_csv(ROOT/'results/full_search/per_fold_metrics.csv').query('experiment==@name')
            for row in metric_rows(name,truth,d,s,w,design=kind):
                if row['endpoint'] in ('accuracy','balanced_accuracy','mcc','roc_auc','obstructed_recall','obstructed_false_negative_rate'):
                    values=fold_endpoint_values(original_fold_values,row['endpoint'])
                    row.update(mean_fold=float(values.mean()),sd_fold=float(values.std(ddof=1)))
                    summary.append(row)
            for endpoint,count in [('obstructed_recall',int(np.sum((truth==0)&(d==0)))),('obstructed_false_negative_rate',int(np.sum((truth==0)&(d==1))))]:
                low,high=wilson(count,int(np.sum(truth==0)))
                summary.append({'model':name,'design':kind,'endpoint':endpoint,'estimate':count/np.sum(truth==0),
                    'ci_low':low,'ci_high':high,'method':'Wilson pooled proportion'})
        for endpoint in ('accuracy','balanced_accuracy','mcc','roc_auc','obstructed_recall','obstructed_false_negative_rate'):
            diff=newdraws[endpoint]-olddraws[endpoint]; points=prediction_metrics(truth,ds,oscores),prediction_metrics(truth,od,oldscore)
            paired.append({'model':name,'endpoint':endpoint,'difference':points[0][endpoint]-points[1][endpoint],
                           'ci_low':float(np.quantile(diff,.025)),'ci_high':float(np.quantile(diff,.975))})
        for group in paired_groups:
            pair=group_table[group_table.group_id.eq(group)].index.tolist()
            for filename in pair:
                peer=next(p for p in pair if p!=filename)
                pair_rows.append({'model':name,'group_id':group,'filename':filename,'peer':peer,
                    'original_test_fold':int(original_folds[filename]),'peer_original_test_fold':int(original_folds[peer]),
                    'opposite_original_train_test':bool(original_folds[filename]!=original_folds[peer]),
                    'grouped_test_fold':int(panel.loc[filename,'fold']),'true_label':int(panel.loc[filename,'true_label']),
                    'original_prediction':int(old.loc[filename,'predicted_label']),
                    'grouped_prediction':int(panel.loc[filename,'predicted_label']),
                    'prediction_changed':bool(old.loc[filename,'predicted_label']!=panel.loc[filename,'predicted_label'])})
    table(summary,folder,'grouped_original_metrics'); table(paired,folder,'grouped_minus_original'); table(pair_rows,folder,'four_pairs')
    write_json(folder/'canonical_feature_provenance.json',store.provenance)


def roc_analysis(output,iterations,smoke):
    folder=output/'roc_analysis'; frame,manifest,truth,groups=inputs(smoke)
    w=weights(output,truth,groups,iterations); aucs={}; specificity={}; scores={}; rows=[]; curves=[]; partial=[]
    targets=(.9,.95,.97)
    for name,panel,d,s in panels(frame):
        scores[name]=s; individual_auc=[auc_draws(truth,score,w) for score in s]
        aucs[name]=np.mean(individual_auc,axis=0)
        specificity[name]=np.mean([specificity_draws(truth,score,w,targets) for score in s],axis=0)
        value=float(np.mean([auc_draws(truth,score,np.ones((1,len(truth))))[0] for score in s]))
        low,high=np.quantile(aucs[name],[.025,.975]); rows.append({'model':name,'analysis':'individual-seed metric mean' if len(s)>1 else 'single fixed',
            'endpoint':'roc_auc','estimate':value,'ci_low':low,'ci_high':high})
        partial_draws=[]; partial_points=[]
        for seed,score,draw in zip(sorted(panel.seed.unique()),s,individual_auc):
            fpr,tpr,cut=roc(truth,score)
            curves.extend({'model':name,'seed':int(seed),'false_positive_rate':a,'obstructed_recall':b,
                'clear_probability_boundary':float(-c),'analysis':'individual seed'} for a,b,c in zip(fpr,tpr,cut))
            low,high=np.quantile(draw,[.025,.975]); rows.append({'model':name,'seed':int(seed),'analysis':'individual seed',
                'endpoint':'roc_auc','estimate':float(auc_draws(truth,score,np.ones((1,len(truth))))[0]),'ci_low':low,'ci_high':high})
            pdraw=partial_auc_draws(truth,score,w); low,high=np.quantile(pdraw,[.025,.975])
            partial_draws.append(pdraw); partial_points.append(partial_auc(truth,score))
            partial.append({'model':name,'seed':int(seed),'partial_auc_mcclish':partial_auc(truth,score),'ci_low':low,'ci_high':high,
                'focus':'obstructed sensitivity [0.90,1.00]','oracle':'post-test descriptive, not deployable'})
        low,high=np.quantile(np.mean(partial_draws,axis=0),[.025,.975])
        partial.append({'model':name,'seed':'mean','analysis':'individual-seed metric mean',
            'partial_auc_mcclish':float(np.mean(partial_points)),'ci_low':low,'ci_high':high,
            'focus':'obstructed sensitivity [0.90,1.00]','oracle':'post-test descriptive, not deployable'})
        if len(s)>1:
            grid=np.linspace(0,1,1001)
            vertical=np.mean([np.interp(grid,*roc(truth,score)[:2]) for score in s],axis=0)
            curves.extend({'model':name,'seed':'mean','false_positive_rate':a,'obstructed_recall':b,
                'analysis':'vertical mean of individual seed ROCs'} for a,b in zip(grid,vertical))
        point=np.mean([specificity_draws(truth,score,np.ones((1,len(truth))),targets)[0] for score in s],axis=0)
        for j,target in enumerate(targets):
            low,high=np.quantile(specificity[name][:,j],[.025,.975])
            rows.append({'model':name,'endpoint':'clear_specificity','recall_target':target,'estimate':point[j],
                'ci_low':low,'ci_high':high,'oracle':'external threshold, not deployable'})
    differences=[]; delong=[]; specpairs=[]
    fixed=[n for n in scores if not n.startswith('ft_')]; ft=[n for n in scores if n.startswith('ft_')]
    for a in fixed:
        for b in ft:
            difference=aucs[a]-aucs[b]; low,high=np.quantile(difference,[.025,.975])
            original=float(np.mean([auc_draws(truth,s,np.ones((1,len(truth))))[0] for s in scores[a]])-
                           np.mean([auc_draws(truth,s,np.ones((1,len(truth))))[0] for s in scores[b]]))
            # Null-centred paired bootstrap, with a finite Monte Carlo correction.
            p=(1+np.sum(np.abs(difference-original)>=abs(original)))/(iterations+1)
            differences.append({'fixed':a,'fine_tuning':b,'difference':original,'ci_low':low,'ci_high':high,'p_value':p})
            contrast=np.r_[1.,np.full(len(scores[b]),-1/len(scores[b]))]
            delong.append({'fixed':a,'fine_tuning':b,**delong_contrast(truth,np.vstack([scores[a],scores[b]]),contrast),
                'notice':'Sensitivity: image independence assumed; principal inference uses scene bootstrap'})
            point_a=np.mean([specificity_draws(truth,s,np.ones((1,len(truth))),targets)[0] for s in scores[a]],axis=0)
            point_b=np.mean([specificity_draws(truth,s,np.ones((1,len(truth))),targets)[0] for s in scores[b]],axis=0)
            for j,target in enumerate(targets):
                low,high=np.quantile(specificity[a][:,j]-specificity[b][:,j],[.025,.975])
                specpairs.append({'fixed':a,'fine_tuning':b,'recall_target':target,'difference':point_a[j]-point_b[j],
                    'ci_low':low,'ci_high':high,'substitution_descriptive':substitution_status(low,high),
                    'oracle':'external recall-matched thresholds, not deployable'})
    for values in (differences,delong):
        adjusted=holm_adjust(np.asarray([r['p_value'] for r in values]))
        for row,p in zip(values,adjusted): row['p_holm']=float(p)
    table(rows,folder,'auc_specificity'); table(differences,folder,'paired_auc'); table(delong,folder,'delong_sensitivity')
    table(specpairs,folder,'paired_specificity'); table(partial,folder,'partial_auc'); table(curves,folder,'roc_coordinates')
    if not smoke:
        ensemble=pd.read_csv(ROOT/'results/external_test/manuscript_tables/ensemble_sensitivity.csv')
        confidence=pd.read_csv(ROOT/'results/external_test/manuscript_tables/cluster_bootstrap_wilson.csv')
        ensemble=ensemble[ensemble.panel.eq('principal')&ensemble.operating_point.eq('default')]
        confidence=confidence[confidence.panel.eq('principal')&confidence.operating_point.eq('default')&
            confidence.endpoint.eq('roc_auc')&confidence.analysis.eq('3-seed ensemble sensitivity')]
        table(ensemble[['model','roc_auc']].merge(confidence[['model','ci_low','ci_high']],on='model').assign(
            analysis='Frozen three-seed ensemble sensitivity; reused scene bootstrap'),folder,'ensemble_auc_sensitivity')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(12,5))
    for name,values in scores.items():
        grid=np.linspace(0,1,1001); tpr=np.mean([np.interp(grid,*roc(truth,s)[:2]) for s in values],axis=0)
        for axis in axes: axis.plot(grid,tpr,label=name)
    axes[0].set(xlabel='False positive rate (clear → obstructed)',ylabel='Obstructed recall',title=SCOPE)
    axes[1].set(xlabel='False positive rate',ylabel='Obstructed recall',ylim=(.9,1),title=f'{SCOPE}: high recall')
    axes[0].legend(fontsize=6); figure.tight_layout()
    folder.mkdir(parents=True,exist_ok=True); figure.savefig(folder/'roc.png',dpi=180); figure.savefig(folder/'roc.pdf'); plt.close(figure)


def dry_run(output):
    costs=pd.read_csv(ROOT/'results/fine_tuning_recovered/computational_costs.csv')
    ft=costs[costs.family.eq('fine_tuning')]
    seconds=float(ft.training_seconds.sum())
    estimates={'analysis_scope':SCOPE,'bootstrap_iterations':10000,'section_order':[0,4,3,2,1,5],
        'stage3_time_hours_range':[.35,2.5],'stage3_additional_disk_gib_range':[6.0,9.0],
        'basis':'40 cached-feature fixed refits plus post-test CPU statistics, image descriptors and protected-file hashing; no new deep forwards',
        'grouped_fine_tuning_optional':{'jobs':150,'time_hours_range':[.65*seconds/3600,1.8*seconds/3600],
        'additional_disk_gib_range':[45,100],'executed':False,'basis':'Measured original 150 FT training times; unchanged grouped outer-training sizes'},
        'principal_inference_jobs':0}
    write_json(output/'dry_run.json',estimates); print(json.dumps(estimates,indent=2),flush=True)
    return estimates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section',choices=['all','0','4','3','2','1','5','6'],default='all')
    parser.add_argument('--dry-run',action='store_true'); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); config=configuration()
    output=ROOT/('results/stage3_smoke' if args.smoke else config['output']); output.mkdir(parents=True,exist_ok=True)
    if args.dry_run: dry_run(output); return
    iterations=30 if args.smoke else config['bootstrap_iterations']
    sections=[0,4,3,2,1,5] if args.section=='all' else [int(args.section)]
    state_path=output/'progress.json'; state=json.loads(state_path.read_text()) if state_path.exists() else {'analysis_scope':SCOPE,'stages':{}}
    with threadpool_limits(config['cpu_threads']):
        for section in sections:
            print(f'Section {section} starting; {SCOPE}; smoke={args.smoke}',flush=True); started=perf_counter()
            if section==0:
                if not args.smoke:
                    from scripts.prepare_stage3 import prepare; prepare(output)
                    dry_run(output)
            elif section==4: native_sensitivity(output,iterations,args.smoke)
            elif section==3: grouped_cv(output,iterations,args.smoke)
            elif section==2: roc_analysis(output,iterations,args.smoke)
            elif section==1:
                from scripts.stage3_error_analysis import analyze; analyze(output,iterations,args.smoke)
            elif section==5:
                from scripts.analyze_blind_relabel import prepare_package; prepare_package(output,smoke=args.smoke)
            elif section==6:
                if args.smoke: raise ValueError('Final scientific verification cannot be a smoke')
                check_protected(output)
                from scripts.build_stage3_report import build; build()
            state['stages'][str(section)]={'completed':True,'smoke':args.smoke,'elapsed_seconds':perf_counter()-started}
            write_json(state_path,state); print(f'Section {section} completed in {perf_counter()-started:.1f}s',flush=True)
    if args.section=='all' and not args.smoke:
        check_protected(output)
        from scripts.build_stage3_report import build; build()


if __name__=='__main__': main()
