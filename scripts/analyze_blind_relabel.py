#!/usr/bin/env python3
"""Prepare a blind package; analyse only supplied or explicitly synthetic replies."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.stage3_io import SCOPE,table
from src.stage3_statistics import bootstrap_weights
from src.study_statistics import wilson
from src.study_io import sha256,write_json


def prepare_package(output,smoke=False):
    sample=pd.read_csv(ROOT/'external-data/label_audit_sample.csv')
    if sample.filename.duplicated().any() or len(sample)!=100: raise ValueError('Blind sample differs from the 100 approved audit images')
    sample=sample.iloc[np.random.default_rng(1980).permutation(len(sample))].reset_index(drop=True)
    package=(Path(output)/'blind_package' if smoke else ROOT/'external-data/blind_relabel')
    images=package/'images'; images.mkdir(parents=True,exist_ok=True)
    groups=pd.read_csv(ROOT/'external-data/external_groups.csv').set_index('filename').group_id
    key=[]
    for index,row in enumerate(sample.itertuples(),1):
        item=f'item_{index:03d}.jpg'; source=ROOT/'external-data/via-dataset-extended'/row.filename
        if sha256(source)!=row.sha256: raise ValueError('Audit source image changed')
        target=images/item
        if target.exists() and sha256(target)!=row.sha256: raise ValueError('Existing blinded image changed')
        if not target.exists(): shutil.copyfile(source,target)
        key.append({'item':item,'filename':row.filename,'original_label':int(row.original_label),
                    'group_id':groups[row.filename],'sha256':row.sha256})
    pd.DataFrame(key).to_csv(package/'key.csv',index=False)
    instructions=('# exploratory/post hoc — rotulagem independente às cegas\n\n'
        'Classifique as 100 imagens sem consultar os rótulos anteriores, resultados de modelos ou key.csv. '
        'Use o critério do VIA: nonclear = há obstáculo no caminho à frente; clear = caminho livre. '
        'Avalie o caminho à frente.\n\n'
        'Preencha responses_template.csv: item (nome recebido), label (clear ou nonclear), '
        'confidence(1-3) (1=baixa, 2=moderada, 3=alta), notes (observação opcional). '
        'Mantenha os 100 itens e devolva o CSV preenchido. Não renomeie as imagens.\n')
    (package/'instructions.md').write_text(instructions)
    template=pd.DataFrame({'item':[row['item'] for row in key],'label':'','confidence(1-3)':'','notes':''})
    template_path=package/'responses_template.csv'
    if not template_path.exists(): template.to_csv(template_path,index=False)
    elif pd.read_csv(template_path,keep_default_na=False).to_dict('records')!=template.to_dict('records'):
        raise ValueError('Preserve an already filled response template')
    destination=package/'blind_relabel_package.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(images.glob('*.jpg'))+[package/'instructions.md',template_path]:
            name=str(path.relative_to(package)); info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED; archive.writestr(info,path.read_bytes())
    with zipfile.ZipFile(destination) as archive:
        included=archive.namelist()
        if 'key.csv' in included or len(included)!=102 or any('clear.' in n or 'nonclear.' in n for n in included):
            raise ValueError('Blind package contains identifying labels/key or has wrong coverage')
    table([{'images':100,'included_files':102,'key_in_archive':False,'responses_analysed':False,
            'zip':str(destination.relative_to(ROOT)),'zip_sha256':sha256(destination),
            'key_kept_outside_package':str((package/'key.csv').relative_to(ROOT))}],Path(output)/'blind_relabel','package_manifest')
    write_json(Path(output)/'blind_relabel/status.json',{'analysis_scope':SCOPE,'prepared':True,
        'responses_received':False,'analysis_executed':False,'zip':str(destination.relative_to(ROOT))})


def agreement(truth,labels,weights):
    truth=np.asarray(truth); labels=np.asarray(labels); w=np.asarray(weights,dtype=float)
    total=w.sum(axis=1); observed=(w@(truth==labels).astype(float))/total
    first=(w@(truth==1).astype(float))/total; second=(w@(labels==1).astype(float))/total
    expected=first*second+(1-first)*(1-second)
    kappa=np.divide(observed-expected,1-expected,out=np.full_like(observed,np.nan),where=expected!=1)
    return observed,kappa


def analyze(responses,key,output,iterations=10000,synthetic=False):
    responses=pd.read_csv(responses,keep_default_na=False); key=pd.read_csv(key)
    required={'item','label','confidence(1-3)','notes'}
    if set(responses.columns)!=required or responses.item.duplicated().any() or set(responses.item)!=set(key.item):
        raise ValueError('Replies must cover each blinded item exactly once')
    if not set(responses.label)<= {'clear','nonclear'}: raise ValueError('Every label must be clear or nonclear')
    confidence=pd.to_numeric(responses['confidence(1-3)'],errors='raise')
    if not set(confidence)<= {1,2,3}: raise ValueError('Confidence must be 1, 2 or 3')
    aligned=responses.set_index('item').loc[key.item]
    labels=aligned.label.map({'clear':1,'nonclear':0}).to_numpy(); truth=key.original_label.to_numpy()
    w=bootstrap_weights(truth,key.group_id.to_numpy() if 'group_id' in key else None,iterations)
    draws,kappa=agreement(truth,labels,w); point,estimate=agreement(truth,labels,np.ones((1,len(truth))))
    low,high=wilson(int(np.sum(truth==labels)),len(truth))
    table([{'agreement':float(point[0]),'agreement_wilson_low':low,'agreement_wilson_high':high,
        'agreement_cluster_ci_low':float(np.quantile(draws,.025)),'agreement_cluster_ci_high':float(np.quantile(draws,.975)),
        'cohen_kappa':float(estimate[0]),'kappa_ci_low':float(np.nanquantile(kappa,.025)),
        'kappa_ci_high':float(np.nanquantile(kappa,.975)),'bootstrap_iterations':iterations,'synthetic':synthetic}],output,'agreement')
    table(pd.crosstab(pd.Series(truth,name='original_label'),pd.Series(labels,name='blind_label')).reset_index(),output,'confusion_matrix')
    mismatch=key[truth!=labels].copy(); mismatch['blind_label']=labels[truth!=labels]
    table(mismatch,output,'discordances')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses',type=Path,required=True); parser.add_argument('--key',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('results/stage3/blind_relabel'))
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.dry_run: print(json.dumps({'analysis_scope':SCOPE,'images':100,'bootstrap_iterations':10000,'fit_jobs':0,'responses_opened':False}))
    else: analyze(args.responses,args.key,args.output,iterations=30 if args.smoke else 10000,synthetic=args.smoke)
