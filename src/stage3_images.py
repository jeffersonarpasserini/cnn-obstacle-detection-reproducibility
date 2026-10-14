"""Image-only descriptors and author review sheets; no automatic human labels."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from .study_io import ROOT
from .stage3_io import SCOPE


def opencv():
    # Reuse the existing isolated image environment, never install into .venv.
    auxiliary=ROOT/'cache/scene-groups-venv/lib/python3.12/site-packages'
    if str(auxiliary) not in sys.path: sys.path.append(str(auxiliary))
    import cv2
    cv2.setNumThreads(2)
    return cv2


def attributes(path):
    cv2=opencv()
    with Image.open(path) as image:
        image=image.convert('RGB'); rgb=np.asarray(image); luminance=np.asarray(image.convert('YCbCr'))[:,:,0]
    saturation=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[:,:,1]
    edges=cv2.Canny(luminance,100,200,apertureSize=3,L2gradient=True)
    return {'filename':Path(path).name, 'luminance_mean':float(luminance.mean()),
            'luminance_p10':float(np.quantile(luminance,.1)), 'saturation_mean':float(saturation.mean()/255),
            'contrast_rms':float(luminance.std()), 'edge_density':float((edges>0).mean()),
            'sharpness_laplacian':float(cv2.Laplacian(luminance,cv2.CV_64F,ksize=1).var())}


def cosine_nearest(query, reference, *, exclude_self=False):
    def normalize(matrix):
        norm=np.sqrt(np.einsum('ij,ij->i',matrix,matrix,dtype=np.float64))
        if np.any(norm==0): raise ValueError('Zero feature norm')
        return matrix/norm[:,None]
    ref=normalize(reference).astype(np.float32); result=[]
    for start in range(0,len(query),32):
        q=normalize(query[start:start+32]).astype(np.float32)
        similarities=q@ref.T
        if exclude_self:
            similarities[np.arange(len(q)),np.arange(start,start+len(q))]=-np.inf
        result.extend(1-np.clip(similarities.max(axis=1),-1,1))
    return np.asarray(result)


def mosaic(rows,destination,*,titles=None,columns=4):
    rows=list(rows); count=len(rows); width,height=220,290
    canvas=Image.new('RGB',(columns*width,max(1,(count+columns-1)//columns)*height+34),'white')
    draw=ImageDraw.Draw(canvas); draw.text((8,8),SCOPE,fill='black')
    for index,row in enumerate(rows):
        path=row['path']; title=(titles[index] if titles is not None else row.get('identifier',''))
        with Image.open(path) as image:
            tile=ImageOps.contain(image.convert('RGB'),(width-10,height-48))
        left=(index%columns)*width; top=34+(index//columns)*height
        canvas.paste(tile,(left+(width-tile.width)//2,top))
        draw.text((left+5,top+height-44),str(title),fill='black')
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(destination)


def human_review_templates(output, clusters, errors):
    names=pd.DataFrame({'cluster':sorted(set(clusters)), 'name':'', 'notes':'', 'reviewer':'', 'date':''})
    annotations=pd.DataFrame(errors,columns=['filename','error_type'])
    for field in ('category','notes','reviewer','date'): annotations[field]=''
    # Never overwrite annotations once an author has started filling them.
    for filename,frame in [('cluster_names.csv',names),('error_annotations.csv',annotations)]:
        path=Path(output)/filename
        if path.exists():
            old=pd.read_csv(path,keep_default_na=False)
            if list(old.columns)!=list(frame.columns) or old.to_dict('records')!=frame.to_dict('records'):
                raise ValueError('Existing author annotation sheet differs; preserve it')
        else: frame.to_csv(path,index=False)
