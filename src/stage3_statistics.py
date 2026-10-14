"""Post-test estimators; preserve scene draws, ties and individual-seed means."""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import roc_curve
from .study_statistics import BootstrapSampler


def bootstrap_weights(truth, groups=None, iterations=10000, seed=1980):
    rng = np.random.default_rng(seed); sampler = BootstrapSampler(truth, groups)
    return np.asarray([sampler.weights(rng) for _ in range(iterations)], dtype=np.uint16)


def divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=np.asarray(b) != 0)


def auc_draws(truth, score, weights):
    order = np.argsort(score, kind='stable'); sorted_scores = np.asarray(score)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0)+1]
    out = []
    for offset in range(0, len(weights), 128):
        w = weights[offset:offset+128, order].astype(float)
        positive = np.add.reduceat(w*(np.asarray(truth)[order] == 1), starts, axis=1)
        negative = np.add.reduceat(w*(np.asarray(truth)[order] == 0), starts, axis=1)
        numerator = np.sum(positive*(np.cumsum(negative, axis=1)-.5*negative), axis=1)
        out.extend(divide(numerator, positive.sum(axis=1)*negative.sum(axis=1)))
    return np.asarray(out)


def metric_draws(truth, predicted, weights, scores=None):
    truth = np.asarray(truth); predicted = np.atleast_2d(predicted)
    arrays = []
    for d in predicted:
        tn = weights@((truth == 0)&(d == 0)).astype(float)
        fp = weights@((truth == 0)&(d == 1)).astype(float)
        fn = weights@((truth == 1)&(d == 0)).astype(float)
        tp = weights@((truth == 1)&(d == 1)).astype(float)
        recall0, recall1 = divide(tn, tn+fp), divide(tp, tp+fn)
        arrays.append({'accuracy': divide(tn+tp, weights.sum(axis=1)),
            'balanced_accuracy': .5*(recall0+recall1),
            'mcc': divide(tp*tn-fp*fn, np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),
            'obstructed_recall': recall0, 'obstructed_false_negative_rate': 1-recall0,
            'clear_recall': recall1, 'recall_nonclear': recall0, 'recall_clear': recall1,
            'precision_nonclear': divide(tn, tn+fn), 'precision_clear': divide(tp, tp+fp),
            'f1_nonclear': divide(2*tn, 2*tn+fp+fn), 'f1_clear': divide(2*tp, 2*tp+fp+fn),
            'tn_obstructed': tn, 'fp_obstructed_as_clear': fp, 'fn_clear_as_obstructed': fn, 'tp_clear': tp})
        a=arrays[-1]
        a.update(f1=a['f1_clear'],f1_obstructed=a['f1_nonclear'],
            macro_f1=.5*(a['f1_clear']+a['f1_nonclear']),
            obstructed_precision=a['precision_nonclear'],clear_precision=a['precision_clear'])
    result = {k: np.mean([a[k] for a in arrays], axis=0) for k in arrays[0]}
    if scores is not None:
        result['roc_auc'] = np.mean([auc_draws(truth, s, weights) for s in np.atleast_2d(scores)], axis=0)
    return result


def intervals(truth, decisions, weights, scores=None):
    draws = metric_draws(truth, decisions, weights, scores)
    points = metric_draws(truth, decisions, np.ones((1,len(truth))), scores)
    truth=np.asarray(truth)
    support0=weights@(truth==0).astype(float); support1=weights@(truth==1).astype(float)
    result={}
    for key,value in draws.items():
        valid=(support0+support1)>0; point_valid=len(truth)>0
        if key in ('obstructed_recall','obstructed_false_negative_rate','recall_nonclear'):
            valid &= support0>0; point_valid &= bool(np.any(truth==0))
        elif key in ('clear_recall','recall_clear'):
            valid &= support1>0; point_valid &= bool(np.any(truth==1))
        elif key in ('balanced_accuracy','roc_auc'):
            valid &= (support0>0)&(support1>0)
            point_valid &= bool(np.any(truth==0) and np.any(truth==1))
        usable=value[valid & np.isfinite(value)]
        low,high=np.quantile(usable,[.025,.975]) if len(usable) else (np.nan,np.nan)
        result[key]={'estimate':float(points[key][0]) if point_valid else np.nan,
            'ci_low':float(low),'ci_high':float(high),'bootstrap_valid':len(usable),
            'bootstrap_total':len(weights)}
    return result


def oracle_threshold(truth, scores):
    """Maximise external accuracy; descriptive oracle, never a deployment threshold."""
    truth, scores = np.asarray(truth), np.asarray(scores)
    order = np.argsort(scores, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0)+1,len(scores)]
    starts = np.r_[0,ends[:-1]]
    prefix0 = np.r_[0,np.cumsum(truth[order] == 0)]
    prefix1 = np.r_[0,np.cumsum(truth[order] == 1)]
    positions = np.r_[starts,len(scores)]
    accuracy = (prefix0[positions]+prefix1[-1]-prefix1[positions])/len(scores)
    index = np.argmax(accuracy)
    threshold = scores[order][positions[index]] if positions[index]<len(scores) else np.nextafter(scores.max(),np.inf)
    return float(threshold), float(accuracy[index])


def roc(truth, score):
    # Rank using -score to avoid losing tiny clear probabilities through 1-score
    # cancellation. It has exactly the mathematical order of 1-score_clear.
    fpr,tpr,cut = roc_curve(1-np.asarray(truth),-np.asarray(score),drop_intermediate=False)
    return fpr,tpr,cut


def specificity_draws(truth, score, weights, targets=(.9,.95,.97)):
    order=np.argsort(score,kind='stable'); s=np.asarray(score)[order]; y=np.asarray(truth)[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
    output=[]
    for offset in range(0,len(weights),128):
        w=weights[offset:offset+128,order].astype(float)
        obstructed=np.add.reduceat(w*(y==0),starts,axis=1)
        clear=np.add.reduceat(w*(y==1),starts,axis=1)
        recall=divide(np.cumsum(obstructed,axis=1),obstructed.sum(axis=1)[:,None])
        false_alarm=divide(np.cumsum(clear,axis=1),clear.sum(axis=1)[:,None])
        output.extend(np.stack([1-false_alarm[np.arange(len(w)),np.argmax(recall>=target,axis=1)] for target in targets],axis=1))
    return np.asarray(output)


def partial_auc(truth, score, lower=.9):
    fpr,tpr,_=roc(truth,score); specificity=1-fpr
    keep=(tpr>=lower)&(tpr<=1)
    i=np.searchsorted(tpr,lower,side='left')
    boundary=float(specificity[i]) if tpr[i]==lower else float(np.interp(lower,tpr[i-1:i+1],specificity[i-1:i+1]))
    x=np.r_[lower,tpr[keep]]; y=np.r_[boundary,specificity[keep]]
    area=float(np.trapezoid(y,x)); minimum=(1-lower)**2/2; maximum=1-lower
    # Below-diagonal correction is explicitly retained, rather than dropping
    # poor bootstrap draws; corresponds to allow.invalid.partial.auc.correct.
    return .5*(1+(area-minimum)/(maximum-minimum))


def partial_auc_draws(truth,score,weights,lower=.9):
    order=np.argsort(score,kind='stable'); s=np.asarray(score)[order]; y=np.asarray(truth)[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]; output=[]
    for offset in range(0,len(weights),128):
        w=weights[offset:offset+128,order].astype(float)
        obstructed=np.add.reduceat(w*(y==0),starts,axis=1); clear=np.add.reduceat(w*(y==1),starts,axis=1)
        x=np.column_stack([np.zeros(len(w)),divide(np.cumsum(obstructed,axis=1),obstructed.sum(axis=1)[:,None])])
        z=np.column_stack([np.ones(len(w)),1-divide(np.cumsum(clear,axis=1),clear.sum(axis=1)[:,None])])
        width=x[:,1:]-x[:,:-1]; left=np.maximum(x[:,:-1],lower)
        selected=np.maximum(x[:,1:]-left,0)
        ordinate=z[:,:-1]+divide(left-x[:,:-1],width)*(z[:,1:]-z[:,:-1])
        area=np.sum(.5*(ordinate+z[:,1:])*selected,axis=1)
        output.extend(.5*(1+(area-(1-lower)**2/2)/((1-lower)-(1-lower)**2/2)))
    return np.asarray(output)


def oracle_metric_draws(truth,score,weights):
    order=np.argsort(score,kind='stable'); s=np.asarray(score)[order]; y=np.asarray(truth)[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]; collected=[]
    for offset in range(0,len(weights),128):
        w=weights[offset:offset+128,order].astype(float)
        obs=np.add.reduceat(w*(y==0),starts,axis=1); clear=np.add.reduceat(w*(y==1),starts,axis=1)
        p0=np.column_stack([np.zeros(len(w)),np.cumsum(obs,axis=1)])
        p1=np.column_stack([np.zeros(len(w)),np.cumsum(clear,axis=1)])
        choice=np.argmax(p0+p1[:,-1,None]-p1,axis=1); indices=np.arange(len(w))
        tn=p0[indices,choice]; fn=p1[indices,choice]; fp=p0[:,-1]-tn; tp=p1[:,-1]-fn
        r0=divide(tn,tn+fp); r1=divide(tp,tp+fn)
        collected.append({'accuracy':divide(tn+tp,w.sum(axis=1)), 'balanced_accuracy':.5*(r0+r1),
            'mcc':divide(tp*tn-fp*fn,np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),
            'obstructed_recall':r0,'obstructed_false_negative_rate':1-r0,'clear_recall':r1,
            'recall_nonclear':r0,'recall_clear':r1,'precision_nonclear':divide(tn,tn+fn),
            'precision_clear':divide(tp,tp+fp),'f1_nonclear':divide(2*tn,2*tn+fp+fn),
            'f1_clear':divide(2*tp,2*tp+fp+fn),'tn_obstructed':tn,'fp_obstructed_as_clear':fp,
            'fn_clear_as_obstructed':fn,'tp_clear':tp})
        a=collected[-1]
        a.update(f1=a['f1_clear'],f1_obstructed=a['f1_nonclear'],
            macro_f1=.5*(a['f1_clear']+a['f1_nonclear']),
            obstructed_precision=a['precision_nonclear'],clear_precision=a['precision_clear'])
    return {key:np.concatenate([v[key] for v in collected]) for key in collected[0]}


def delong_contrast(truth,scores,contrast):
    """Multivariate structural components, including correlated FT seed means."""
    truth=np.asarray(truth); scores=np.atleast_2d(scores)
    positive=scores[:,truth==1]; negative=scores[:,truth==0]
    components=[]
    for p,n in zip(positive,negative):
        comparisons=(p[:,None]>n[None,:]).astype(float)+.5*(p[:,None]==n[None,:])
        components.append((comparisons.mean(axis=1),comparisons.mean(axis=0)))
    vp=np.asarray([a for a,b in components]); vn=np.asarray([b for a,b in components])
    covariance=np.atleast_2d(np.cov(vp,ddof=1))/positive.shape[1]+np.atleast_2d(np.cov(vn,ddof=1))/negative.shape[1]
    contrast=np.asarray(contrast); difference=float(contrast@vp.mean(axis=1)); variance=float(contrast@covariance@contrast)
    se=np.sqrt(max(variance,0))
    if se<1e-12:
        if abs(difference)<1e-12: difference=0.
        p=float(difference==0)
    else: p=float(2*norm.sf(abs(difference)/se))
    return {'difference':difference,'standard_error':float(se),'ci_low':difference-1.95996398454*se,
            'ci_high':difference+1.95996398454*se,'p_value':p}


def cluster_logistic(attributes,correct,groups):
    """Unpenalized descriptive likelihood; cluster sandwich, no causal claim."""
    x=np.asarray(attributes,dtype=float); y=np.asarray(correct,dtype=float)
    x=np.column_stack([np.ones(len(x)),x])
    def objective(beta):
        eta=x@beta
        return np.sum(np.logaddexp(0,eta)-y*eta),x.T@(expit(eta)-y)
    fit=minimize(objective,np.zeros(x.shape[1]),jac=True,method='BFGS',options={'gtol':1e-6,'maxiter':1000})
    probability=expit(x@fit.x); hessian=x.T@((probability*(1-probability))[:,None]*x)
    bread=np.linalg.pinv(hessian)
    unique,inverse=np.unique(groups,return_inverse=True); score=np.zeros((len(unique),x.shape[1]))
    np.add.at(score,inverse,x*(y-probability)[:,None])
    correction=len(unique)/(len(unique)-1)*(len(x)-1)/(len(x)-x.shape[1])
    covariance=bread@(score.T@score)@bread*correction; se=np.sqrt(np.maximum(np.diag(covariance),0))
    return fit.x,se,{'converged':bool(fit.success or np.linalg.norm(fit.jac,np.inf)<1e-4),
        'max_abs_gradient':float(np.linalg.norm(fit.jac,np.inf)), 'design_rank':int(np.linalg.matrix_rank(x)),
        'parameters':x.shape[1], 'clusters':len(unique), 'observations':len(x),
        'separation_warning':bool(np.max(np.abs(fit.x))>30), 'status':str(fit.message)}
