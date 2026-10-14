"""Image/whole-scene intervals and VIA-only operating thresholds."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from .artifact import prediction_metrics


def wilson(successes, count, confidence=0.95):
    if count == 0:
        return (float('nan'), float('nan'))
    z = norm.ppf(0.5 + confidence / 2)
    p = successes / count
    centre = (p + z*z/(2*count)) / (1+z*z/count)
    radius = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return (0.0 if successes == 0 else max(0.0, float(centre-radius)),
            1.0 if successes == count else min(1.0, float(centre+radius)))


def safety_threshold(truth, scores, target=0.95, *, source='via_oof'):
    """Scores increase towards clear. Ties predict obstructed; source gate is explicit."""
    if source not in ('via_inner_oof', 'via_oof'):
        raise ValueError('Safety thresholds may use only VIA out-of-fold observations')
    truth, scores = np.asarray(truth), np.asarray(scores, dtype=float)
    if len(truth) != len(scores) or not np.isfinite(scores).all() or set(truth) != {0, 1}:
        raise ValueError('Finite scores and both VIA classes are required')
    if not 0 < target <= 1:
        raise ValueError('Invalid obstructed recall target')
    candidates = np.r_[np.nextafter(scores.min(), -np.inf), np.unique(scores),
                       np.nextafter(scores.max(), np.inf)]
    best = None
    for threshold in candidates:
        clear = scores > threshold
        recall = float(np.mean(~clear[truth == 0]))
        specificity = float(np.mean(clear[truth == 1]))
        if recall >= target and (best is None or specificity > best['specificity']):
            best = {'threshold': float(threshold), 'obstructed_recall': recall,
                    'specificity': specificity, 'target': target, 'source': source,
                    'rule': 'clear iff score > threshold'}
    return best


def resample_indices(labels, rng, groups=None):
    labels = np.asarray(labels)
    if groups is None:
        return np.concatenate([rng.choice(np.flatnonzero(labels == label),
                                          np.sum(labels == label), replace=True) for label in (0, 1)])
    groups = np.asarray(groups)
    members = {group: np.flatnonzero(groups == group) for group in np.unique(groups)}
    # Mixed-class scenes form a third stratum; no scene is ever split by class.
    strata = {}
    for group, indices in members.items():
        strata.setdefault(tuple(np.unique(labels[indices])), []).append(group)
    draws = []
    for key in sorted(strata):
        options = strata[key]
        draws.extend(members[group] for group in rng.choice(options, len(options), replace=True))
    return np.concatenate(draws)


class BootstrapSampler:
    """Precompute scene membership; identical stratified draws without rebuilding the graph."""
    def __init__(self, labels, groups=None):
        labels = np.asarray(labels)
        self.count = len(labels)
        self.groups = groups
        if groups is None:
            self.strata = [np.flatnonzero(labels == label) for label in (0, 1)]
        else:
            unique, self.inverse = np.unique(np.asarray(groups), return_inverse=True)
            strata = {}
            for index in range(len(unique)):
                key = tuple(np.unique(labels[self.inverse == index]))
                strata.setdefault(key, []).append(index)
            self.strata = [np.asarray(strata[key]) for key in sorted(strata)]
            self.group_count = len(unique)

    def weights(self, rng):
        draws = np.concatenate([rng.choice(options, len(options), replace=True) for options in self.strata])
        if self.groups is None:
            return np.bincount(draws, minlength=self.count)
        return np.bincount(draws, minlength=self.group_count)[self.inverse]


def bootstrap_intervals(truth, predicted, scores=None, *, groups=None, iterations=10000, seed=1980, endpoints=None):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    scores = None if scores is None else np.asarray(scores)
    metrics = endpoints or (['accuracy', 'balanced_accuracy', 'mcc'] + (['roc_auc'] if scores is not None else []))
    rng = np.random.default_rng(seed)
    sampler = BootstrapSampler(truth, groups)
    samples = np.empty((iterations, len(metrics)))
    for i in range(iterations):
        value = weighted_metrics(truth, predicted, scores, sampler.weights(rng))
        samples[i] = [value[key] for key in metrics]
    return {key: [float(v) for v in np.nanquantile(samples[:, j], [.025, .975])]
            for j, key in enumerate(metrics)}


def paired_bootstrap(truth, first, second, *, groups=None, iterations=10000, seed=1980):
    """Second may be a seeds x images matrix: average individual seed metrics, not decisions."""
    truth, first, second = np.asarray(truth), np.asarray(first), np.asarray(second)
    if second.ndim == 1:
        second = second[None, :]
    keys = ['accuracy', 'obstructed_recall', 'obstructed_false_negative_rate']
    rng = np.random.default_rng(seed)
    sampler = BootstrapSampler(truth, groups)
    samples = []
    for _ in range(iterations):
        weights = sampler.weights(rng)
        a = weighted_metrics(truth, first, None, weights)
        b = weighted_metrics(truth, second, None, weights)
        samples.append([a[key] - b[key] for key in keys])
    point_a = prediction_metrics(truth, first)
    point_b = [prediction_metrics(truth, s) for s in second]
    return {key: {'difference': float(point_a[key]-np.mean([v[key] for v in point_b])),
                  'ci_low': float(np.quantile(np.asarray(samples)[:, j], .025)),
                  'ci_high': float(np.quantile(np.asarray(samples)[:, j], .975))}
            for j, key in enumerate(keys)}


def substitution_status(low, high, margin=.03, *, lower_is_better=False):
    if lower_is_better:
        low, high = -high, -low
    return 'supported' if low >= -margin else 'not_supported' if high < -margin else 'inconclusive'


def weighted_metrics(truth, predicted, scores, weights):
    """Bootstrap sufficient counts, with exact weighted AUC/ties and seed-metric means."""
    truth, weights = np.asarray(truth), np.asarray(weights, dtype=float)
    predicted = np.atleast_2d(predicted)
    tn = np.sum(weights*((truth == 0) & (predicted == 0)), axis=1)
    fp = np.sum(weights*((truth == 0) & (predicted == 1)), axis=1)
    fn = np.sum(weights*((truth == 1) & (predicted == 0)), axis=1)
    tp = np.sum(weights*((truth == 1) & (predicted == 1)), axis=1)
    denominator = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    values = {'accuracy': (tn+tp)/weights.sum(), 'balanced_accuracy': .5*(tn/(tn+fp)+tp/(tp+fn)),
              'mcc': np.divide(tp*tn-fp*fn, denominator, out=np.zeros_like(tp), where=denominator != 0),
              'obstructed_recall': tn/(tn+fp), 'obstructed_false_negative_rate': fp/(tn+fp)}
    if scores is not None:
        aucs = []
        for score in np.atleast_2d(scores):
            order = np.argsort(score, kind='stable')
            sorted_scores = np.asarray(score)[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0)+1]
            positive = np.add.reduceat(weights[order]*(truth[order] == 1), starts)
            negative = np.add.reduceat(weights[order]*(truth[order] == 0), starts)
            prior_negative = np.cumsum(negative)-negative
            total = positive.sum()*negative.sum()
            aucs.append(np.sum(positive*(prior_negative+.5*negative))/total if total else np.nan)
        values['roc_auc'] = np.asarray(aucs)
    return {key: float(np.mean(value)) for key, value in values.items()}
