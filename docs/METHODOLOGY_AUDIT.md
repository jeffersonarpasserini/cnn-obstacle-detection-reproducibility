# Methodology audit

## Scope

This audit compares the legacy Python programs in `legacy/` with the methods and numerical claims in the associated IEEE Access manuscript.

## Findings

### Confirmed by code and archived results

- ImageNet weights were used with `include_top=False` and no global pooling.
- The final convolutional tensor was flattened to form each feature vector.
- The global seed was 1980.
- The experiment used 10 folds and one repeat.
- The published search contains 272 Approach A, 7,200 Approach B, 2,592 Approach C, and 2,592 Approach D configurations.
- The eight reported classifiers exclude the experimental PCC classifier.
- The selected linear SVM used `C=0.025`.
- The MLP used `random_state=1` and `max_iter=1000`.
- Logistic regression used `max_iter=500`.

### Corrections required for confirmatory evaluation

1. **PCA/UMAP leakage.** In the legacy PCA/UMAP programs, `fit_transform` is called on the complete feature matrix before the cross-validation loop. This allows the held-out fold to influence the learned representation.
2. **Fold reproducibility.** The legacy pipeline uses `RepeatedKFold` and unsorted filesystem input. The corrected runner uses sorted filenames and `StratifiedKFold(shuffle=True, random_state=1980)`.
3. **ROC metric.** The legacy `roc_auc_score` call receives hard predictions. In a binary task this is equivalent to balanced accuracy. The corrected runner stores both balanced accuracy and a conventional ROC-AUC computed from `decision_function` or `predict_proba` scores.
4. **Stochastic classifiers.** Random state was not passed explicitly to the tree, random forest, AdaBoost, PCA, or UMAP. The corrected runner passes the global seed wherever supported.
5. **Model selection.** Selecting winners from thousands of configurations on the same folds can lead to optimistic estimates. A nested or external confirmatory evaluation is recommended before the comparative equivalence claim is restored.

## Interpretation of archived results

Approach A uses complete pretrained feature vectors and is not affected by PCA/UMAP fitting leakage. Results for selected Approaches B, C, and D use PCA and must be rerun with the corrected pipeline. The archived results remain available for provenance and for reproducing the descriptive tables in the draft manuscript.

## Manuscript status

The manuscript has been updated to identify the affected results as exploratory, to describe the original fold generator and classifier parameters accurately, and to report the legacy hard-label `ROC` quantity as balanced accuracy. Numerical tables must be replaced after the corrected experiments are run.

