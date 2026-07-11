# Legacy experimental code

These files are preserved unchanged from the 2022 research repository so the provenance of the archived CSV and SPSS results remains traceable.

They are **not** the recommended entry point for new experiments. A retrospective audit found that:

- `RepeatedKFold` was used without class stratification;
- input filenames were obtained with unsorted `os.listdir()`;
- PCA and UMAP were fitted to the complete feature matrix before the folds were applied;
- Relief-F was fitted on the training portion of each fold;
- the column named `ROC` was computed from hard predicted labels and is therefore equivalent to balanced accuracy in this binary task, rather than a conventional score-based ROC-AUC;
- paths and experiment selections were edited directly in the scripts.

Use `src/run_corrected_experiments.py` for leakage-free validation. The legacy files are supplied only to document how the archived results were produced.

