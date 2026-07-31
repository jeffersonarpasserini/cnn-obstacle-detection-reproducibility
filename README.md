# CNN Feature Extractors for Obstacle Detection

Reproducibility repository for the manuscript:

> **Cross-Analysis of CNN Architectures as Feature Extractors for Obstacle Detection to Aid the Visually Impaired**  
> Jefferson Antonio Ribeiro Passerini and Fabricio Aparecido Breve  
> São Paulo State University (UNESP), Rio Claro, Brazil

## Scope

This repository contains the dataset manifest, experiment definitions, Python
programs, per-fold results, statistical analyses, and manuscript tables for a
comparison of ImageNet-pretrained CNNs used as fixed feature extractors. The
task classifies 342 path images as clear or obstructed without fine-tuning the
CNN weights.

The full experiment evaluates 12,656 configurations organized into four
approaches:

| Approach | Feature construction |
|---|---|
| A | Complete feature vector from one CNN or a concatenated CNN pair |
| B | Dimensionality reduction or feature selection applied to one CNN |
| C | Concatenate a CNN pair, then reduce/select the joint vector |
| D | Reduce/select each CNN independently, then concatenate the outputs |

PCA, UMAP, Relief-F, scaling, and classifier fitting are performed using only
the training partition of each fold. The test partition is used only for
evaluation. CNN feature extraction is fixed and therefore does not fit the
study images.

## Repository layout

```text
.
├── configs/
│   ├── full_search.json
│   ├── article_selected_models_10fold.json
│   └── article_selected_models_loocv.json
├── data/
│   ├── README.md
│   └── dataset_manifest.csv
├── docs/
│   ├── ARTICLE_COMPLETION.md
│   └── PROVENANCE.md
├── results/
│   └── full_search/
│       ├── per_fold_metrics.csv
│       ├── summary_metrics.csv
│       ├── fold_assignments.csv
│       └── manuscript_tables/
├── scripts/
│   ├── article_followup_status.py
│   ├── analyze_article_followup.py
│   ├── analyze_results.py
│   ├── build_manuscript_tables.py
│   ├── generate_full_search_config.py
│   ├── run_article_followup_linux.sh
│   └── verify_dataset.py
├── src/
│   ├── artifact.py
│   └── run_experiments.py
├── tests/
└── via-dataset/
```

## Completed result set

The canonical run is `results/full_search/`. It contains:

- 342 images: 175 clear and 167 obstructed;
- 12,656 classifier configurations;
- 1,582 preprocessing groups;
- 10 stratified folds with seed 1980;
- 126,560 unique experiment/fold records;
- a completed run manifest and the exact fold assignment for every image.

`results/full_search/manuscript_tables/` contains the CSV tables generated
from this run. These files are intended to be the numerical source for the
article; do not manually copy values from console output.

The hierarchical selection procedure reproduces the structure described in
the manuscript: all Approach A configurations are ranked together; Approach B
first selects one configuration per CNN; Approaches C and D first select one
configuration per CNN pair. Finalists are ranked by mean within-fold Friedman
rank of accuracy. Ties are resolved by median accuracy, pooled out-of-fold
accuracy, pooled obstructed-path recall, and experiment identifier.

The selected configurations are:

| Approach | Configuration | Median fold accuracy | Pooled obstructed recall |
|---|---|---:|---:|
| A | ResNet101V2 + logistic regression | 0.9412 | 0.9162 |
| B | EfficientNetB3 + PCA(300) + linear SVM | 0.9412 | 0.9281 |
| C | MobileNet + ResNet50 + PCA(300) + linear SVM | 0.9412 | 0.9042 |
| D | MobileNet + ResNet50 + PCA(300 per CNN) + linear SVM | 0.9412 | 0.9042 |

Because configuration selection and comparison use the same ten folds, the
finalist-level inferential tests are reported as exploratory. LOOCV on the same
dataset measures resampling stability and is not independent external
validation.

## Environment

For inspecting and regenerating result tables:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-results.txt
```

For CNN feature extraction and complete experiments on Linux:

```bash
conda env create -f environment.yml
conda activate ieee-access-cnn-obstacle
```

Alternatively:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-experiments.txt
```

The maintained environment uses Python 3.12 and TensorFlow 2.21.0. On Linux,
the requirements include TensorFlow's NVIDIA CUDA extra. Confirm GPU detection
before extracting features:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Dataset

The expected layout is:

```text
via-dataset/clear.000.jpg
via-dataset/nonclear.000.jpg
...
```

Verify the images against the committed manifest:

```bash
python scripts/verify_dataset.py
```

The dataset source is <https://github.com/fbreve/via-dataset>. Dataset use and
citation are governed by that repository.

## Regenerate manuscript tables

```bash
python scripts/build_manuscript_tables.py \
  --results results/full_search \
  --config configs/full_search.json
```

The command writes one ranking CSV and one LaTeX-row file for each approach,
the selected finalist table, per-fold finalist values, Friedman statistics,
pairwise Wilcoxon tests with Holm adjustment, and an analysis manifest with
source hashes.

For a general descriptive summary of the complete search:

```bash
python scripts/analyze_results.py \
  --results results/full_search \
  --top 20
```

## Run the full search

```bash
python src/run_experiments.py \
  --dataset via-dataset \
  --config configs/full_search.json \
  --cache-dir cache/features \
  --output-dir results/full_search \
  --max-loaded-extractors 2 \
  --relieff-n-jobs 2 \
  --prediction-mode errors
```

The runner checkpoints each preprocessing group. Repeating the same command
continues an interrupted run and skips complete groups. Feature caches are
bound to the ordered dataset fingerprint. Relief-F rankings are computed once
per training fold at the largest requested cutoff and reused for every nested
feature count.

Key outputs:

| File | Contents |
|---|---|
| `per_fold_metrics.csv` | Accuracy, class metrics, F1, balanced accuracy, MCC, ROC-AUC, dimensions, and timings |
| `summary_metrics.csv` | Q1, median, and Q3 for every configuration |
| `fold_assignments.csv` | Exact held-out fold for each image |
| `dataset_index.csv` | Sorted filenames, labels, and dataset identity |
| `feature_dimensions.csv` | Feature-vector dimensions and memory sizes |
| `run_metadata.json` | Completion state, command, hashes, hardware, platform, and package versions |
| `relieff_rankings/` | Fold-specific reusable Relief-F rankings |

Labels are explicit in every output: `nonclear`/obstructed is class 0 and
`clear` is class 1.

## Selected-model 10-fold and LOOCV runs

The follow-up commands and expected output files are documented in
[`docs/ARTICLE_COMPLETION.md`](docs/ARTICLE_COMPLETION.md). Both selected runs
must use `--prediction-mode all` so the repository can generate per-image error
tables and the article's focused subset analysis.

On the validated Linux workstation, start the complete resumable workflow with:

```bash
mkdir -p logs
nohup bash scripts/run_article_followup_linux.sh \
  > logs/article_followup_master.log 2>&1 &
echo $! > logs/article_followup.pid
```

Monitor it with `watch -n 30 python scripts/article_followup_status.py`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover deterministic folds, train-only transformation fitting,
prediction output integrity, Relief-F cache reuse, ranking ties, and multiple
comparison adjustment.

## Publication and citation

Before article submission:

1. run the selected 10-fold and LOOCV commands;
2. generate `results/article_followup/*.csv`;
3. update the corresponding article tables only from those CSV files;
4. create a tagged GitHub release;
5. archive that release in Zenodo and insert its DOI in `CITATION.cff` and the article.

Software is released under the MIT License. See `CITATION.cff` for citation
metadata.
