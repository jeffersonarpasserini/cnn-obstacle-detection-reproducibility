# Article completion workflow

## Recommended Linux execution on the RTX 4070 Ti

The complete follow-up can be started in the background with:

```bash
mkdir -p logs
nohup bash scripts/run_article_followup_linux.sh \
  > logs/article_followup_master.log 2>&1 &
echo $! > logs/article_followup.pid
```

The script verifies the dataset and GPU, runs the protocol tests, executes the
selected 10-fold run, resumes or executes LOOCV, and generates all follow-up
CSV files. It uses a batch size of 16 for GPU extraction, keeps the four needed
feature arrays available in RAM, and assigns eight BLAS threads to each
sequential PCA fit. Repeating the command safely resumes existing output.

Monitor progress in another terminal:

```bash
watch -n 30 python scripts/article_followup_status.py
```

For the detailed current group and ETA:

```bash
tail -f logs/article_selected_loocv.log
```

GPU utilization is expected mainly while missing CNN feature caches are being
created. During LOOCV, high CPU use with low GPU use is normal because PCA and
the classifiers run on the CPU.

This document defines the remaining computations required for the manuscript.
Run the commands from the repository root on the Linux workstation that holds
`cache/features/`.

## 1. Regenerate the main manuscript tables

```bash
python scripts/build_manuscript_tables.py \
  --results results/full_search \
  --config configs/full_search.json
```

Required outputs are written to `results/full_search/manuscript_tables/`:

- `approach_a_ranking.csv` through `approach_d_ranking.csv`;
- `approach_a_stage_one_winners.csv` through
  `approach_d_stage_one_winners.csv`;
- `selected_approaches.csv`;
- `selected_approaches_per_fold.csv`;
- `selected_approaches_statistics.json`;
- `selected_approaches_wilcoxon_holm.csv`;
- matching `*.tex` row fragments;
- `analysis_manifest.json` with source hashes and selection rules.

## 2. Generate all out-of-fold predictions for the four finalists

```bash
python src/run_experiments.py \
  --dataset via-dataset \
  --config configs/article_selected_models_10fold.json \
  --cache-dir cache/features \
  --output-dir results/article_selected_10fold \
  --max-loaded-extractors 2 \
  --prediction-mode all
```

This run uses the same stratified ten-fold protocol as the full search. Its
purpose is to persist one out-of-fold prediction for every image and finalist.

## 3. Run LOOCV for the same four fixed finalists

```bash
python src/run_experiments.py \
  --dataset via-dataset \
  --config configs/article_selected_models_loocv.json \
  --cache-dir cache/features \
  --output-dir results/article_selected_loocv \
  --max-loaded-extractors 2 \
  --prediction-mode all
```

The configuration choices must not be changed after inspecting LOOCV. This
keeps LOOCV as a stability analysis of the four preselected finalists.

## 4. Build LOOCV and error-analysis tables

```bash
python scripts/analyze_article_followup.py \
  --tenfold-results results/article_selected_10fold \
  --loocv-results results/article_selected_loocv \
  --output results/article_followup
```

The analyzer refuses incomplete runs and verifies the expected prediction
count. It produces:

- `validation_protocol_comparison.csv`;
- `validation_protocol_rows.tex`;
- `tenfold_error_predictions.csv` and `loocv_error_predictions.csv`;
- per-image error tables and error-frequency distributions for both protocols;
- `followup_manifest.json`.

## 5. Generate spatial decision-attribution maps

After both selected-model runs are complete, generate explanations for every
LOOCV image misclassified by all four finalists:

```bash
bash scripts/run_decision_attribution_linux.sh
```

The script reuses the four CNN feature caches, refits each finalist with the
explained image held out, and verifies three values before creating a map:

1. the regenerated class prediction equals the stored LOOCV prediction;
2. the regenerated continuous score equals the stored LOOCV score;
3. the score reconstructed from raw activations and back-projected linear
   coefficients equals the classifier score.

The process checkpoints every validated image/model pair under `records/`.
If interrupted, repeat the same command; completed records with matching input
signatures are reused.

Review `results/decision_attribution/attribution_validation.csv` and require
all absolute errors to remain within the implemented numerical tolerance.
Use `decision_attribution_representative.png` in the manuscript only after
visually checking the original images and overlays. Interpret the maps as
qualitative spatial decision evidence because localization annotations are not
available.

## 6. Optional focused ground-signage analysis

Create `data/ground_signage_manifest.csv` with one `filename` column and one
row per image included in the predefined subset. Then rerun:

```bash
python scripts/analyze_article_followup.py \
  --tenfold-results results/article_selected_10fold \
  --loocv-results results/article_selected_loocv \
  --output results/article_followup \
  --ground-signage-manifest data/ground_signage_manifest.csv
```

Do not reconstruct this subset from model errors. Its membership must be
defined from image content independently of predictions.

## 7. Integrity checks before using numbers in the article

```bash
python -m unittest discover -s tests -v
python scripts/verify_dataset.py
```

Confirm that both selected-run metadata files contain `"completed": true`,
that each run contains four experiments and 342 images, and that the follow-up
directory contains the generated CSV files. Article values should be copied
only from these generated tables.
