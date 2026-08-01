#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONHASHSEED="${PYTHONHASHSEED:-1980}"
export TF_DETERMINISTIC_OPS="${TF_DETERMINISTIC_OPS:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "Checking dataset..."
"$PYTHON_BIN" scripts/verify_dataset.py --dataset via-dataset

echo "Checking TensorFlow GPU visibility..."
"$PYTHON_BIN" -c 'import tensorflow as tf; print("TensorFlow:", tf.__version__); print("GPUs:", tf.config.list_physical_devices("GPU"))'

echo "Generating exact decision-contribution maps..."
"$PYTHON_BIN" scripts/build_decision_attribution.py \
  --dataset via-dataset \
  --config configs/article_selected_models_loocv.json \
  --predictions results/article_selected_loocv \
  --cache-dir cache/features \
  --output results/decision_attribution \
  --batch-size "$BATCH_SIZE"

echo "Done. Inspect results/decision_attribution/decision_attribution_representative.png"
