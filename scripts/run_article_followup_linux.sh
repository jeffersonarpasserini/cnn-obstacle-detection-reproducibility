#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

# The workstation used for the complete experiment has 16 logical CPU cores.
# Eight BLAS threads normally map well to its physical cores while avoiding
# nested oversubscription inside PCA/SVM. Override these values if needed.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export PYTHONUNBUFFERED=1

echo "Checking dataset..."
python scripts/verify_dataset.py

echo "Checking TensorFlow and GPU..."
python - <<'PY'
import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)
if not gpus:
    sys.exit("No TensorFlow GPU detected; fix the environment before starting.")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  echo "Running protocol tests..."
  python -m unittest discover -s tests -v
fi

echo "Running selected models with stratified 10-fold CV..."
python src/run_experiments.py \
  --dataset via-dataset \
  --config configs/article_selected_models_10fold.json \
  --cache-dir cache/features \
  --output-dir results/article_selected_10fold \
  --batch-size 16 \
  --max-loaded-extractors 4 \
  --prediction-mode all \
  2>&1 | tee -a logs/article_selected_10fold.log

python scripts/article_followup_status.py

echo "Running the same four fixed finalists with LOOCV..."
python src/run_experiments.py \
  --dataset via-dataset \
  --config configs/article_selected_models_loocv.json \
  --cache-dir cache/features \
  --output-dir results/article_selected_loocv \
  --batch-size 16 \
  --max-loaded-extractors 4 \
  --prediction-mode all \
  2>&1 | tee -a logs/article_selected_loocv.log

echo "Generating CSV and LaTeX tables..."
python scripts/analyze_article_followup.py \
  --tenfold-results results/article_selected_10fold \
  --loocv-results results/article_selected_loocv \
  --output results/article_followup \
  2>&1 | tee -a logs/article_followup_analysis.log

python scripts/article_followup_status.py
echo "Complete: results/article_followup/ contains the article tables."
