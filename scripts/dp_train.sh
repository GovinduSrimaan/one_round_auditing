#!/usr/bin/env bash
# dp_train.sh — train shadow models for the privacy audit.
#
# SLURM header (comment out when running locally):
# #SBATCH --array=0-1            # one job per shadow model
# #SBATCH --job-name=dp_audit
# #SBATCH --gpus=1
# #SBATCH --mem-per-cpu=2G
# #SBATCH --ntasks=1 --cpus-per-task=8
# #SBATCH --time=24:00:00
# #SBATCH --output=logs/%A_%a.out

# ── User-configurable paths ──────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_DIR="${REPO_DIR}/experiments"
DATA_DIR="${REPO_DIR}/data"

# ── Hyperparameters ──────────────────────────────────────────────────────────
SEED=2024
NUM_SHADOW=2

AUGMULT_FACTOR="1"
LEARNING_RATE="4.0"
MAX_GRAD_NORM="1.0"

# ε ≈ 100  (fast / for testing)
BATCH_SIZE="64"
NUM_EPOCHS="2"
NOISE_MULTIPLIER="0.4"

# ε ≈ 7   (uncomment to use)
# BATCH_SIZE="4096"
# NUM_EPOCHS="220"
# NOISE_MULTIPLIER="3.0"

NUM_CANARIES=100
NUM_POISON=0
POISON_TYPE="canary_duplicates_noisy"

echo "BATCH_SIZE=${BATCH_SIZE}, NUM_EPOCHS=${NUM_EPOCHS}, NOISE_MULTIPLIER=${NOISE_MULTIPLIER}"

CANARY_TYPE_ALL=("label_noise")

# When running under SLURM, SLURM_ARRAY_TASK_ID is set automatically.
# When running locally, pass the shadow-model index via the first CLI arg or default to 0.
TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:-0}}"

CANARY_TYPE_IDX=$((TASK_ID / NUM_SHADOW))
SHADOW_MODEL_IDX=$((TASK_ID % NUM_SHADOW))
CANARY_TYPE="${CANARY_TYPE_ALL[$CANARY_TYPE_IDX]}"
EXPERIMENT="${CANARY_TYPE}_audit"

echo "Task ${TASK_ID}: experiment=${EXPERIMENT}, shadow_model=${SHADOW_MODEL_IDX}"

export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"
export DOWNLOAD_DATA=1   # let torchvision download CIFAR-10 if absent

python -u -m dp_audit \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --experiment "${EXPERIMENT}" \
    --data-dir "${DATA_DIR}" \
    --seed "${SEED}" \
    --num-shadow "${NUM_SHADOW}" \
    --num-canaries "${NUM_CANARIES}" \
    --canary-type "${CANARY_TYPE}" \
    --num-poison "${NUM_POISON}" \
    --poison-type "${POISON_TYPE}" \
    train \
    --shadow-model-idx "${SHADOW_MODEL_IDX}" \
    --augmult-factor "${AUGMULT_FACTOR}" \
    --learning-rate "${LEARNING_RATE}" \
    --max-grad-norm "${MAX_GRAD_NORM}" \
    --batch-size "${BATCH_SIZE}" \
    --noise-multiplier "${NOISE_MULTIPLIER}" \
    --num-epochs "${NUM_EPOCHS}"

echo "Task finished"
