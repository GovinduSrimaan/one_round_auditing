#!/usr/bin/env bash
# dp_audit.sh — run the privacy audit across shadow models.
#
# SLURM header (comment out when running locally):
# #SBATCH --array=0-31
# #SBATCH --job-name=dp_audit
# #SBATCH --gpus=1
# #SBATCH --mem-per-cpu=2G
# #SBATCH --ntasks=1 --cpus-per-task=8
# #SBATCH --time=24:00:00
# #SBATCH --output=logs/%A_%a.out

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_DIR="${REPO_DIR}/experiments"
DATA_DIR="${REPO_DIR}/data"

SEED=2024
NUM_SHADOW=32

# Must match the values used during training
BATCH_SIZE="64"
NUM_EPOCHS="2"
NOISE_MULTIPLIER="0.4"

NUM_CANARIES=1000
NUM_POISON=0
POISON_TYPE="canary_duplicates_noisy"

CANARY_TYPE_ALL=("label_noise")
TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
CANARY_TYPE_IDX=$((TASK_ID / NUM_SHADOW))
SHADOW_MODEL_IDX=$((TASK_ID % NUM_SHADOW))
CANARY_TYPE="${CANARY_TYPE_ALL[$CANARY_TYPE_IDX]}"
EXPERIMENT="${CANARY_TYPE}_audit"

echo "Task ${TASK_ID}: experiment=${EXPERIMENT}, shadow_model=${SHADOW_MODEL_IDX}"

export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"
export DOWNLOAD_DATA=1

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
    audit

echo "Task finished"
