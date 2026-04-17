#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./utils/run_oft_finetune_knn.sh <dataset_name> <lora_tag> [gpu_id]
# Example:
#   ./utils/run_oft_finetune_knn.sh pick_up_red_cube_200 lora-r32 9
#
# Behavior:
# - Saves policies under: /bluesclues-data/home/pingpong-nima/raj/eccv_26/0_policies
# - Creates run folder: knn_{dataset}_{lora}_{DD_MM_YY}
# - Writes terminal output to: stdout.log inside run folder
# - Ensures all checkpoint folders (--2000_chkpt, --4000_chkpt, ...) are moved inside run folder

DATASET_NAME="${1:?dataset_name required}"
LORA_TAG="${2:?lora_tag required}"
GPU_ID="${3:-9}"

DATE="$(date +%d_%m_%y)"
TASK="${DATASET_NAME}"
RUN_ID="knn_${DATASET_NAME}_${LORA_TAG}_${DATE}"

ROOT="/bluesclues-data/home/pingpong-nima/raj/eccv_26"
OPENVLA_OFT_DIR="${ROOT}/openvla-oft"
DATA_ROOT_DIR="${ROOT}/tensorflow_datasets"
POLICY_ROOT="${ROOT}/0_policies"
POLICY_DIR="${POLICY_ROOT}/${RUN_ID}"

mkdir -p "${POLICY_DIR}"

# Safety: make TFDS resolution explicit for any downstream loader behavior.
export TFDS_DATA_DIR="${DATA_ROOT_DIR}"

cd "${OPENVLA_OFT_DIR}"

echo "Starting KNN OFT fine-tune run: ${RUN_ID}"
echo "Using GPU: ${GPU_ID}"
echo "Policy output folder: ${POLICY_DIR}"
echo "Log file: ${POLICY_DIR}/stdout.log"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python vla-scripts/finetune_knn.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${POLICY_ROOT}" \
  --run_id_override "${RUN_ID}" \
  --use_l1_regression True \
  --use_diffusion False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 4 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --lr_warmup_steps 500 \
  --max_steps 8000 \
  --save_freq 2000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --knn True \
  --number_heads 20 \
  --knn_neighbors 30 \
  --knn_k_grid "10,20,30,40" \
  --knn_max_samples 200 \
  --knn_token_mode state \
  --knn_weighted_neighbors True \
  --knn_temp_excl_w 30 \
  --knn_include_mlp True \
  --wandb_entity "raj-sarav-uc-berkeley-electrical-engineering-computer-sc" \
  --wandb_project "${TASK}-openvla-oft-finetune-knn" \
  2>&1 | tee "${POLICY_DIR}/stdout.log"

# Move checkpoint folders generated as:
#   ${POLICY_ROOT}/${RUN_ID}--{step}_chkpt
# into:
#   ${POLICY_DIR}/
shopt -s nullglob
for chkpt_dir in "${POLICY_ROOT}/${RUN_ID}"--*_chkpt; do
  mv "${chkpt_dir}" "${POLICY_DIR}/"
done
shopt -u nullglob

echo "Run finished."
echo "Verifying artifacts..."
test -f "${POLICY_DIR}/stdout.log"
ls -d "${POLICY_DIR}"/*_chkpt
echo "All artifacts are in: ${POLICY_DIR}"

