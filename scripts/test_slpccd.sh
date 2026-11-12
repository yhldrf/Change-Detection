#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs cache
: "${DATA_ROOT:?Set DATA_ROOT to the dataset directory}"
PYTHON="${PYTHON:-python}"

export MPLCONFIGDIR=/tmp/matplotlib
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CHECKPOINT="${CHECKPOINT:-checkpoints/slpccd.pth}"
LOG_PATH="outputs/slpccd_4096_1vote.log"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing retained checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

"${PYTHON}" engine.py \
  --train_cache_path cache/slpccd_train.pt \
  --val_cache_path cache/slpccd_val.pt \
  --test_cache_path cache/slpccd_test.pt \
  --dataset slpccd \
  --test_model \
  --eval_split test \
  --slpccd_sample_mode street \
  --checkpoint "${CHECKPOINT}" \
  --log_path "${LOG_PATH}" \
  --data_root "${DATA_ROOT}" \
  --model_arch ptv3_xy_xyz_relation_attention \
  --batch_size 1 \
  --num_workers 4 \
  --num_points 4096 \
  --change_ratio 0.5 \
  --test_votes 1 \
  --seed 42 \
  --token_match_k 8 \
  --token_match_tau_feat 0.1 \
  --token_match_tau_geo 1.0 \
  --ptv3_order z,z-trans \
  --ptv3_patch_size 512 \
  --loss_type focal_ce \
  --lovasz_weight 1.0 \
  --binary_loss_weight 0.2 \
  --semantic_loss_weight 0.0 \
  --no-temporal_embedding \
  --temporal_aux_loss_weight 0.0 \
  --changed_conf_threshold 0.0 \
  --disable_swapped_supervision

echo "Retained SLPCCD evaluation saved to: ${LOG_PATH}"
