#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs cache
: "${DATA_ROOT:?Set DATA_ROOT to the dataset directory}"
PYTHON="${PYTHON:-python}"

export MPLCONFIGDIR=/tmp/matplotlib
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

"${PYTHON}" engine.py \
  --dataset slpccd \
  --data_root "${DATA_ROOT}" \
  --train_cache_path cache/slpccd_train.pt \
  --val_cache_path cache/slpccd_val.pt \
  --test_cache_path cache/slpccd_test.pt \
  --model_arch ptv3_xy_xyz_relation_attention \
  --epochs 100 \
  --batch_size 1 \
  --num_points 4096 \
  --change_ratio 0.5 \
  --token_match_k 8 \
  --ptv3_order z,z-trans \
  --ptv3_patch_size 512 \
  --loss_type focal_ce \
  --lovasz_weight 1.0 \
  --binary_loss_weight 0.2 \
  --semantic_loss_weight 0 \
  --relation_loss_weight 0 \
  --no-temporal_embedding \
  --disable_swapped_supervision \
  --temporal_aux_loss_weight 0 \
  --changed_direction_loss_weight 0 \
  --existence_loss_weight 0 \
  --checkpoint slpccd.pth
