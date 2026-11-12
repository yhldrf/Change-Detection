#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs cache
: "${DATA_ROOT:?Set DATA_ROOT to the dataset directory}"
PYTHON="${PYTHON:-python}"

export MPLCONFIGDIR=/tmp/matplotlib
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

"${PYTHON}" train_urb3dcd_v2.py \
  --sensor 1-Lidar05 \
  --data_root "${DATA_ROOT}" \
  --model_arch ptv3_semantic_xy_xyz_relation_attention \
  --epochs 100 \
  --batch_size 1 \
  --num_points 4096 \
  --radius 50 \
  --voxel_size 0.3 \
  --train_samples_per_epoch 500 \
  --val_samples_per_epoch 200 \
  --test_samples_per_epoch 200 \
  --checkpoint outputs/urb3dcd_v2_fast500_semantic_xy_xyz_relation_attention.pth \
  --token_match_k 8 \
  --ptv3_order z,z-trans,hilbert,hilbert-trans \
  --ptv3_patch_size 1024 \
  --loss_type focal_ce \
  --lovasz_weight 1.0 \
  --binary_loss_weight 0.2 \
  --semantic_loss_weight 0.2
