import argparse
import copy
import csv
import os
import os.path as osp
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from cached_dataset import (
    CachedDataset,
    CDDatasetHKSCDForPipeline,
    CDDatasetUrb3DCDV2ForPipeline,
    MECPTFullCoverageTileDataset,
    STREET_CLASS_LABELS,
)
from changemamba import (
    JointTemporalMambaPoint,
    JointTemporalMatchingMambaPoint,
    NoDirectionFusionMambaPoint,
    PTV3CleanChangePoint,
    PTV3DirectionFusionPoint,
    PTV3MatchingEvidencePoint,
    PTV3RelationAttentionPoint,
    PTV3SemanticRelationAttentionPoint,
    PTV3T1AnchoredSemanticXYXYZRelationAttentionPoint,
    PTV3RelationEncoderPoint,
    PTV3RelationFusionPoint,
    PTV3SiameseConcatPoint,
    PTV3UpdatePTV3RefinePoint,
    PTV3UpdateRefinePoint,
    PTV3XYXYZRelationAttentionPoint,
    SimpleDirectionFusionMambaPoint,
    TwoStageInteractionMambaPoint,
)
from metric import ConfusionMatrix


NUM_CLASS = len(STREET_CLASS_LABELS)
TARGET_MODE = "pair_3class"
TEMPORAL_NUM_CLASS = 4
TEMPORAL_CLASS_LABELS = [
    "t0_unchanged",
    "removed",
    "t1_unchanged",
    "added",
]
IGNORE_INDEX = -1
ORIGINAL_STDERR = sys.stderr


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_log_file(args):
    if args.disable_log_file:
        return None
    log_path = args.log_path or osp.splitext(args.checkpoint)[0] + ".log"
    log_dir = osp.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    handle = open(log_path, "w", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, handle)
    sys.stderr = TeeStream(sys.stderr, handle)
    print(f"Logging to: {log_path}")
    return handle


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("PointMamba training requires CUDA.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_task(args):
    global NUM_CLASS, STREET_CLASS_LABELS, TARGET_MODE
    urb3dcd_v2_all_labels = getattr(args, "urb3dcd_v2_all_labels", False)
    legacy_nyc_all_labels = getattr(args, "nyc_all_labels", False)
    use_urb3dcd_v2_all_labels = args.dataset in ("nyc", "urb3dcd_v2") and (
        legacy_nyc_all_labels or urb3dcd_v2_all_labels
    )
    if use_urb3dcd_v2_all_labels:
        if args.train_samples_per_epoch is None:
            args.train_samples_per_epoch = 6000
        if args.val_samples_per_epoch is None:
            args.val_samples_per_epoch = 1000
        if args.test_samples_per_epoch is None:
            args.test_samples_per_epoch = 1000
        STREET_CLASS_LABELS = [
            "unchanged",
            "new_building",
            "demolition",
            "new_vegetation",
            "vegetation_growth",
            "missing_vegetation",
            "mobile_objects",
        ]
        NUM_CLASS = len(STREET_CLASS_LABELS)
        TARGET_MODE = "label1_multiclass"
    elif args.dataset == "hkscd":
        if args.train_samples_per_epoch is None:
            args.train_samples_per_epoch = 500
        if args.val_samples_per_epoch is None:
            args.val_samples_per_epoch = 200
        if args.test_samples_per_epoch is None:
            args.test_samples_per_epoch = 1000
        STREET_CLASS_LABELS = ["unchanged", "removed", "added"]
        NUM_CLASS = len(STREET_CLASS_LABELS)
        TARGET_MODE = "pair_3class"
        args.semantic_num_classes = int(getattr(args, "semantic_num_classes", 6))
    else:
        STREET_CLASS_LABELS = ["unchanged", "removed", "added"]
        NUM_CLASS = len(STREET_CLASS_LABELS)
        if (
            args.dataset == "slpccd"
            and getattr(args, "slpccd_sample_mode", "street")
            == "mecpt_full_pair"
        ):
            if not args.model_arch.startswith("ptv3_t1_anchored_"):
                raise ValueError(
                    "SLPCCD ME-CPT full-pair sampling produces unequal T0/T1 "
                    "point counts and requires a ptv3_t1_anchored_* model."
                )
            if args.batch_size != 1:
                raise ValueError(
                    "SLPCCD ME-CPT full-pair sampling requires --batch_size 1 "
                    "because complete object pairs have variable point counts."
                )
            TARGET_MODE = "anchored_pair_3class"
        else:
            TARGET_MODE = "pair_3class"


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def swap_branches(batch):
    swapped = dict(batch)
    swapped["coord0"], swapped["coord1"] = batch["coord1"], batch["coord0"]
    swapped["feat0"], swapped["feat1"] = batch["feat1"], batch["feat0"]
    swapped["label0"] = batch["label1"].clone()
    swapped["label1"] = batch["label0"].clone()
    swapped["label0"][batch["label1"] == 1] = 2
    swapped["label0"][batch["label1"] == 2] = 1
    swapped["label1"][batch["label0"] == 1] = 2
    swapped["label1"][batch["label0"] == 2] = 1
    # Preserve absolute epoch identity when an anchored model is evaluated in
    # reverse, so original-T0 changes remain removed rather than becoming added.
    swapped["_time_value0"] = batch.get("_time_value1", 1.0)
    swapped["_time_value1"] = batch.get("_time_value0", 0.0)
    if "semantic0" in batch and "semantic1" in batch:
        swapped["semantic0"] = batch["semantic1"]
        swapped["semantic1"] = batch["semantic0"]
    return swapped


def build_pair_targets(batch):
    label0 = batch["label0"].long()
    label1 = batch["label1"].long()
    target = torch.full_like(label1, IGNORE_INDEX)
    valid = (label0 != IGNORE_INDEX) & (label1 != IGNORE_INDEX)
    target[valid] = 0
    target[valid & (label0 == 1)] = 1
    target[valid & (label1 == 2)] = 2
    return target


def build_label1_targets(batch):
    target = batch["label1"].long().clone()
    invalid = (target < 0) | (target >= NUM_CLASS)
    target[invalid] = IGNORE_INDEX
    return target


def build_anchored_pair_targets(batch):
    """Build global 3-class labels on the current branch-1 anchors."""
    label1 = batch["label1"].long()
    target = torch.full_like(label1, IGNORE_INDEX)
    valid = label1 != IGNORE_INDEX
    target[valid & (label1 == 0)] = 0
    anchor_time = float(batch.get("_time_value1", 1.0))
    changed_class = 2 if anchor_time >= 0.5 else 1
    target[valid & (label1 != 0)] = changed_class
    return target


def build_main_targets(batch):
    if TARGET_MODE == "label1_multiclass":
        return build_label1_targets(batch)
    if TARGET_MODE == "anchored_pair_3class":
        return build_anchored_pair_targets(batch)
    return build_pair_targets(batch)


def build_temporal_targets(batch, unchanged_class=2):
    label0 = batch["label0"].long()
    label1 = batch["label1"].long()
    target = torch.full_like(label1, IGNORE_INDEX)
    valid = (label0 != IGNORE_INDEX) & (label1 != IGNORE_INDEX)
    target[valid] = unchanged_class
    target[valid & (label0 == 1)] = 1
    target[valid & (label1 == 2)] = 3
    return target


def merge_temporal_to_semantic(values):
    merged = values.clone()
    valid = merged != IGNORE_INDEX
    merged[valid & (merged == 2)] = 0
    merged[valid & (merged == 3)] = 2
    return merged


def build_targets(batch, use_temporal_4class=False, unchanged_class=2):
    if use_temporal_4class:
        return build_temporal_targets(batch, unchanged_class=unchanged_class)
    return build_pair_targets(batch)


def parse_ptv3_order(raw_order):
    order = tuple(part.strip() for part in raw_order.split(",") if part.strip())
    if not order:
        raise ValueError("--ptv3_order must contain at least one order.")
    return order


def parse_class_weight_boosts(raw_boosts):
    boosts = {}
    if not raw_boosts:
        return boosts
    for item in raw_boosts.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "--class_weight_boosts entries must look like class_id:factor"
            )
        class_id_raw, factor_raw = item.split(":", 1)
        class_id = int(class_id_raw)
        factor = float(factor_raw)
        if class_id < 0:
            raise ValueError("class_weight_boosts class ids must be non-negative.")
        if factor <= 0:
            raise ValueError("class_weight_boosts factors must be positive.")
        boosts[class_id] = factor
    return boosts


def flatten_logits(logits):
    return logits.reshape(-1, logits.size(-1))


def flatten_targets(target):
    return target.reshape(-1)


class FocalCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if weight is not None:
            self.register_buffer("weight", weight.clone().detach().float())
        else:
            self.weight = None

    def forward(self, logits, target):
        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        logits = logits[valid]
        target = target[valid]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - target_probs).pow(self.gamma)
        alpha = self.weight[target] if self.weight is not None else 1.0
        return (-alpha * focal_factor * target_log_probs).mean()


class PlainCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.ignore_index = ignore_index
        if weight is not None:
            self.register_buffer("weight", weight.clone().detach().float())
        else:
            self.weight = None

    def forward(self, logits, target):
        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        return torch.nn.functional.cross_entropy(
            logits[valid],
            target[valid],
            weight=self.weight,
        )


class LovaszSoftmaxLoss(torch.nn.Module):
    def __init__(self, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.ignore_index = ignore_index

    @staticmethod
    def _lovasz_grad(gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1.0 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union.clamp_min(1e-6)
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def forward(self, logits, target):
        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        probs = torch.softmax(logits[valid], dim=-1)
        labels = target[valid]
        losses = []
        for class_idx in labels.unique():
            fg = (labels == class_idx).type_as(probs)
            class_prob = probs[:, class_idx]
            errors = (fg - class_prob).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, self._lovasz_grad(fg_sorted)))
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class CombinedSegLoss(torch.nn.Module):
    def __init__(self, ce_loss, lovasz_weight=0.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.ce_loss = ce_loss
        self.lovasz_weight = lovasz_weight
        self.lovasz_loss = LovaszSoftmaxLoss(ignore_index=ignore_index)

    def forward(self, logits, target):
        loss = self.ce_loss(logits, target)
        if self.lovasz_weight > 0.0:
            loss = loss + self.lovasz_weight * self.lovasz_loss(logits, target)
        return loss


def build_model(args, device):
    prior_mode = getattr(args, "correspondence_prior", "full")
    if prior_mode != "full" and args.model_arch != "ptv3_xy_xyz_relation_attention":
        raise ValueError("Prior ablation requires ptv3_xy_xyz_relation_attention")
    prior_injection_mode = getattr(args, "prior_injection_mode", "p5_full")
    if (
        prior_injection_mode != "p5_full"
        and args.model_arch != "ptv3_semantic_xy_xyz_relation_attention"
    ):
        raise ValueError(
            "P1-P4 prior-injection ablations require "
            "ptv3_semantic_xy_xyz_relation_attention"
        )
    ptv3_order = parse_ptv3_order(args.ptv3_order)
    semantic_num_classes = int(getattr(args, "semantic_num_classes", 4))
    if args.model_arch == "ptv3_siamese_concat":
        return PTV3SiameseConcatPoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
        ).to(device)
    if args.model_arch == "ptv3_matching_evidence":
        return PTV3MatchingEvidencePoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
        ).to(device)
    if args.model_arch == "joint":
        return JointTemporalMambaPoint(
            num_classes=NUM_CLASS,
            num_group=args.num_group,
            group_size=args.group_size,
            signed_diff=True,
        ).to(device)
    if args.model_arch == "joint_match":
        return JointTemporalMatchingMambaPoint(
            num_classes=NUM_CLASS,
            token_match_k=args.token_match_k,
            token_match_tau_feat=args.token_match_tau_feat,
            token_match_tau_geo=args.token_match_tau_geo,
            token_match_detach_scores=not args.token_match_score_grad,
            num_group=args.num_group,
            group_size=args.group_size,
            signed_diff=True,
        ).to(device)
    if args.model_arch == "two_stage":
        return TwoStageInteractionMambaPoint(
            num_classes=NUM_CLASS,
            token_match_k=args.token_match_k,
            token_match_tau_feat=args.token_match_tau_feat,
            token_match_tau_geo=args.token_match_tau_geo,
            token_match_detach_scores=not args.token_match_score_grad,
            num_group=args.num_group,
            group_size=args.group_size,
            signed_diff=True,
            stage2_depth=args.stage2_depth,
            use_temporal_embedding=args.temporal_embedding,
            use_existence_branch=args.existence_loss_weight > 0.0,
            use_point_decoder=args.use_point_decoder,
        ).to(device)
    if args.model_arch == "no_direction":
        return NoDirectionFusionMambaPoint(
            num_classes=NUM_CLASS,
            token_match_k=args.token_match_k,
            token_match_tau_feat=args.token_match_tau_feat,
            token_match_tau_geo=args.token_match_tau_geo,
            token_match_detach_scores=not args.token_match_score_grad,
            num_group=args.num_group,
            group_size=args.group_size,
            signed_diff=True,
        ).to(device)
    if args.model_arch == "ptv3_match":
        return PTV3DirectionFusionPoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            direction_aware=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
            use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
            use_removal_like_auxiliary_loss=args.removal_loss_weight > 0.0,
        ).to(device)
    if args.model_arch == "ptv3_no_direction":
        return PTV3DirectionFusionPoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            direction_aware=False,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
            use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
            use_removal_like_auxiliary_loss=args.removal_loss_weight > 0.0,
        ).to(device)
    if args.model_arch == "ptv3_relation_fusion":
        return PTV3RelationFusionPoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
            use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
            use_removal_like_auxiliary_loss=args.removal_loss_weight > 0.0,
        ).to(device)
    if args.model_arch == "ptv3_relation_encoder":
        return PTV3RelationEncoderPoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
            use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
            use_removal_like_auxiliary_loss=args.removal_loss_weight > 0.0,
        ).to(device)
    if args.model_arch == "ptv3_relation_attention":
        return PTV3RelationAttentionPoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
            use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
            use_removal_like_auxiliary_loss=args.removal_loss_weight > 0.0,
            use_semantic_auxiliary_loss=args.semantic_loss_weight > 0.0,
            semantic_num_classes=semantic_num_classes,
        ).to(device)
    if args.model_arch == "ptv3_prior_attention_only":
        return PTV3RelationAttentionPoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
            use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
            use_removal_like_auxiliary_loss=args.removal_loss_weight > 0.0,
            use_semantic_auxiliary_loss=False,
            semantic_num_classes=semantic_num_classes,
            use_symmetric_xyz_relation=False,
        ).to(device)
    if args.model_arch in (
        "ptv3_xy_xyz_relation_attention",
        "ptv3_xy_xyz_concat",
        "ptv3_xyz_xy_relation_attention",
        "ptv3_xyz_xy_height_relation_attention",
    ):
        return PTV3XYXYZRelationAttentionPoint(
            correspondence_prior=prior_mode,
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
            use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
            use_removal_like_auxiliary_loss=args.removal_loss_weight > 0.0,
            use_height_cues=(
                args.model_arch != "ptv3_xyz_xy_relation_attention"
            ),
            use_adaptive_relation_fusion=(
                args.model_arch != "ptv3_xy_xyz_concat"
            ),
        ).to(device)
    if args.model_arch == "ptv3_semantic_relation_attention":
        return PTV3SemanticRelationAttentionPoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
            use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
            use_removal_like_auxiliary_loss=args.removal_loss_weight > 0.0,
            semantic_num_classes=semantic_num_classes,
        ).to(device)
    if args.model_arch == "ptv3_semantic_xy_xyz_concat":
        return PTV3SemanticRelationAttentionPoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
            use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
            use_removal_like_auxiliary_loss=args.removal_loss_weight > 0.0,
            semantic_num_classes=semantic_num_classes,
            use_xy_xyz_prior=True,
            use_adaptive_semantic_spatial_fusion=False,
        ).to(device)
    if args.model_arch in (
        "ptv3_semantic_xy_xyz_relation_attention",
        "ptv3_semantic_xy_xyz_ordered_transition_attention",
        "ptv3_semantic_xy_xyz_residual_ordered_transition_attention",
    ):
        return PTV3SemanticRelationAttentionPoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
            use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
            use_removal_like_auxiliary_loss=args.removal_loss_weight > 0.0,
            semantic_num_classes=semantic_num_classes,
            use_xy_xyz_prior=True,
            use_ordered_semantic_probability_transition=(
                args.model_arch
                in (
                    "ptv3_semantic_xy_xyz_ordered_transition_attention",
                    "ptv3_semantic_xy_xyz_residual_ordered_transition_attention",
                )
            ),
            use_residual_ordered_semantic_transition=(
                args.model_arch
                == "ptv3_semantic_xy_xyz_residual_ordered_transition_attention"
            ),
            prior_injection_mode=prior_injection_mode,
        ).to(device)
    if args.model_arch in (
        "ptv3_t1_anchored_semantic_xy_xyz_relation_attention",
        "ptv3_t1_anchored_ordered_semantic_transition_attention",
        "ptv3_t1_anchored_local_vertical_profile_attention",
    ):
        return PTV3T1AnchoredSemanticXYXYZRelationAttentionPoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
            use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
            use_removal_like_auxiliary_loss=args.removal_loss_weight > 0.0,
            semantic_num_classes=semantic_num_classes,
            use_scalable_knn=getattr(args, "scalable_knn", False),
            match_detach_scores=not args.token_match_score_grad,
            match_query_chunk_size=args.match_query_chunk_size,
            use_t1_ordered_semantic_transition=(
                args.model_arch
                == "ptv3_t1_anchored_ordered_semantic_transition_attention"
            ),
            use_local_vertical_profile=(
                args.model_arch
                == "ptv3_t1_anchored_local_vertical_profile_attention"
            ),
        ).to(device)
    if args.model_arch == "ptv3_directional_semantic_xy_xyz_relation_attention":
        return PTV3SemanticRelationAttentionPoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
            use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
            use_removal_like_auxiliary_loss=args.removal_loss_weight > 0.0,
            semantic_num_classes=semantic_num_classes,
            use_xy_xyz_prior=True,
            use_directional_semantic_transition=True,
        ).to(device)
    if args.model_arch == "ptv3_update_refine":
        return PTV3UpdateRefinePoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
        ).to(device)
    if args.model_arch == "ptv3_clean_change":
        return PTV3CleanChangePoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
        ).to(device)
    if args.model_arch == "ptv3_update_ptv3_refine":
        return PTV3UpdatePTV3RefinePoint(
            num_classes=NUM_CLASS,
            match_k=args.token_match_k,
            match_tau_feat=args.token_match_tau_feat,
            match_tau_geo=args.token_match_tau_geo,
            signed_diff=True,
            ptv3_order=ptv3_order,
            ptv3_patch_size=args.ptv3_patch_size,
        ).to(device)

    return SimpleDirectionFusionMambaPoint(
        num_classes=NUM_CLASS,
        token_match_k=args.token_match_k,
        token_match_tau_feat=args.token_match_tau_feat,
        token_match_tau_geo=args.token_match_tau_geo,
        token_match_detach_scores=not args.token_match_score_grad,
        num_group=args.num_group,
        group_size=args.group_size,
        signed_diff=True,
        use_binary_change_auxiliary_loss=args.binary_loss_weight > 0.0,
        use_changed_direction_auxiliary_loss=args.changed_direction_loss_weight > 0.0,
        use_temporal_embedding=args.temporal_embedding,
        use_temporal_4class_auxiliary_loss=args.temporal_aux_loss_weight > 0.0,
    ).to(device)


def make_seg_criterion(args, class_weights, device):
    if args.loss_type == "focal_ce":
        ce_criterion = FocalCrossEntropyLoss(
            weight=class_weights,
            gamma=args.focal_gamma,
            ignore_index=IGNORE_INDEX,
        )
    elif args.loss_type == "weighted_ce":
        ce_criterion = PlainCrossEntropyLoss(
            weight=class_weights,
            ignore_index=IGNORE_INDEX,
        )
    elif args.loss_type == "ce":
        ce_criterion = PlainCrossEntropyLoss(
            weight=torch.ones(NUM_CLASS, device=device),
            ignore_index=IGNORE_INDEX,
        )
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")
    return CombinedSegLoss(
        ce_criterion,
        lovasz_weight=args.lovasz_weight,
        ignore_index=IGNORE_INDEX,
    )


def load_warm_start_weights(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    if not osp.exists(checkpoint_path):
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")
    source_state = torch.load(checkpoint_path, map_location=device)
    target_state = model.state_dict()
    loadable_state = {}
    skipped = []
    for name, tensor in source_state.items():
        if name not in target_state:
            skipped.append((name, "missing_in_target"))
            continue
        if target_state[name].shape != tensor.shape:
            skipped.append(
                (
                    name,
                    f"shape {tuple(tensor.shape)} -> {tuple(target_state[name].shape)}",
                )
            )
            continue
        loadable_state[name] = tensor
    missing, unexpected = model.load_state_dict(loadable_state, strict=False)
    print(f"Warm-start checkpoint: {checkpoint_path}")
    print(f"Warm-start loaded tensors: {len(loadable_state)} / {len(target_state)}")
    print(f"Warm-start missing tensors after partial load: {len(missing)}")
    print(f"Warm-start unexpected tensors after partial load: {len(unexpected)}")
    if skipped:
        print("Warm-start skipped tensors:")
        for name, reason in skipped[:30]:
            print(f"  {name}: {reason}")
        if len(skipped) > 30:
            print(f"  ... {len(skipped) - 30} more skipped tensors")


def freeze_ptv3_encoder_if_requested(model, freeze_encoder):
    if not freeze_encoder:
        return
    if not hasattr(model, "encoder"):
        print("--freeze_ptv3_encoder was set, but this model has no encoder attribute.")
        return
    frozen = 0
    for param in model.encoder.parameters():
        param.requires_grad = False
        frozen += param.numel()
    model.encoder.eval()
    model._keep_encoder_eval = True
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Frozen PTv3 encoder parameters: {frozen}")
    print(f"Trainable parameters after freezing: {trainable}")


def compute_class_counts(samples):
    counts = torch.zeros(NUM_CLASS, dtype=torch.float32)
    for sample in samples:
        labels = torch.cat(
            [sample["label0"].view(-1).long(), sample["label1"].view(-1).long()],
            dim=0,
        )
        labels = labels[(labels >= 0) & (labels < NUM_CLASS)]
        if labels.numel() > 0:
            counts += torch.bincount(labels, minlength=NUM_CLASS).float()
    return counts.clamp_min(1.0)


def compute_sampled_class_counts(dataset):
    counts = torch.zeros(NUM_CLASS, dtype=torch.float32)
    for idx in tqdm(range(len(dataset)), desc="class weights", leave=False, file=ORIGINAL_STDERR):
        sample = dataset[idx]
        labels = torch.cat(
            [sample["label0"].view(-1).long(), sample["label1"].view(-1).long()],
            dim=0,
        )
        labels = labels[(labels >= 0) & (labels < NUM_CLASS)]
        if labels.numel() > 0:
            counts += torch.bincount(labels, minlength=NUM_CLASS).float()
    return counts.clamp_min(1.0)


def compute_temporal_class_counts_from_samples(samples):
    counts = torch.zeros(TEMPORAL_NUM_CLASS, dtype=torch.float32)
    for sample in samples:
        label0 = sample["label0"].view(-1).long()
        label1 = sample["label1"].view(-1).long()

        # Raw cached samples are not paired point-by-point: T0 and T1 can have
        # different lengths. Count branch labels independently here. Exact
        # paired counts are computed by compute_sampled_temporal_class_counts.
        counts[0] += (label0 == 0).sum().float()
        counts[1] += (label0 == 1).sum().float()
        counts[2] += (label1 == 0).sum().float()
        counts[3] += (label1 == 2).sum().float()

        if label0.numel() == label1.numel():
            # Also include the swapped training direction when a sample already
            # has paired point tensors, e.g. after sampling.
            for unchanged_class, current_sample in (
                (2, sample),
                (0, swap_branches(sample)),
            ):
                target = build_temporal_targets(current_sample, unchanged_class=unchanged_class)
                target = target.view(-1)
                target = target[(target >= 0) & (target < TEMPORAL_NUM_CLASS)]
                if target.numel() > 0:
                    counts += torch.bincount(target, minlength=TEMPORAL_NUM_CLASS).float()
            continue

        # Swapped direction maps original T1 added to removed and original T0
        # removed to added; unchanged belongs to the swapped T0 unchanged class.
        counts[0] += (label1 == 0).sum().float()
        counts[1] += (label1 == 2).sum().float()
        counts[2] += (label0 == 0).sum().float()
        counts[3] += (label0 == 1).sum().float()
    return counts.clamp_min(1.0)


def compute_sampled_temporal_class_counts(dataset):
    counts = torch.zeros(TEMPORAL_NUM_CLASS, dtype=torch.float32)
    for idx in tqdm(range(len(dataset)), desc="temporal class weights", leave=False, file=ORIGINAL_STDERR):
        sample = dataset[idx]
        for unchanged_class, current_sample in (
            (2, sample),
            (0, swap_branches(sample)),
        ):
            target = build_temporal_targets(current_sample, unchanged_class=unchanged_class)
            target = target.view(-1)
            target = target[(target >= 0) & (target < TEMPORAL_NUM_CLASS)]
            if target.numel() > 0:
                counts += torch.bincount(target, minlength=TEMPORAL_NUM_CLASS).float()
    return counts.clamp_min(1.0)


def compute_class_weights(counts, min_weight, max_weight):
    weights = counts.sum() / (counts.numel() * counts)
    weights = weights / weights.mean()
    return torch.clamp(weights, min=min_weight, max=max_weight)


def compute_misnet_semantic_weights(samples, num_semantic_classes=4):
    counts = torch.zeros(num_semantic_classes, dtype=torch.float64)
    for sample in samples:
        for key in ("semantic0", "semantic1"):
            if key not in sample:
                continue
            target = sample[key].reshape(-1).long()
            valid = (target >= 0) & (target < num_semantic_classes)
            if torch.any(valid):
                counts += torch.bincount(
                    target[valid], minlength=num_semantic_classes
                ).to(dtype=counts.dtype)
    if counts.sum() <= 0:
        raise ValueError(
            "MISNet semantic weighting requires semantic0/semantic1 labels "
            "in the training cache."
        )
    frequencies = counts / counts.sum()
    return (1.0 - frequencies).float(), counts


def compute_sample_weights(dataset, class_weights, focus_classes, focus_power):
    class_weights = class_weights.cpu()
    sample_weights = []
    for sample in dataset.samples:
        labels = torch.cat(
            [sample["label0"].view(-1).long(), sample["label1"].view(-1).long()],
            dim=0,
        )
        labels = labels[(labels >= 0) & (labels < NUM_CLASS)]
        if labels.numel() == 0:
            sample_weights.append(1.0)
            continue
        hist = torch.bincount(labels, minlength=NUM_CLASS).float()
        base = float((hist * class_weights).sum() / hist.sum().clamp_min(1.0))
        focus = float(hist[list(focus_classes)].sum() / hist.sum().clamp_min(1.0))
        sample_weights.append(base * (1.0 + focus_power * focus))
    return torch.as_tensor(sample_weights, dtype=torch.double)


def parse_class_ids(raw_value):
    class_ids = []
    for part in raw_value.split(","):
        part = part.strip()
        if part:
            class_ids.append(int(part))
    if not class_ids:
        raise ValueError("sample_focus_classes cannot be empty.")
    for class_id in class_ids:
        if not 0 <= class_id < NUM_CLASS:
            raise ValueError(f"class id {class_id} is out of range.")
    return tuple(class_ids)


def make_dataset(args, split, cache_path):
    if args.dataset == "hkscd":
        virtual_sample_count = {
            "train": args.train_samples_per_epoch,
            "val": args.val_samples_per_epoch,
            "test": args.test_samples_per_epoch,
        }[split]
        return CachedDataset(
            cache_path=cache_path,
            dataset_cls=CDDatasetHKSCDForPipeline,
            data_root=args.data_root,
            split=split,
            num_points=args.num_points,
            change_ratio=args.change_ratio,
            remap_labels=False,
            sample_mode=args.hkscd_sample_mode,
            virtual_sample_count=virtual_sample_count,
            cylinder_radius=args.hkscd_radius,
            cylinder_voxel_size=args.hkscd_voxel_size,
            require_semantic_labels=args.semantic_loss_weight > 0.0,
        )
    if args.dataset in ("nyc", "urb3dcd_v2"):
        use_all_change_labels = (
            getattr(args, "nyc_all_labels", False)
            or getattr(args, "urb3dcd_v2_all_labels", False)
        )
        removed_labels = getattr(
            args,
            "urb3dcd_removed_labels",
            getattr(args, "nyc_removed_labels", "1"),
        )
        added_labels = getattr(
            args,
            "urb3dcd_added_labels",
            getattr(args, "nyc_added_labels", "2,3"),
        )
        ignore_labels = getattr(
            args,
            "urb3dcd_ignore_labels",
            getattr(args, "nyc_ignore_labels", "4,5,6"),
        )
        virtual_sample_count = {
            "train": args.train_samples_per_epoch,
            "val": args.val_samples_per_epoch,
            "test": args.test_samples_per_epoch,
        }[split]
        sample_mode = args.urb3dcd_sample_mode if use_all_change_labels else "street"
        if use_all_change_labels:
            split_sample_mode = getattr(
                args,
                f"urb3dcd_{split}_sample_mode",
                None,
            )
            if split_sample_mode:
                sample_mode = split_sample_mode
        return CachedDataset(
            cache_path=cache_path,
            dataset_cls=CDDatasetUrb3DCDV2ForPipeline,
            data_root=args.data_root,
            split=split,
            num_points=args.num_points,
            change_ratio=args.change_ratio,
            remap_labels=not use_all_change_labels,
            sample_mode=sample_mode,
            virtual_sample_count=virtual_sample_count,
            cylinder_radius=args.urb3dcd_radius,
            cylinder_voxel_size=args.urb3dcd_voxel_size,
            require_semantic_labels=args.semantic_loss_weight > 0.0,
            removed_labels=removed_labels,
            added_labels=added_labels,
            ignore_labels=ignore_labels,
            use_all_change_labels=use_all_change_labels,
            sensor=args.urb3dcd_sensor,
        )
    sample_mode = getattr(args, "slpccd_sample_mode", "street")
    split_mode = getattr(args, f"slpccd_{split}_sample_mode", None)
    if split_mode is not None:
        sample_mode = split_mode
    return CachedDataset(
        cache_path=cache_path,
        data_root=args.data_root,
        split=split,
        num_points=args.num_points,
        change_ratio=args.change_ratio,
        sample_mode=sample_mode,
        street_voxel_size=getattr(args, "slpccd_voxel_size", 0.0),
        street_coverage_voxel_size=getattr(
            args, "slpccd_coverage_voxel_size", 0.1
        ),
    )


def make_train_loader(args, dataset, class_weights, focus_classes):
    if getattr(args, "train_sampling_strategy", "weighted") == "uniform":
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    sample_weights = compute_sample_weights(
        dataset,
        class_weights,
        focus_classes=focus_classes,
        focus_power=args.sample_focus_power,
    )
    if len(sample_weights) != len(dataset):
        repeat = int(np.ceil(len(dataset) / max(len(sample_weights), 1)))
        sample_weights = sample_weights.repeat(repeat)[: len(dataset)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )


class FixedSampleDataset(Dataset):
    """Materialize random crops once so validation is identical every epoch."""

    def __init__(self, dataset, seed, stratified=False, cache_path=None):
        if cache_path and osp.exists(cache_path):
            payload = torch.load(cache_path, map_location="cpu")
            self.samples = payload["samples"]
            self.center_labels = payload.get("center_labels", [])
            print(f"Loaded frozen evaluation samples: {cache_path}")
            return
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            center_labels = dataset.available_center_labels() if stratified else []
            if stratified and not center_labels:
                raise ValueError(
                    "Stratified validation requires a cylinder dataset with valid center labels."
                )
            frozen_samples = []
            for idx in tqdm(
                range(len(dataset)),
                desc="freezing validation samples",
                leave=False,
                file=ORIGINAL_STDERR,
            ):
                if center_labels:
                    center_label = center_labels[idx % len(center_labels)]
                    frozen_samples.append(dataset.sample_for_center_label(center_label))
                else:
                    frozen_samples.append(dataset[idx])
            self.samples = frozen_samples
            self.center_labels = center_labels
            if cache_path:
                cache_dir = osp.dirname(osp.abspath(cache_path))
                os.makedirs(cache_dir, exist_ok=True)
                torch.save(
                    {
                        "samples": self.samples,
                        "center_labels": self.center_labels,
                        "seed": int(seed),
                    },
                    cache_path,
                )
                print(f"Saved frozen evaluation samples: {cache_path}")
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_eval_loader(args, split):
    cache_path = {
        "train": args.train_cache_path,
        "val": args.val_cache_path,
        "test": args.test_cache_path,
    }[split]
    dataset = make_dataset(args, split, cache_path)
    urb3dcd_mode = getattr(args, f"urb3dcd_{split}_sample_mode", None)
    if urb3dcd_mode == "mecpt_full_scene_tiled":
        dataset = MECPTFullCoverageTileDataset(
            dataset,
            voxel_size=args.urb3dcd_voxel_size,
            tile_size=getattr(args, "mecpt_tile_size", 200.0),
            halo=getattr(args, "mecpt_tile_halo", 20.0),
        )
        print(
            f"ME-CPT full-coverage test: {len(dataset)} deterministic tiles, "
            f"tile={getattr(args, 'mecpt_tile_size', 200.0):g} m, "
            f"halo={getattr(args, 'mecpt_tile_halo', 20.0):g} m"
        )
    if split == "val" and getattr(args, "fixed_val_sampling", False):
        val_sampling_seed = getattr(args, "val_sampling_seed", args.seed + 1000)
        stratified = getattr(args, "stratified_val_sampling", False)
        dataset = FixedSampleDataset(
            dataset,
            seed=val_sampling_seed,
            stratified=stratified,
        )
        print(
            f"Fixed validation set: {len(dataset)} cylinders "
            f"(sampling seed {val_sampling_seed}, stratified={stratified})"
        )
    elif getattr(args, "fixed_eval_sampling", False):
        eval_sampling_seed = getattr(args, "eval_sampling_seed", args.seed + 3000)
        dataset = FixedSampleDataset(
            dataset,
            seed=eval_sampling_seed,
            stratified=False,
            cache_path=getattr(args, "fixed_eval_cache_path", None),
        )
        print(
            f"Fixed {split} set: {len(dataset)} cylinders "
            f"(sampling seed {eval_sampling_seed}, stratified=False)"
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def summarize_confusion(confusion_matrix, labels):
    cm = confusion_matrix.confusion_matrix
    row_sums = cm.sum(axis=1)
    for class_idx, class_name in enumerate(labels):
        total = int(row_sums[class_idx])
        correct = int(cm[class_idx, class_idx])
        acc = 0.0 if total == 0 else correct / total
        mistakes = []
        for pred_idx, count in enumerate(cm[class_idx]):
            if pred_idx != class_idx and count > 0:
                mistakes.append((int(count), pred_idx))
        mistakes.sort(reverse=True)
        detail = ", ".join(
            f"{labels[pred_idx]}: {count}" for count, pred_idx in mistakes[:2]
        )
        detail = detail or "none"
        print(f"{class_name}: acc={acc:.4f}, correct={correct}/{total}, confused_as=[{detail}]")


def flatten_aux_logits(logits):
    return logits.reshape(-1, logits.size(-1))


def compute_auxiliary_losses(
    model_output,
    target,
    temporal_target=None,
    semantic_target0=None,
    semantic_target1=None,
    semantic_class_weights=None,
    relation_loss_margin=1.0,
):
    if not isinstance(model_output, dict):
        return None, None, None, None, None, None, None

    target_flat = flatten_targets(target)
    valid = target_flat != IGNORE_INDEX

    binary_loss = None
    if "binary_change_logits" in model_output:
        binary_logits = flatten_aux_logits(model_output["binary_change_logits"])
        if torch.any(valid):
            binary_target = (target_flat[valid] > 0).long()
            binary_loss = torch.nn.functional.cross_entropy(
                binary_logits[valid],
                binary_target,
            )
        else:
            binary_loss = binary_logits.sum() * 0.0

    removal_loss = None
    if "removal_like_logits" in model_output:
        removal_logits = flatten_aux_logits(model_output["removal_like_logits"])
        if torch.any(valid):
            if TARGET_MODE == "label1_multiclass" and NUM_CLASS >= 6:
                removal_target = (
                    (target_flat[valid] == 2) | (target_flat[valid] == 5)
                ).long()
            else:
                removal_target = (target_flat[valid] == 1).long()
            removal_loss = torch.nn.functional.cross_entropy(
                removal_logits[valid],
                removal_target,
            )
        else:
            removal_loss = removal_logits.sum() * 0.0

    relation_loss = None
    if "relation_residual" in model_output:
        relation_residual = model_output["relation_residual"].reshape(-1)
        if torch.any(valid):
            residual = relation_residual[valid]
            changed = target_flat[valid] > 0
            unchanged = ~changed
            loss_terms = []
            if torch.any(unchanged):
                loss_terms.append(residual[unchanged].pow(2).mean())
            if torch.any(changed):
                margin = float(relation_loss_margin)
                loss_terms.append(
                    torch.relu(margin - residual[changed]).pow(2).mean()
                )
            relation_loss = sum(loss_terms) / max(len(loss_terms), 1)
        else:
            relation_loss = relation_residual.sum() * 0.0

    direction_loss = None
    if "changed_direction_logits" in model_output:
        direction_logits = flatten_aux_logits(model_output["changed_direction_logits"])
        changed = (target_flat == 1) | (target_flat == 2)
        if torch.any(changed):
            direction_target = (target_flat[changed] == 2).long()
            direction_loss = torch.nn.functional.cross_entropy(
                direction_logits[changed],
                direction_target,
            )
        else:
            direction_loss = direction_logits.sum() * 0.0

    temporal_aux_loss = None
    if "temporal_4class_logits" in model_output:
        temporal_logits = flatten_aux_logits(model_output["temporal_4class_logits"])
        if temporal_target is None:
            temporal_aux_loss = temporal_logits.sum() * 0.0
        else:
            temporal_target_flat = flatten_targets(temporal_target)
            temporal_valid = temporal_target_flat != IGNORE_INDEX
            if torch.any(temporal_valid):
                temporal_aux_loss = torch.nn.functional.cross_entropy(
                    temporal_logits[temporal_valid],
                    temporal_target_flat[temporal_valid],
                )
            else:
                temporal_aux_loss = temporal_logits.sum() * 0.0

    existence_loss = None
    if "existence_logits_t0" in model_output and "existence_logits_t1" in model_output:
        existence_logits_t0 = flatten_aux_logits(model_output["existence_logits_t0"])
        existence_logits_t1 = flatten_aux_logits(model_output["existence_logits_t1"])
        if torch.any(valid):
            # Class 1 means the current-time point has a correspondence in the
            # other time. Removed points break T0->T1 existence; added points
            # break T1->T0 existence.
            exists_t0_in_t1 = (target_flat[valid] != 1).long()
            exists_t1_in_t0 = (target_flat[valid] != 2).long()
            existence_loss_t0 = torch.nn.functional.cross_entropy(
                existence_logits_t0[valid],
                exists_t0_in_t1,
            )
            existence_loss_t1 = torch.nn.functional.cross_entropy(
                existence_logits_t1[valid],
                exists_t1_in_t0,
            )
            existence_loss = 0.5 * (existence_loss_t0 + existence_loss_t1)
        else:
            existence_loss = (
                existence_logits_t0.sum() + existence_logits_t1.sum()
            ) * 0.0

    semantic_loss = None
    semantic_pairs = (
        ("semantic_logits_t0", semantic_target0),
        ("semantic_logits_t1", semantic_target1),
    )
    semantic_terms = []
    for logits_key, semantic_target in semantic_pairs:
        if logits_key not in model_output or semantic_target is None:
            continue
        semantic_logits = flatten_aux_logits(model_output[logits_key])
        semantic_target_flat = flatten_targets(semantic_target)
        semantic_valid = (
            (semantic_target_flat >= 0)
            & (semantic_target_flat < semantic_logits.size(-1))
        )
        if torch.any(semantic_valid):
            semantic_terms.append(
                torch.nn.functional.cross_entropy(
                    semantic_logits[semantic_valid],
                    semantic_target_flat[semantic_valid],
                    weight=semantic_class_weights,
                )
            )
        else:
            semantic_terms.append(semantic_logits.sum() * 0.0)
    if semantic_terms:
        semantic_loss = sum(semantic_terms) / len(semantic_terms)

    return (
        binary_loss,
        removal_loss,
        relation_loss,
        direction_loss,
        temporal_aux_loss,
        existence_loss,
        semantic_loss,
    )


def forward_one_direction(
    model,
    batch,
    criterion,
    binary_loss_weight=0.0,
    removal_loss_weight=0.0,
    relation_loss_weight=0.0,
    relation_loss_margin=1.0,
    changed_direction_loss_weight=0.0,
    temporal_aux_loss_weight=0.0,
    existence_loss_weight=0.0,
    semantic_loss_weight=0.0,
    semantic_class_weights=None,
    unchanged_class=2,
):
    target = build_main_targets(batch)
    temporal_target = (
        build_temporal_targets(batch, unchanged_class=unchanged_class)
        if TARGET_MODE == "pair_3class"
        else None
    )
    model_output = model(batch)
    logits = model_output["point_logits"] if isinstance(model_output, dict) else model_output
    logits = flatten_logits(logits)
    target_flat = flatten_targets(target)
    tri_loss = criterion(logits, target_flat)
    (
        binary_loss,
        removal_loss,
        relation_loss,
        direction_loss,
        temporal_aux_loss,
        existence_loss,
        semantic_loss,
    ) = compute_auxiliary_losses(
        model_output,
        target,
        temporal_target=temporal_target,
        semantic_target0=batch.get("semantic0"),
        semantic_target1=batch.get("semantic1"),
        semantic_class_weights=semantic_class_weights,
        relation_loss_margin=relation_loss_margin,
    )

    loss = tri_loss
    if binary_loss is not None and binary_loss_weight > 0.0:
        loss = loss + binary_loss_weight * binary_loss
    if removal_loss is not None and removal_loss_weight > 0.0:
        loss = loss + removal_loss_weight * removal_loss
    if relation_loss is not None and relation_loss_weight > 0.0:
        loss = loss + relation_loss_weight * relation_loss
    if direction_loss is not None and changed_direction_loss_weight > 0.0:
        loss = loss + changed_direction_loss_weight * direction_loss
    if temporal_aux_loss is not None and temporal_aux_loss_weight > 0.0:
        loss = loss + temporal_aux_loss_weight * temporal_aux_loss
    if existence_loss is not None and existence_loss_weight > 0.0:
        loss = loss + existence_loss_weight * existence_loss
    if semantic_loss is not None and semantic_loss_weight > 0.0:
        loss = loss + semantic_loss_weight * semantic_loss
    return (
        loss,
        logits,
        target_flat,
        tri_loss.detach(),
        binary_loss,
        removal_loss,
        relation_loss,
        direction_loss,
        temporal_aux_loss,
        existence_loss,
        semantic_loss,
    )


def run_epoch(
    loader,
    model,
    criterion,
    device,
    optimizer=None,
    desc="epoch",
    binary_loss_weight=0.0,
    removal_loss_weight=0.0,
    relation_loss_weight=0.0,
    relation_loss_margin=1.0,
    changed_direction_loss_weight=0.0,
    temporal_aux_loss_weight=0.0,
    existence_loss_weight=0.0,
    semantic_loss_weight=0.0,
    semantic_class_weights=None,
    use_swapped_supervision=True,
    changed_conf_threshold=0.0,
    batch_scheduler=None,
    amp_enabled=False,
    grad_scaler=None,
):
    is_train = optimizer is not None
    model.train(mode=is_train)
    confusion = ConfusionMatrix(NUM_CLASS)
    temporal_confusion = ConfusionMatrix(TEMPORAL_NUM_CLASS)
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    label_counter = torch.zeros(NUM_CLASS)
    temporal_label_counter = torch.zeros(TEMPORAL_NUM_CLASS)
    batch_times = []

    progress = tqdm(loader, desc=desc, leave=False, file=ORIGINAL_STDERR)
    for batch in progress:
        batch_start = time.perf_counter()
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        direction_batches = [(batch, 2)]
        if use_swapped_supervision:
            direction_batches.append((swap_branches(batch), 0))
        batch_loss_sum = 0.0
        batch_tri_loss_sum = 0.0
        batch_binary_loss_sum = 0.0
        batch_removal_loss_sum = 0.0
        batch_relation_loss_sum = 0.0
        batch_direction_loss_sum = 0.0
        batch_temporal_aux_loss_sum = 0.0
        batch_existence_loss_sum = 0.0
        batch_semantic_loss_sum = 0.0
        batch_points = 0

        for direction_batch, unchanged_class in direction_batches:
            with torch.set_grad_enabled(is_train):
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    (
                        loss,
                        logits,
                        target,
                        tri_loss,
                        binary_loss,
                        removal_loss,
                        relation_loss,
                        direction_loss,
                        temporal_aux_loss,
                        existence_loss,
                        semantic_loss,
                    ) = forward_one_direction(
                        model,
                        direction_batch,
                        criterion,
                        binary_loss_weight=binary_loss_weight,
                        removal_loss_weight=removal_loss_weight,
                        relation_loss_weight=relation_loss_weight,
                        relation_loss_margin=relation_loss_margin,
                        changed_direction_loss_weight=changed_direction_loss_weight,
                        temporal_aux_loss_weight=temporal_aux_loss_weight,
                        existence_loss_weight=existence_loss_weight,
                        semantic_loss_weight=semantic_loss_weight,
                        semantic_class_weights=semantic_class_weights,
                        unchanged_class=unchanged_class,
                    )
                if is_train:
                    scaled_loss = loss / len(direction_batches)
                    if grad_scaler is not None:
                        grad_scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

            valid = target != IGNORE_INDEX
            with torch.no_grad():
                valid_target = target[valid]
                if valid_target.numel() == 0:
                    continue
                if (
                    changed_conf_threshold > 0.0
                    and not is_train
                    and TARGET_MODE in ("pair_3class", "anchored_pair_3class")
                ):
                    prob = torch.softmax(logits, dim=-1)
                    pred_all = prob.argmax(dim=-1)
                    changed_conf = prob[:, 1] + prob[:, 2]
                    changed_pred = pred_all != 0
                    pred_all = pred_all.clone()
                    pred_all[changed_pred & (changed_conf < changed_conf_threshold)] = 0
                    pred = pred_all[valid]
                else:
                    pred = logits.argmax(dim=-1)[valid]
                metric_target = valid_target
                metric_pred = pred
                if TARGET_MODE == "pair_3class":
                    temporal_target = build_temporal_targets(
                        direction_batch,
                        unchanged_class=unchanged_class,
                    ).reshape(-1)
                    temporal_valid = temporal_target != IGNORE_INDEX
                else:
                    temporal_target = torch.full_like(target, IGNORE_INDEX)
                    temporal_valid = temporal_target != IGNORE_INDEX
                num_valid = valid_target.numel()
                total_loss += float(loss.detach().cpu()) * num_valid
                batch_loss_sum += float(loss.detach().cpu()) * num_valid
                batch_tri_loss_sum += float(tri_loss.detach().cpu()) * num_valid
                if binary_loss is not None:
                    batch_binary_loss_sum += float(binary_loss.detach().cpu()) * num_valid
                if removal_loss is not None:
                    batch_removal_loss_sum += float(removal_loss.detach().cpu()) * num_valid
                if relation_loss is not None:
                    batch_relation_loss_sum += float(relation_loss.detach().cpu()) * num_valid
                if direction_loss is not None:
                    batch_direction_loss_sum += float(direction_loss.detach().cpu()) * num_valid
                if temporal_aux_loss is not None:
                    batch_temporal_aux_loss_sum += float(temporal_aux_loss.detach().cpu()) * num_valid
                if existence_loss is not None:
                    batch_existence_loss_sum += float(existence_loss.detach().cpu()) * num_valid
                if semantic_loss is not None:
                    batch_semantic_loss_sum += float(semantic_loss.detach().cpu()) * num_valid
                batch_points += num_valid
                total_points += num_valid
                total_correct += metric_pred.eq(metric_target).sum().item()
                confusion.increment_from_list(
                    metric_target.detach().cpu().tolist(),
                    metric_pred.detach().cpu().tolist(),
                )
                label_counter += torch.bincount(
                    valid_target.detach().cpu(), minlength=NUM_CLASS
                )
                temporal_label_counter += torch.bincount(
                    temporal_target[temporal_valid].detach().cpu(),
                    minlength=TEMPORAL_NUM_CLASS,
                )

        if is_train:
            if grad_scaler is not None:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            if batch_scheduler is not None:
                batch_scheduler.step()

        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time)
        postfix = {
            "loss": f"{batch_loss_sum / max(batch_points, 1):.4f}",
            "tri": f"{batch_tri_loss_sum / max(batch_points, 1):.4f}",
            "acc": f"{total_correct / max(total_points, 1):.4f}",
            "pts": total_points,
            "batch_s": f"{batch_time:.2f}",
        }
        if binary_loss_weight > 0.0 and batch_points > 0:
            postfix["bin"] = f"{batch_binary_loss_sum / batch_points:.4f}"
        if removal_loss_weight > 0.0 and batch_points > 0:
            postfix["rem"] = f"{batch_removal_loss_sum / batch_points:.4f}"
        if relation_loss_weight > 0.0 and batch_points > 0:
            postfix["rel"] = f"{batch_relation_loss_sum / batch_points:.4f}"
        if changed_direction_loss_weight > 0.0 and batch_points > 0:
            postfix["dir"] = f"{batch_direction_loss_sum / batch_points:.4f}"
        if temporal_aux_loss_weight > 0.0 and batch_points > 0:
            postfix["aux4"] = f"{batch_temporal_aux_loss_sum / batch_points:.4f}"
        if existence_loss_weight > 0.0 and batch_points > 0:
            postfix["exist"] = f"{batch_existence_loss_sum / batch_points:.4f}"
        if semantic_loss_weight > 0.0 and batch_points > 0:
            postfix["sem"] = f"{batch_semantic_loss_sum / batch_points:.4f}"
        progress.set_postfix(**postfix)

    avg_loss = total_loss / max(total_points, 1)
    avg_acc = total_correct / max(total_points, 1)
    print("Epoch label distribution:", label_counter.tolist())
    if temporal_aux_loss_weight > 0.0:
        print("Epoch temporal aux label distribution:", temporal_label_counter.tolist())
    if batch_times:
        print(
            f"{desc} timing: mean {np.mean(batch_times):.2f}s/batch, "
            f"max {np.max(batch_times):.2f}s/batch, batches={len(batch_times)}"
        )
    return avg_loss, avg_acc, confusion, temporal_confusion


def print_epoch_metrics(epoch, train_loss, train_acc, val_loss, val_acc, val_conf, elapsed):
    per_class_iou = np.asarray(val_conf.get_per_class_ious(), dtype=np.float64)
    per_class_acc = np.asarray(val_conf.get_per_class_accuracy(), dtype=np.float64)
    miou = float(np.nan_to_num(per_class_iou.mean(), nan=0.0))
    print(f"Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, internal acc: {train_acc:.4f}")
    print(
        f"Epoch: {epoch:03d}, Val loss: {val_loss:.4f}, "
        f"internal acc: {val_acc:.4f}, semantic acc: {val_acc:.4f}, "
        f"mIoU: {miou:.4f}, internal per_class_iou: {per_class_iou}, "
        f"internal per_class_acc: {per_class_acc}, "
        f"semantic per_class_iou: {per_class_iou}, semantic per_class_acc: {per_class_acc}"
    )
    print(f"Epoch: {epoch:03d}, total_time: {elapsed:.1f}s")
    return miou, per_class_iou


def train(args):
    device = torch.device("cuda")
    focus_classes = parse_class_ids(args.sample_focus_classes)
    train_dataset = make_dataset(args, "train", args.train_cache_path)

    semantic_class_weights = None
    semantic_weight_counts = None
    if args.semantic_class_weighting == "misnet":
        semantic_class_weights, semantic_weight_counts = compute_misnet_semantic_weights(
            train_dataset.samples,
            num_semantic_classes=int(getattr(args, "semantic_num_classes", 4)),
        )
        semantic_class_weights = semantic_class_weights.to(device)

    raw_counts = compute_class_counts(train_dataset.samples)
    if args.class_weight_source == "sampled":
        semantic_class_counts = compute_sampled_class_counts(train_dataset)
    else:
        semantic_class_counts = raw_counts
    sampler_class_weights = compute_class_weights(
        semantic_class_counts,
        min_weight=args.min_class_weight,
        max_weight=args.max_class_weight,
    )

    loss_class_weights = compute_class_weights(
        semantic_class_counts,
        min_weight=args.min_class_weight,
        max_weight=args.max_class_weight,
    ).to(device)
    class_weight_boosts = parse_class_weight_boosts(args.class_weight_boosts)
    for class_id, boost in class_weight_boosts.items():
        if class_id >= loss_class_weights.numel():
            raise ValueError(
                f"class_weight_boosts class id {class_id} is out of range for {NUM_CLASS} classes."
            )
        loss_class_weights[class_id] *= boost

    pipeline_by_arch = {
        "separate": "PointMamba -> bidirectional local soft matching -> direction-aware change fusion -> token-to-point projection -> 3-class prediction",
        "no_direction": "PointMamba -> bidirectional local soft matching -> shared no-direction change fusion -> token-to-point projection -> 3-class prediction",
        "ptv3_siamese_concat": "Shared Siamese PTv3 -> temporal point-feature concatenation -> dataset-specific prediction head",
        "ptv3_matching_evidence": "Shared Siamese PTv3 -> bidirectional XYZ local soft matching evidence -> dataset-specific prediction head",
        "ptv3_match": "PTv3 -> bidirectional point-level local soft matching -> direction-aware fusion -> 3-class prediction",
        "ptv3_no_direction": "PTv3 -> bidirectional point-level local soft matching -> shared no-direction fusion -> 3-class prediction",
        "ptv3_relation_fusion": "PTv3 -> bidirectional point-level local soft matching -> class-agnostic relation decomposition/fusion -> dataset-specific prediction head",
        "ptv3_relation_encoder": "PTv3 -> bidirectional point-level local soft matching -> unified class-agnostic relation encoder + rich point context -> dataset-specific prediction head",
        "ptv3_relation_attention": "PTv3 -> bidirectional local soft correspondence prior -> gated relation-attention feature update -> shared change representation -> dataset-specific prediction head",
        "ptv3_prior_attention_only": "PTv3 -> bidirectional XYZ soft correspondence prior -> gated relation-attention feature update -> direct point prediction without symmetric XYZ relation",
        "ptv3_xy_xyz_relation_attention": "PTv3 -> bidirectional XYZ/XY soft correspondence -> height-aware dual-space relation gate -> dataset-specific prediction head",
        "ptv3_xy_xyz_concat": "PTv3 -> bidirectional XYZ/XY soft correspondence -> XYZ/XY/height fixed concat-MLP fusion without adaptive relation gate -> dataset-specific prediction head",
        "ptv3_xyz_xy_relation_attention": "PTv3 -> XYZ prior-guided relation attention -> complementary XY relation -> adaptive XYZ/XY fusion -> dataset-specific prediction head",
        "ptv3_xyz_xy_height_relation_attention": "PTv3 -> XYZ prior-guided relation attention -> complementary XY relation + signed/unsigned height evidence -> adaptive fusion -> dataset-specific prediction head",
        "ptv3_semantic_relation_attention": "PTv3 -> correspondence-prior relation update -> semantic transition encoding -> correspondence-gated semantic-change fusion -> dataset-specific prediction head",
        "ptv3_semantic_xy_xyz_concat": "PTv3 -> correspondence-prior feature update -> XYZ/XY/height/semantic relation construction -> fixed concatenation -> dataset-specific prediction head",
        "ptv3_semantic_xy_xyz_relation_attention": "PTv3 -> bidirectional XYZ/XY soft correspondence -> dual-space and height prior -> gated semantic-change fusion -> dataset-specific prediction head",
        "ptv3_t1_anchored_semantic_xy_xyz_relation_attention": "independent T0/T1 cylinders -> Siamese PTv3 -> T1-anchored XYZ/XY soft correspondence -> gated semantic-spatial fusion -> T1 prediction",
        "ptv3_t1_anchored_ordered_semantic_transition_attention": "independent T0/T1 cylinders -> Siamese PTv3 -> T1-anchored XYZ/XY correspondence -> semantic-spatial fusion -> zero-initialized T1 ordered semantic-transition residual -> T1 prediction",
        "ptv3_t1_anchored_local_vertical_profile_attention": "independent T0/T1 cylinders -> Siamese PTv3 -> T1-anchored XYZ/XY correspondence -> semantic-spatial fusion -> gated local vertical-profile residual -> T1 prediction",
        "ptv3_semantic_xy_xyz_ordered_transition_attention": "PTv3 -> bidirectional XYZ/XY soft correspondence -> T1-anchored ordered semantic probability transition -> gated semantic-change fusion -> dataset-specific prediction head",
        "ptv3_semantic_xy_xyz_residual_ordered_transition_attention": "PTv3 -> XYZ/XY correspondence-prior relation attention -> baseline semantic-change fusion + zero-initialized gated ordered-transition residual -> dataset-specific prediction head",
        "ptv3_directional_semantic_xy_xyz_relation_attention": "PTv3 -> XYZ/XY soft correspondence -> ordered bidirectional semantic transitions -> dual-space gated change fusion -> dataset-specific prediction head",
        "ptv3_update_refine": "PTv3 -> one bidirectional point-level local soft matching -> gated cross-temporal update -> lightweight refinement -> change representation -> 3-class prediction",
        "ptv3_clean_change": "PTv3 -> bidirectional point-level local soft matching -> compact shared change representation -> 3-class prediction",
        "ptv3_update_ptv3_refine": "PTv3 -> one bidirectional point-level local soft matching -> gated cross-temporal update -> lightweight PTv3 refinement -> shared no-direction fusion -> 3-class prediction",
        "two_stage": "Stage1 PointMamba -> bidirectional local soft matching -> gated token update -> Stage2 token Mamba -> direction-aware fusion -> token-to-point projection -> 3-class prediction",
        "joint": "joint T0/T1 PointMamba -> token-to-point projection -> 3-class prediction",
        "joint_match": "joint T0/T1 PointMamba -> token split -> bidirectional local soft matching -> direction-aware fusion -> token-to-point projection -> 3-class prediction",
    }
    print("Pipeline:", pipeline_by_arch.get(args.model_arch, args.model_arch))
    print("Semantic class labels:", STREET_CLASS_LABELS)
    print("Aux temporal class labels:", TEMPORAL_CLASS_LABELS)
    print("Raw class counts:", raw_counts.tolist())
    print("Semantic class-weight counts:", semantic_class_counts.tolist())
    print(f"{NUM_CLASS}-class training weights:", loss_class_weights.detach().cpu().tolist())
    print("Class weight boosts:", class_weight_boosts)
    print("Semantic class weighting:", args.semantic_class_weighting)
    if semantic_class_weights is not None:
        print("Semantic class counts:", semantic_weight_counts.tolist())
        print(
            "Semantic class weights (1-frequency):",
            semantic_class_weights.detach().cpu().tolist(),
        )
    print("Sampler focus classes:", focus_classes)
    print(
        "Model config:",
        f"model_arch={args.model_arch},",
        f"token_match_k={args.token_match_k},",
        f"ptv3_order={parse_ptv3_order(args.ptv3_order)},",
        f"ptv3_patch_size={args.ptv3_patch_size},",
        f"num_group={args.num_group},",
        f"group_size={args.group_size},",
        f"stage2_depth={args.stage2_depth},",
        f"use_point_decoder={args.use_point_decoder},",
        f"signed_diff=True, swapped_supervision={not args.disable_swapped_supervision},",
        f"temporal_embedding={args.temporal_embedding},",
        f"loss_type={args.loss_type},",
        f"lovasz_weight={args.lovasz_weight},",
        f"class_weight_boosts={args.class_weight_boosts},",
        f"warm_start_checkpoint={args.warm_start_checkpoint},",
        f"freeze_ptv3_encoder={args.freeze_ptv3_encoder},",
        f"temporal_aux_loss_weight={args.temporal_aux_loss_weight},",
        f"binary_loss_weight={args.binary_loss_weight},",
        f"semantic_loss_weight={args.semantic_loss_weight},",
        f"prior_injection_mode={getattr(args, 'prior_injection_mode', 'p5_full')},",
        f"semantic_class_weighting={args.semantic_class_weighting},",
        f"removal_loss_weight={args.removal_loss_weight},",
        f"relation_loss_weight={args.relation_loss_weight},",
        f"relation_loss_margin={args.relation_loss_margin},",
        f"changed_direction_loss_weight={args.changed_direction_loss_weight},",
        f"existence_loss_weight={args.existence_loss_weight}",
    )

    train_loader = make_train_loader(args, train_dataset, sampler_class_weights, focus_classes)
    val_loader = make_eval_loader(args, "val")
    model = build_model(args, device)
    load_warm_start_weights(model, args.warm_start_checkpoint, device)
    freeze_ptv3_encoder_if_requested(model, args.freeze_ptv3_encoder)
    total_parameter_count = sum(param.numel() for param in model.parameters())
    trainable_parameter_count = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(
        "Model parameters:",
        f"total={total_parameter_count},",
        f"trainable={trainable_parameter_count}",
    )
    criterion = make_seg_criterion(args, loss_class_weights, device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after freezing.")
    optimizer_name = getattr(args, "optimizer", "adam").lower()
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=args.lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp) if args.amp else None

    scheduler_name = getattr(args, "scheduler", "step").lower()
    epoch_scheduler = None
    batch_scheduler = None
    if scheduler_name == "step":
        epoch_scheduler = StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.step_gamma,
        )
    elif scheduler_name == "onecycle":
        if len(train_loader) < 1:
            raise RuntimeError("OneCycleLR requires at least one training batch.")
        batch_scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=float(getattr(args, "onecycle_pct_start", 0.1)),
            anneal_strategy="cos",
            div_factor=float(getattr(args, "onecycle_div_factor", 10.0)),
            final_div_factor=float(
                getattr(args, "onecycle_final_div_factor", 1000.0)
            ),
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    print(
        "Optimization config:",
        f"optimizer={optimizer_name},",
        f"scheduler={scheduler_name},",
        f"max_lr={args.lr},",
        f"initial_lr={optimizer.param_groups[0]['lr']:.8g},",
        f"weight_decay={weight_decay},",
        f"steps_per_epoch={len(train_loader)}",
    )

    best_miou = -1.0
    best_epoch = 0
    best_iou = None
    checkpoint_root, _ = osp.splitext(args.checkpoint)
    top_k_checkpoints = max(int(getattr(args, "top_k_checkpoints", 1)), 1)
    top_candidates = []
    early_stopping_patience = max(
        int(getattr(args, "early_stopping_patience", 0)),
        0,
    )
    early_stopping_min_delta = max(
        float(getattr(args, "early_stopping_min_delta", 0.0)),
        0.0,
    )
    epochs_without_improvement = 0
    metrics_path = f"{checkpoint_root}.metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "val_miou",
                *[f"{name}_iou" for name in STREET_CLASS_LABELS],
                "lr",
                "epoch_time",
            ]
        )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        print("epoch is:", epoch)
        train_loss, train_acc, _, _ = run_epoch(
            train_loader,
            model,
            criterion,
            device,
            optimizer=optimizer,
            desc=f"train {epoch:03d}",
            binary_loss_weight=args.binary_loss_weight,
            removal_loss_weight=args.removal_loss_weight,
            relation_loss_weight=args.relation_loss_weight,
            relation_loss_margin=args.relation_loss_margin,
            changed_direction_loss_weight=args.changed_direction_loss_weight,
            temporal_aux_loss_weight=args.temporal_aux_loss_weight,
            existence_loss_weight=args.existence_loss_weight,
            semantic_loss_weight=args.semantic_loss_weight,
            semantic_class_weights=semantic_class_weights,
            use_swapped_supervision=not args.disable_swapped_supervision,
            batch_scheduler=batch_scheduler,
            amp_enabled=args.amp,
            grad_scaler=grad_scaler,
        )
        val_loss, val_acc, val_conf, val_internal_conf = run_epoch(
            val_loader,
            model,
            criterion,
            device,
            desc=f"val {epoch:03d}",
            binary_loss_weight=0.0,
            removal_loss_weight=0.0,
            relation_loss_weight=0.0,
            relation_loss_margin=args.relation_loss_margin,
            changed_direction_loss_weight=0.0,
            temporal_aux_loss_weight=0.0,
            existence_loss_weight=0.0,
            semantic_loss_weight=0.0,
            use_swapped_supervision=not args.disable_swapped_supervision,
            amp_enabled=args.amp,
        )
        if epoch_scheduler is not None:
            epoch_scheduler.step()
        epoch_time = time.perf_counter() - epoch_start
        miou, per_class_iou = print_epoch_metrics(
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_conf,
            epoch_time,
        )
        with open(metrics_path, "a", newline="", encoding="utf-8") as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.8f}",
                    f"{train_acc:.8f}",
                    f"{val_loss:.8f}",
                    f"{val_acc:.8f}",
                    f"{miou:.8f}",
                    *[f"{value:.8f}" for value in per_class_iou],
                    f"{optimizer.param_groups[0]['lr']:.8g}",
                    f"{epoch_time:.4f}",
                ]
            )
        print(f"Semantic {NUM_CLASS}-class metrics:")
        val_conf.print_metrics(labels=STREET_CLASS_LABELS)
        print(f"Semantic {NUM_CLASS}-class error summary:")
        summarize_confusion(val_conf, STREET_CLASS_LABELS)

        if top_k_checkpoints > 1:
            candidate_qualifies = (
                len(top_candidates) < top_k_checkpoints
                or miou > top_candidates[-1]["miou"]
            )
            if candidate_qualifies:
                candidate_path = f"{checkpoint_root}.candidate_epoch{epoch:03d}.pth"
                torch.save(model.state_dict(), candidate_path)
                top_candidates.append(
                    {
                        "epoch": epoch,
                        "miou": miou,
                        "iou": per_class_iou.tolist(),
                        "path": candidate_path,
                    }
                )
                top_candidates.sort(key=lambda item: item["miou"], reverse=True)
                while len(top_candidates) > top_k_checkpoints:
                    removed = top_candidates.pop()
                    if osp.exists(removed["path"]):
                        os.remove(removed["path"])

        improved = miou > best_miou + early_stopping_min_delta
        if improved:
            best_miou = miou
            best_epoch = epoch
            best_iou = per_class_iou.tolist()
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.checkpoint)
            with open(f"{checkpoint_root}_internal_conf.pickle", "wb") as handle:
                pickle.dump(val_conf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch:03d}: validation mIoU did not "
                f"improve by more than {early_stopping_min_delta:.6f} for "
                f"{early_stopping_patience} consecutive epochs."
            )
            break

    print(
        f"Small-panel best epoch: {best_epoch:03d}, best mIoU: {best_miou:.4f}, "
        f"semantic per class IoU: {best_iou}"
    )

    selection_val_samples = int(getattr(args, "selection_val_samples", 0))
    if selection_val_samples > 0 and top_candidates:
        selection_args = copy.copy(args)
        selection_args.val_samples_per_epoch = selection_val_samples
        selection_args.fixed_val_sampling = True
        selection_args.stratified_val_sampling = True
        selection_args.val_sampling_seed = int(
            getattr(args, "selection_val_seed", args.seed + 2000)
        )
        print(
            f"Selecting among {len(top_candidates)} candidates on "
            f"{selection_val_samples} fixed class-stratified validation cylinders."
        )
        selection_loader = make_eval_loader(selection_args, "val")
        selection_rows = []
        selected_conf = None
        for candidate in top_candidates:
            model.load_state_dict(torch.load(candidate["path"], map_location=device))
            candidate_loss, candidate_acc, candidate_conf, _ = run_epoch(
                selection_loader,
                model,
                criterion,
                device,
                desc=f"select epoch {candidate['epoch']:03d}",
                relation_loss_margin=args.relation_loss_margin,
                use_swapped_supervision=not args.disable_swapped_supervision,
                amp_enabled=args.amp,
            )
            candidate_iou = np.asarray(
                candidate_conf.get_per_class_ious(), dtype=np.float64
            )
            candidate_miou = float(np.nan_to_num(candidate_iou.mean(), nan=0.0))
            selection_rows.append(
                {
                    **candidate,
                    "selection_loss": candidate_loss,
                    "selection_acc": candidate_acc,
                    "selection_miou": candidate_miou,
                    "selection_iou": candidate_iou.tolist(),
                    "confusion": candidate_conf,
                }
            )
            print(
                f"Selection candidate epoch {candidate['epoch']:03d}: "
                f"small-panel mIoU={candidate['miou']:.4f}, "
                f"selection mIoU={candidate_miou:.4f}, IoU={candidate_iou}"
            )

        selected = max(selection_rows, key=lambda item: item["selection_miou"])
        model.load_state_dict(torch.load(selected["path"], map_location=device))
        torch.save(model.state_dict(), args.checkpoint)
        selected_conf = selected["confusion"]
        with open(f"{checkpoint_root}_internal_conf.pickle", "wb") as handle:
            pickle.dump(selected_conf, handle, protocol=pickle.HIGHEST_PROTOCOL)

        selection_metrics_path = f"{checkpoint_root}.selection.csv"
        with open(selection_metrics_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "epoch",
                    "small_panel_miou",
                    "selection_loss",
                    "selection_acc",
                    "selection_miou",
                    *[f"{name}_iou" for name in STREET_CLASS_LABELS],
                    "candidate_path",
                ]
            )
            for row in sorted(selection_rows, key=lambda item: item["epoch"]):
                writer.writerow(
                    [
                        row["epoch"],
                        f"{row['miou']:.8f}",
                        f"{row['selection_loss']:.8f}",
                        f"{row['selection_acc']:.8f}",
                        f"{row['selection_miou']:.8f}",
                        *[f"{value:.8f}" for value in row["selection_iou"]],
                        row["path"],
                    ]
                )
        print(
            f"Final selected epoch: {selected['epoch']:03d}, "
            f"selection mIoU: {selected['selection_miou']:.4f}, "
            f"semantic per class IoU: {selected['selection_iou']}"
        )
        print(f"Final checkpoint: {args.checkpoint}")
        print(f"Selection metrics: {selection_metrics_path}")
    else:
        print(
            f"Best epoch: {best_epoch:03d}, best mIoU: {best_miou:.4f}, "
            f"semantic per class IoU: {best_iou}"
        )


def evaluate(args):
    device = torch.device("cuda")
    model = build_model(args, device)
    if not osp.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    criterion = make_seg_criterion(args, torch.ones(NUM_CLASS, device=device), device)
    if getattr(args, "class_bias_sweep", ""):
        calibrate_class_logit_bias(args, model, criterion, device)
        return
    if args.threshold_sweep:
        thresholds = [
            float(value.strip())
            for value in args.threshold_sweep.split(",")
            if value.strip()
        ]
    else:
        thresholds = [args.changed_conf_threshold]

    for threshold_idx, threshold in enumerate(thresholds):
        if len(thresholds) > 1:
            print(f"\n========== Threshold {threshold:.3f} ({threshold_idx + 1}/{len(thresholds)}) ==========")
        evaluate_with_threshold(args, model, criterion, device, threshold)


def parse_float_sweep(spec):
    """Parse either comma-separated values or an inclusive start:end:step range."""
    spec = spec.strip()
    if not spec:
        return []
    if ":" not in spec:
        return [float(value.strip()) for value in spec.split(",") if value.strip()]

    parts = [float(value.strip()) for value in spec.split(":")]
    if len(parts) != 3:
        raise ValueError(
            "A range sweep must use start:end:step, for example -0.6:0.6:0.05"
        )
    start, end, step = parts
    if step == 0.0 or (end - start) * step < 0.0:
        raise ValueError(f"Invalid sweep range: {spec}")

    count = int(np.floor((end - start) / step + 1e-9)) + 1
    values = [start + idx * step for idx in range(count)]
    values = [0.0 if abs(value) <= abs(step) * 1e-9 else value for value in values]
    if values and abs(values[-1] - end) <= abs(step) * 1e-6:
        values[-1] = end
    return values


def evaluate_class_logit_biases(args, model, criterion, device, split, biases, class_id):
    if not 0 <= class_id < NUM_CLASS:
        raise ValueError(f"class_bias_id must be in [0, {NUM_CLASS - 1}], got {class_id}")
    if not biases:
        raise ValueError("At least one class-logit bias is required")

    loader = make_eval_loader(args, split)
    confusion_matrices = np.zeros(
        (len(biases), NUM_CLASS, NUM_CLASS), dtype=np.int64
    )
    bias_tensor = torch.as_tensor(biases, device=device).reshape(-1, 1)
    bias_offsets = (
        torch.arange(len(biases), device=device).reshape(-1, 1)
        * NUM_CLASS
        * NUM_CLASS
    )
    total_loss = 0.0
    total_points = 0
    model.eval()
    progress = tqdm(loader, desc=f"{split} class-bias evaluation", leave=False, file=ORIGINAL_STDERR)

    with torch.no_grad():
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            model_output = model(batch)
            logits = model_output["point_logits"] if isinstance(model_output, dict) else model_output
            logits = flatten_logits(logits)
            target = flatten_targets(build_main_targets(batch))
            valid = target != IGNORE_INDEX
            if not torch.any(valid):
                continue

            loss = criterion(logits, target)
            valid_logits = logits[valid]
            valid_target = target[valid]
            num_valid = valid_target.numel()
            total_loss += float(loss.detach().cpu()) * num_valid
            total_points += num_valid

            other_logits = valid_logits.clone()
            other_logits[:, class_id] = -torch.inf
            other_score, other_pred = other_logits.max(dim=-1)
            class_score = valid_logits[:, class_id]

            biased_class_score = class_score.reshape(1, -1) + bias_tensor
            choose_class = biased_class_score > other_score.reshape(1, -1)
            choose_class |= (
                (biased_class_score == other_score.reshape(1, -1))
                & (class_id < other_pred.reshape(1, -1))
            )
            pred = torch.where(
                choose_class,
                torch.full_like(choose_class, class_id, dtype=other_pred.dtype),
                other_pred.reshape(1, -1),
            )
            encoded = (
                bias_offsets
                + valid_target.reshape(1, -1) * NUM_CLASS
                + pred
            )
            counts = torch.bincount(
                encoded.reshape(-1),
                minlength=len(biases) * NUM_CLASS * NUM_CLASS,
            )
            confusion_matrices += (
                counts.reshape(len(biases), NUM_CLASS, NUM_CLASS).cpu().numpy()
            )

    results = []
    for bias, matrix in zip(biases, confusion_matrices):
        conf = ConfusionMatrix(NUM_CLASS)
        conf.confusion_matrix += matrix
        ious = np.asarray(conf.get_per_class_ious(), dtype=np.float64)
        results.append(
            {
                "bias": float(bias),
                "miou": float(np.nan_to_num(ious.mean(), nan=0.0)),
                "accuracy": float(conf.get_accuracy()),
                "ious": ious,
                "confusion": conf,
            }
        )

    return total_loss / max(total_points, 1), results


def write_class_bias_results(args, results, class_id):
    output_stem = osp.splitext(args.log_path or args.checkpoint)[0]
    csv_path = output_stem + ".class_bias_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["class_id", "class_name", "bias", "miou", "accuracy"]
            + [f"{name}_iou" for name in STREET_CLASS_LABELS]
        )
        for result in results:
            writer.writerow(
                [
                    class_id,
                    STREET_CLASS_LABELS[class_id],
                    result["bias"],
                    result["miou"],
                    result["accuracy"],
                    *result["ious"].tolist(),
                ]
            )
    print(f"Saved class-bias sweep: {csv_path}")


def calibrate_class_logit_bias(args, model, criterion, device):
    class_id = int(args.class_bias_id)
    biases = parse_float_sweep(args.class_bias_sweep)
    print(
        f"Calibrating logit bias for class {class_id} "
        f"({STREET_CLASS_LABELS[class_id]}) on validation only."
    )
    val_loss, val_results = evaluate_class_logit_biases(
        args,
        model,
        criterion,
        device,
        split="val",
        biases=biases,
        class_id=class_id,
    )
    print(f"Validation loss before logit calibration: {val_loss:.4f}")
    print("bias      val_mIoU   val_OA     class_IoU")
    for result in val_results:
        print(
            f"{result['bias']:+.4f}   {result['miou']:.6f}   "
            f"{result['accuracy']:.6f}   {result['ious'][class_id]:.6f}"
        )

    best = max(
        val_results,
        key=lambda result: (result["miou"], -abs(result["bias"])),
    )
    print(
        f"Selected validation bias: {best['bias']:+.4f}; "
        f"mIoU={best['miou']:.6f}, OA={best['accuracy']:.6f}, "
        f"{STREET_CLASS_LABELS[class_id]} IoU={best['ious'][class_id]:.6f}"
    )
    write_class_bias_results(args, val_results, class_id)

    if not getattr(args, "class_bias_test_after_sweep", False):
        return

    print("\nApplying the validation-selected bias to the test split once.")
    test_loss, test_results = evaluate_class_logit_biases(
        args,
        model,
        criterion,
        device,
        split="test",
        biases=[best["bias"]],
        class_id=class_id,
    )
    test_result = test_results[0]
    print(f"Selected class-logit bias: {best['bias']:+.4f}")
    print(f"test loss: {test_loss:.4f}, semantic acc: {test_result['accuracy']:.4f}")
    print(f"Semantic {NUM_CLASS}-class metrics:")
    test_result["confusion"].print_metrics(labels=STREET_CLASS_LABELS)
    print(f"Semantic {NUM_CLASS}-class error summary:")
    summarize_confusion(test_result["confusion"], STREET_CLASS_LABELS)


def evaluate_with_threshold(args, model, criterion, device, threshold):
    total_loss = 0.0
    total_acc = 0.0
    conf = ConfusionMatrix(NUM_CLASS)
    votes = max(args.test_votes, 1)
    for vote_idx in range(votes):
        loader = make_eval_loader(args, args.eval_split)
        vote_loss, vote_acc, vote_conf, _ = run_epoch(
            loader,
            model,
            criterion,
            device,
            desc=f"{args.eval_split} vote {vote_idx + 1:02d}/{votes:02d}",
            temporal_aux_loss_weight=0.0,
            existence_loss_weight=0.0,
            use_swapped_supervision=not args.disable_swapped_supervision,
            changed_conf_threshold=threshold,
            amp_enabled=args.amp,
        )
        total_loss += vote_loss
        total_acc += vote_acc
        conf.confusion_matrix += vote_conf.confusion_matrix
        if votes > 1:
            vote_iou = np.asarray(vote_conf.get_per_class_ious(), dtype=np.float64)
            print(
                f"Vote {vote_idx + 1:02d}/{votes:02d}: "
                f"loss={vote_loss:.4f}, acc={vote_acc:.4f}, "
                f"mIoU={float(np.nan_to_num(vote_iou.mean(), nan=0.0)):.4f}, "
                f"IoU={vote_iou}"
            )
    loss = total_loss / votes
    acc = total_acc / votes
    if votes > 1:
        print(f"Aggregated test-time voting over {votes} votes.")
    if threshold > 0.0:
        print(f"Changed confidence threshold: {threshold:.3f}")
    print(f"{args.eval_split} loss: {loss:.4f}, semantic acc: {acc:.4f}")
    print(f"Semantic {NUM_CLASS}-class metrics:")
    conf.print_metrics(labels=STREET_CLASS_LABELS)
    print(f"Semantic {NUM_CLASS}-class error summary:")
    summarize_confusion(conf, STREET_CLASS_LABELS)


def get_args():
    parser = argparse.ArgumentParser(
        description="Clean training script for simple PointMamba direction-aware fusion."
    )
    parser.add_argument("--dataset", choices=("slpccd", "urb3dcd_v2", "nyc", "hkscd"), default="slpccd")
    parser.add_argument(
        "--slpccd_sample_mode",
        choices=(
            "street",
            "natural_bidirectional_pair",
            "spatial_coverage_bidirectional_pair",
            "fps_bidirectional_pair",
            "mecpt_full_pair",
        ),
        default="street",
    )
    parser.add_argument(
        "--slpccd_val_sample_mode",
        choices=(
            "street",
            "natural_bidirectional_pair",
            "spatial_coverage_bidirectional_pair",
            "fps_bidirectional_pair",
            "mecpt_full_pair",
        ),
        default=None,
        help=(
            "Optional validation-only sampler. Use a natural or spatial-coverage "
            "mode to validate without label-balanced point sampling."
        ),
    )
    parser.add_argument(
        "--slpccd_test_sample_mode",
        choices=(
            "street",
            "natural_bidirectional_pair",
            "spatial_coverage_bidirectional_pair",
            "fps_bidirectional_pair",
            "mecpt_full_pair",
        ),
        default=None,
        help="Optional test-only SLPCCD sampler.",
    )
    parser.add_argument("--data_root", type=str, default="data/SLPCCD/SHREC2020-CD")
    parser.add_argument(
        "--slpccd_voxel_size",
        type=float,
        default=0.0,
        help="Voxel size applied before natural SLPCCD pair sampling; ME-CPT uses 0.03 m.",
    )
    parser.add_argument(
        "--slpccd_coverage_voxel_size",
        type=float,
        default=0.1,
        help=(
            "Voxel size used to distribute coverage anchors or construct the "
            "hierarchical FPS candidate set. It does not voxelize the final input."
        ),
    )
    parser.add_argument("--train_cache_path", type=str, default="train.pt")
    parser.add_argument("--val_cache_path", type=str, default="val.pt")
    parser.add_argument("--test_cache_path", type=str, default="test.pt")
    parser.add_argument("--nyc_removed_labels", type=str, default="1")
    parser.add_argument("--nyc_added_labels", type=str, default="2,3")
    parser.add_argument("--nyc_ignore_labels", type=str, default="4,5,6")
    parser.add_argument("--nyc_all_labels", action="store_true")
    parser.add_argument("--urb3dcd_v2_all_labels", action="store_true")
    parser.add_argument("--urb3dcd_removed_labels", type=str, default="1")
    parser.add_argument("--urb3dcd_added_labels", type=str, default="2,3")
    parser.add_argument("--urb3dcd_ignore_labels", type=str, default="4,5,6")
    parser.add_argument("--urb3dcd_sensor", type=str, default="1-Lidar05")
    parser.add_argument("--urb3dcd_sample_mode", choices=("label1_multiclass", "label1_multiclass_cylinder", "label1_multiclass_independent_cylinder", "label1_multiclass_independent_uniform_cylinder", "label1_multiclass_natural_cylinder", "label1_multiclass_variable_cylinder", "label1_multiclass_mecpt_grid_cylinder", "label1_multiclass_mecpt_grid_paired_cylinder", "mecpt_full_scene", "mecpt_full_scene_tiled"), default="label1_multiclass_cylinder")
    parser.add_argument("--mecpt_tile_size", type=float, default=200.0)
    parser.add_argument("--mecpt_tile_halo", type=float, default=20.0)
    parser.add_argument("--urb3dcd_radius", type=float, default=50.0)
    parser.add_argument("--urb3dcd_voxel_size", type=float, default=0.3)
    parser.add_argument("--hkscd_sample_mode", choices=("street", "pair_3class_cylinder"), default="pair_3class_cylinder")
    parser.add_argument("--hkscd_radius", type=float, default=50.0)
    parser.add_argument("--hkscd_voxel_size", type=float, default=0.3)
    parser.add_argument("--train_samples_per_epoch", type=int, default=None)
    parser.add_argument(
        "--train_sampling_strategy",
        choices=("weighted", "uniform"),
        default="weighted",
    )
    parser.add_argument("--val_samples_per_epoch", type=int, default=None)
    parser.add_argument("--test_samples_per_epoch", type=int, default=None)
    parser.add_argument("--fixed_val_sampling", action="store_true")
    parser.add_argument("--val_sampling_seed", type=int, default=1042)
    parser.add_argument("--fixed_eval_sampling", action="store_true")
    parser.add_argument("--eval_sampling_seed", type=int, default=3042)
    parser.add_argument("--fixed_eval_cache_path", default=None)
    parser.add_argument("--stratified_val_sampling", action="store_true")
    parser.add_argument("--top_k_checkpoints", type=int, default=1)
    parser.add_argument("--selection_val_samples", type=int, default=0)
    parser.add_argument("--selection_val_seed", type=int, default=2042)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=str, default="lovasz_time_aux4_50e.pth")
    parser.add_argument("--warm_start_checkpoint", type=str, default=None)
    parser.add_argument("--freeze_ptv3_encoder", action="store_true")
    parser.add_argument("--model_arch", choices=("separate", "no_direction", "ptv3_siamese_concat", "ptv3_matching_evidence", "ptv3_match", "ptv3_no_direction", "ptv3_relation_fusion", "ptv3_relation_encoder", "ptv3_relation_attention", "ptv3_prior_attention_only", "ptv3_xy_xyz_relation_attention", "ptv3_xy_xyz_concat", "ptv3_xyz_xy_relation_attention", "ptv3_xyz_xy_height_relation_attention", "ptv3_semantic_relation_attention", "ptv3_semantic_xy_xyz_concat", "ptv3_semantic_xy_xyz_relation_attention", "ptv3_t1_anchored_semantic_xy_xyz_relation_attention", "ptv3_t1_anchored_ordered_semantic_transition_attention", "ptv3_t1_anchored_local_vertical_profile_attention", "ptv3_semantic_xy_xyz_ordered_transition_attention", "ptv3_semantic_xy_xyz_residual_ordered_transition_attention", "ptv3_directional_semantic_xy_xyz_relation_attention", "ptv3_update_refine", "ptv3_clean_change", "ptv3_update_ptv3_refine", "two_stage", "joint", "joint_match"), default="separate")
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--disable_log_file", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision to reduce training memory.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--num_group", type=int, default=128)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--stage2_depth", type=int, default=2)
    parser.add_argument("--use_point_decoder", action="store_true")
    parser.add_argument("--disable_swapped_supervision", action="store_true")
    parser.add_argument("--change_ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("step", "onecycle"), default="step")
    parser.add_argument("--step_size", type=int, default=15)
    parser.add_argument("--step_gamma", type=float, default=0.1)
    parser.add_argument("--onecycle_pct_start", type=float, default=0.1)
    parser.add_argument("--onecycle_div_factor", type=float, default=10.0)
    parser.add_argument(
        "--onecycle_final_div_factor",
        type=float,
        default=1000.0,
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--loss_type", choices=("focal_ce", "ce", "weighted_ce"), default="focal_ce")
    parser.add_argument("--lovasz_weight", type=float, default=1.0)
    parser.add_argument("--token_match_k", type=int, default=16)
    parser.add_argument("--ptv3_order", type=str, default="z,z-trans,hilbert,hilbert-trans")
    parser.add_argument("--ptv3_patch_size", type=int, default=512)
    parser.add_argument("--token_match_tau_feat", type=float, default=0.1)
    parser.add_argument("--token_match_tau_geo", type=float, default=1.0)
    parser.add_argument(
        "--scalable_knn",
        action="store_true",
        help="Use torch_cluster GPU KNN instead of all-pairs cdist for matching candidates.",
    )
    parser.add_argument(
        "--match_query_chunk_size",
        type=int,
        default=4096,
        help="Query chunk size for memory-efficient scalable soft matching.",
    )
    parser.add_argument("--token_match_score_grad", action="store_true")
    temporal_group = parser.add_mutually_exclusive_group()
    temporal_group.add_argument("--temporal_embedding", dest="temporal_embedding", action="store_true")
    temporal_group.add_argument("--no-temporal_embedding", dest="temporal_embedding", action="store_false")
    parser.set_defaults(temporal_embedding=True)
    parser.add_argument("--temporal_aux_loss_weight", type=float, default=0.2)
    parser.add_argument("--binary_loss_weight", type=float, default=0.0)
    parser.add_argument("--semantic_loss_weight", type=float, default=0.0)
    parser.add_argument("--semantic_num_classes", type=int, default=4)
    parser.add_argument(
        "--semantic_class_weighting",
        choices=("none", "misnet"),
        default="none",
    )
    parser.add_argument("--removal_loss_weight", type=float, default=0.0)
    parser.add_argument("--relation_loss_weight", type=float, default=0.0)
    parser.add_argument("--relation_loss_margin", type=float, default=1.0)
    parser.add_argument("--changed_direction_loss_weight", type=float, default=0.0)
    parser.add_argument("--existence_loss_weight", type=float, default=0.0)
    parser.add_argument("--sample_focus_classes", type=str, default="1,2")
    parser.add_argument("--sample_focus_power", type=float, default=0.5)
    parser.add_argument("--class_weight_source", choices=("sampled", "raw"), default="sampled")
    parser.add_argument("--min_class_weight", type=float, default=0.5)
    parser.add_argument("--max_class_weight", type=float, default=2.0)
    parser.add_argument("--class_weight_boosts", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_model", action="store_true")
    parser.add_argument(
        "--correspondence_prior",
        choices=(
            "confidence",
            "distance",
            "confidence_entropy",
            "confidence_distance",
            "confidence_entropy_distance",
            "full",
        ),
        default="full",
        help="Explicit correspondence prior channels; omitted channels are zeroed.",
    )
    parser.add_argument(
        "--prior_injection_mode",
        choices=(
            "p1_no_prior",
            "p2_late_concat",
            "p3_interaction_only",
            "p4_fusion_only",
            "p5_full",
        ),
        default="p5_full",
        help=(
            "Controls where correspondence statistics condition the full "
            "semantic XYZ/XY CPGRL model."
        ),
    )
    parser.add_argument("--eval_split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--test_votes", type=int, default=1)
    parser.add_argument("--changed_conf_threshold", type=float, default=0.0)
    parser.add_argument("--threshold_sweep", type=str, default="")
    parser.add_argument(
        "--class_bias_id",
        type=int,
        default=4,
        help="Class whose prediction logit is shifted during validation calibration.",
    )
    parser.add_argument(
        "--class_bias_sweep",
        type=str,
        default="",
        help="Comma list or inclusive start:end:step validation sweep.",
    )
    parser.add_argument(
        "--class_bias_test_after_sweep",
        action="store_true",
        help="Evaluate test once using the bias selected exclusively on validation.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    configure_task(args)
    setup_log_file(args)
    print(args)
    require_cuda()
    set_seed(args.seed)
    if args.test_model:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
