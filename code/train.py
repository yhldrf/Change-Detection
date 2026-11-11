#!/usr/bin/env python3
"""Dedicated trainer/evaluator for Urb3DCD-V2 7-class change detection."""

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass

from train_mamba_point_clean import (
    configure_task,
    evaluate,
    require_cuda,
    set_seed,
    setup_log_file,
    train,
)


@dataclass
class Urb3DCDV2Config:
    data_root: str = "/home/dell/Downloads/IEEE_Dataset_V2_Lid05_MS"
    sensor: str = "1-Lidar05"
    checkpoint: str = "ptv3_no_direction_urb3dcd_v2_1-Lidar05_all_labels_4096_lovasz_100e.pth"
    warm_start_checkpoint: str = None
    log_path: str = None
    test_model: bool = False
    eval_split: str = "val"
    test_votes: int = 1
    changed_conf_threshold: float = 0.0
    class_bias_id: int = 4
    class_bias_sweep: str = ""
    class_bias_test_after_sweep: bool = False

    epochs: int = 100
    batch_size: int = 1
    amp: bool = False
    num_workers: int = 4
    num_points: int = 4096
    train_samples_per_epoch: int = 6000
    val_samples_per_epoch: int = 1000
    test_samples_per_epoch: int = 1000
    train_sampling_strategy: str = "weighted"
    fixed_val_sampling: bool = False
    val_sampling_seed: int = 1042
    fixed_eval_sampling: bool = False
    eval_sampling_seed: int = 3042
    fixed_eval_cache_path: str = None
    stratified_val_sampling: bool = False
    top_k_checkpoints: int = 1
    selection_val_samples: int = 0
    selection_val_seed: int = 2042
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    model_arch: str = "ptv3_no_direction"
    prior_injection_mode: str = "p5_full"
    token_match_k: int = 8
    token_match_tau_feat: float = 0.1
    token_match_tau_geo: float = 1.0
    scalable_knn: bool = False
    ptv3_order: str = "z,z-trans,hilbert,hilbert-trans"
    ptv3_patch_size: int = 1024
    loss_type: str = "focal_ce"
    lovasz_weight: float = 1.0
    lr: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0
    scheduler: str = "step"
    step_size: int = 15
    step_gamma: float = 0.1
    onecycle_pct_start: float = 0.1
    onecycle_div_factor: float = 10.0
    onecycle_final_div_factor: float = 1000.0
    focal_gamma: float = 2.0

    radius: float = 50.0
    voxel_size: float = 0.3
    change_ratio: float = 0.5
    sample_mode: str = "label1_multiclass_cylinder"
    train_sample_mode: str = None
    val_sample_mode: str = None
    test_sample_mode: str = None
    mecpt_tile_size: float = 200.0
    mecpt_tile_halo: float = 20.0

    seed: int = 42
    class_weight_source: str = "sampled"
    min_class_weight: float = 0.5
    max_class_weight: float = 2.0
    class_weight_boosts: str = ""
    binary_loss_weight: float = 0.0
    semantic_loss_weight: float = 0.0
    semantic_class_weighting: str = "none"
    removal_loss_weight: float = 0.0
    relation_loss_weight: float = 0.0
    relation_loss_margin: float = 1.0

    def to_train_namespace(self):
        values = asdict(self)
        sensor = values.pop("sensor")
        radius = values.pop("radius")
        voxel_size = values.pop("voxel_size")
        sample_mode = values.pop("sample_mode")
        train_sample_mode = values.pop("train_sample_mode")
        val_sample_mode = values.pop("val_sample_mode")
        test_sample_mode = values.pop("test_sample_mode")

        values.update(
            {
                "dataset": "urb3dcd_v2",
                "urb3dcd_v2_all_labels": True,
                "urb3dcd_sensor": sensor,
                "urb3dcd_sample_mode": sample_mode,
                "urb3dcd_train_sample_mode": train_sample_mode,
                "urb3dcd_val_sample_mode": val_sample_mode,
                "urb3dcd_test_sample_mode": test_sample_mode,
                "urb3dcd_radius": radius,
                "urb3dcd_voxel_size": voxel_size,
                "train_cache_path": f"urb3dcd_v2_{sensor}_all_train.pt",
                "val_cache_path": f"urb3dcd_v2_{sensor}_all_val.pt",
                "test_cache_path": f"urb3dcd_v2_{sensor}_all_test.pt",
                "urb3dcd_removed_labels": "1",
                "urb3dcd_added_labels": "2,3",
                "urb3dcd_ignore_labels": "4,5,6",
                "freeze_ptv3_encoder": False,
                "disable_log_file": False,
                "num_group": 128,
                "group_size": 16,
                "stage2_depth": 2,
                "use_point_decoder": False,
                "disable_swapped_supervision": True,
                "token_match_score_grad": False,
                "temporal_embedding": False,
                "temporal_aux_loss_weight": 0.0,
                "changed_direction_loss_weight": 0.0,
                "existence_loss_weight": 0.0,
                "sample_focus_classes": "1,2,3,4,5,6",
                "sample_focus_power": 0.5,
                "threshold_sweep": "",
            }
        )
        return Namespace(**values)


class Urb3DCDV2Trainer:
    def __init__(self, config: Urb3DCDV2Config):
        self.config = config

    def run(self):
        args = self.config.to_train_namespace()
        configure_task(args)
        setup_log_file(args)
        print(args)
        require_cuda()
        set_seed(args.seed)
        if args.test_model:
            evaluate(args)
        else:
            train(args)


def parse_args():
    parser = ArgumentParser(description="Train/test Urb3DCD-V2 with PTv3 no-direction pipeline.")
    parser.add_argument("--data_root", default=Urb3DCDV2Config.data_root)
    parser.add_argument("--sensor", default=Urb3DCDV2Config.sensor, choices=("1-Lidar05", "5-MultiSensor"))
    parser.add_argument("--checkpoint", default=Urb3DCDV2Config.checkpoint)
    parser.add_argument("--warm_start_checkpoint", default=None)
    parser.add_argument("--log_path", default=None)
    parser.add_argument("--test_model", action="store_true")
    parser.add_argument("--eval_split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--test_votes", type=int, default=1)
    parser.add_argument(
        "--changed_conf_threshold",
        type=float,
        default=Urb3DCDV2Config.changed_conf_threshold,
        help=(
            "Optional changed-class confidence threshold used during evaluation; "
            "0 disables threshold calibration."
        ),
    )
    parser.add_argument("--class_bias_id", type=int, default=4)
    parser.add_argument(
        "--class_bias_sweep",
        default="",
        help="Comma list or inclusive start:end:step validation logit-bias sweep.",
    )
    parser.add_argument(
        "--class_bias_test_after_sweep",
        action="store_true",
        help="Test once using the class bias selected only from validation.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--train_samples_per_epoch", type=int, default=6000)
    parser.add_argument("--val_samples_per_epoch", type=int, default=1000)
    parser.add_argument("--test_samples_per_epoch", type=int, default=1000)
    parser.add_argument(
        "--train_sampling_strategy",
        choices=("weighted", "uniform"),
        default="weighted",
        help=(
            "Outer scene sampler. Use uniform when the cylinder-center sampler "
            "already applies ME-CPT-style class balancing."
        ),
    )
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
    parser.add_argument("--model_arch", default="ptv3_no_direction")
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
    )
    parser.add_argument("--token_match_k", type=int, default=8)
    parser.add_argument("--token_match_tau_feat", type=float, default=0.1)
    parser.add_argument("--token_match_tau_geo", type=float, default=1.0)
    parser.add_argument("--scalable_knn", action="store_true")
    parser.add_argument("--ptv3_order", default="z,z-trans,hilbert,hilbert-trans")
    parser.add_argument("--ptv3_patch_size", type=int, default=1024)
    parser.add_argument("--loss_type", choices=("focal_ce", "ce", "weighted_ce"), default="focal_ce")
    parser.add_argument("--lovasz_weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("step", "onecycle"), default="step")
    parser.add_argument("--step_size", type=int, default=15)
    parser.add_argument("--step_gamma", type=float, default=0.1)
    parser.add_argument("--onecycle_pct_start", type=float, default=0.1)
    parser.add_argument("--onecycle_div_factor", type=float, default=10.0)
    parser.add_argument("--onecycle_final_div_factor", type=float, default=1000.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--voxel_size", type=float, default=0.3)
    parser.add_argument("--change_ratio", type=float, default=0.5)
    parser.add_argument("--sample_mode", choices=("label1_multiclass", "label1_multiclass_cylinder", "label1_multiclass_independent_cylinder", "label1_multiclass_independent_uniform_cylinder", "label1_multiclass_natural_cylinder", "label1_multiclass_variable_cylinder", "label1_multiclass_mecpt_grid_cylinder", "label1_multiclass_mecpt_grid_paired_cylinder", "label1_multiclass_mecpt_grid_independent_uniform_cylinder", "mecpt_full_scene", "mecpt_full_scene_tiled"), default="label1_multiclass_cylinder")
    sample_mode_choices = (
        "label1_multiclass",
        "label1_multiclass_cylinder",
        "label1_multiclass_independent_cylinder",
        "label1_multiclass_independent_uniform_cylinder",
        "label1_multiclass_natural_cylinder",
        "label1_multiclass_variable_cylinder",
        "label1_multiclass_mecpt_grid_cylinder",
        "label1_multiclass_mecpt_grid_paired_cylinder",
        "label1_multiclass_mecpt_grid_independent_uniform_cylinder",
        "mecpt_full_scene",
        "mecpt_full_scene_tiled",
    )
    parser.add_argument("--train_sample_mode", choices=sample_mode_choices, default=None)
    parser.add_argument("--val_sample_mode", choices=sample_mode_choices, default=None)
    parser.add_argument("--test_sample_mode", choices=sample_mode_choices, default=None)
    parser.add_argument("--mecpt_tile_size", type=float, default=200.0)
    parser.add_argument("--mecpt_tile_halo", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_weight_source", choices=("sampled", "raw"), default="sampled")
    parser.add_argument("--min_class_weight", type=float, default=0.5)
    parser.add_argument("--max_class_weight", type=float, default=2.0)
    parser.add_argument("--class_weight_boosts", default="")
    parser.add_argument("--binary_loss_weight", type=float, default=0.0)
    parser.add_argument("--semantic_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--semantic_class_weighting",
        choices=("none", "misnet"),
        default="none",
    )
    parser.add_argument("--removal_loss_weight", type=float, default=0.0)
    parser.add_argument("--relation_loss_weight", type=float, default=0.0)
    parser.add_argument("--relation_loss_margin", type=float, default=1.0)
    return parser.parse_args()


def main():
    cli_args = parse_args()
    config = Urb3DCDV2Config(**vars(cli_args))
    Urb3DCDV2Trainer(config).run()


if __name__ == "__main__":
    main()
