import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import KDTree

try:
    from torch_cluster import fps as torch_cluster_fps
except ImportError:
    torch_cluster_fps = None

from cd_siamese import CDDatasetStreet
from independent_temporal_sampling import (
    sample_mecpt_full_scene,
    sample_independent_label1_multiclass_cylinder,
    sample_independent_uniform_label1_multiclass_cylinder,
    sample_natural_independent_label1_multiclass_cylinder,
    sample_variable_label1_multiclass_cylinder,
)

STREET_CLASS_LABELS = ["unchanged", "removed", "added"]
DEFAULT_NUM_POINTS = 8192
DEFAULT_CHANGE_RATIO = 0.5
NUM_CLASSES = len(STREET_CLASS_LABELS)


def _parse_int_set(raw_value):
    if raw_value is None:
        return set()
    if isinstance(raw_value, (list, tuple, set)):
        return {int(value) for value in raw_value}
    return {int(part.strip()) for part in str(raw_value).split(",") if part.strip()}


def _sample_from_pool(pool, count):
    if count <= 0 or pool.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    if pool.numel() >= count:
        return pool[torch.randperm(pool.numel())[:count]]
    extra = pool[torch.randint(0, pool.numel(), (count - pool.numel(),))]
    return torch.cat([pool[torch.randperm(pool.numel())], extra], dim=0)


def _voxelize_indices(coord, voxel_size):
    if voxel_size is None or voxel_size <= 0 or coord.numel() == 0:
        return torch.arange(coord.shape[0], dtype=torch.long)
    grid = torch.floor(coord / float(voxel_size)).to(torch.int64).cpu().numpy()
    _, unique_idx = np.unique(grid, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return torch.from_numpy(unique_idx).long()


def _sample_spatial_coverage_indices(coord, count, voxel_size):
    """Sample spatially distributed points without consulting labels.

    One random representative is first retained from every occupied voxel.
    When there are more occupied voxels than the point budget, voxels are
    sampled uniformly. Otherwise, all voxel representatives are retained and
    the remaining budget is filled from the points not selected yet.
    """
    num_points = coord.shape[0]
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    if num_points == 0:
        raise ValueError("Spatial-coverage sampling requires a non-empty cloud.")
    if voxel_size is None or voxel_size <= 0:
        return _sample_from_pool(torch.arange(num_points, dtype=torch.long), count)

    # Randomizing before np.unique prevents the representative of each voxel
    # from always being determined by the source-file point order.
    permutation = torch.randperm(num_points)
    shuffled_coord = coord[permutation]
    origin = shuffled_coord.amin(dim=0, keepdim=True)
    grid = torch.floor((shuffled_coord - origin) / float(voxel_size))
    grid = grid.to(torch.int64).cpu().numpy()
    _, first_positions = np.unique(grid, axis=0, return_index=True)
    representatives = permutation[torch.from_numpy(first_positions).long()]

    if representatives.numel() >= count:
        return representatives[torch.randperm(representatives.numel())[:count]]

    selected_mask = torch.zeros(num_points, dtype=torch.bool)
    selected_mask[representatives] = True
    remaining = torch.arange(num_points, dtype=torch.long)[~selected_mask]
    fill = _sample_from_pool(remaining, count - representatives.numel())
    if fill.numel() < count - representatives.numel():
        fallback = _sample_from_pool(
            torch.arange(num_points, dtype=torch.long),
            count - representatives.numel() - fill.numel(),
        )
        fill = torch.cat([fill, fallback], dim=0)
    selected = torch.cat([representatives, fill], dim=0)
    return selected[torch.randperm(selected.numel())]


def _voxel_representative_indices(coord, voxel_size):
    """Return one randomized representative for every occupied voxel."""
    if voxel_size is None or voxel_size <= 0 or coord.numel() == 0:
        return torch.arange(coord.shape[0], dtype=torch.long)
    permutation = torch.randperm(coord.shape[0])
    shuffled_coord = coord[permutation]
    origin = shuffled_coord.amin(dim=0, keepdim=True)
    grid = torch.floor((shuffled_coord - origin) / float(voxel_size))
    grid = grid.to(torch.int64).cpu().numpy()
    _, first_positions = np.unique(grid, axis=0, return_index=True)
    return permutation[torch.from_numpy(first_positions).long()]


def _iterative_farthest_point_indices(coord, count):
    """Exact iterative FPS fallback used when torch-cluster is unavailable."""
    num_points = coord.shape[0]
    if num_points <= count:
        return _sample_from_pool(torch.arange(num_points, dtype=torch.long), count)
    points = coord.float()
    selected = torch.empty(count, dtype=torch.long, device=points.device)
    min_distance = torch.full(
        (num_points,), float("inf"), dtype=points.dtype, device=points.device
    )
    farthest = torch.randint(0, num_points, (1,), device=points.device).item()
    for sample_idx in range(count):
        selected[sample_idx] = farthest
        centroid = points[farthest].unsqueeze(0)
        squared_distance = ((points - centroid) ** 2).sum(dim=-1)
        min_distance = torch.minimum(min_distance, squared_distance)
        farthest = int(min_distance.argmax().item())
    return selected.cpu()


def _sample_fps_indices(coord, count, candidate_voxel_size=0.1):
    """Hierarchical FPS suitable for raw clouds containing millions of points.

    Adaptive voxel representatives provide a spatially complete candidate set;
    FPS, rather than voxel selection, determines the final anchors.
    """
    num_points = coord.shape[0]
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    if num_points == 0:
        raise ValueError("FPS requires a non-empty point cloud.")
    if num_points <= count:
        return _sample_from_pool(torch.arange(num_points, dtype=torch.long), count)

    max_candidates = max(count * 4, count)
    voxel_size = max(float(candidate_voxel_size), 1e-4)
    best_candidates = None
    for _ in range(12):
        candidates = _voxel_representative_indices(coord, voxel_size)
        if candidates.numel() >= count and (
            best_candidates is None
            or candidates.numel() < best_candidates.numel()
        ):
            best_candidates = candidates
        if count <= candidates.numel() <= max_candidates:
            best_candidates = candidates
            break
        if candidates.numel() > max_candidates:
            scale = max((candidates.numel() / max_candidates) ** (1.0 / 3.0), 1.1)
            voxel_size *= scale
        else:
            voxel_size *= 0.75

    if best_candidates is None:
        best_candidates = torch.arange(num_points, dtype=torch.long)
    candidate_coord = coord[best_candidates].float().contiguous()

    if torch_cluster_fps is not None:
        ratio = min(float(count) / candidate_coord.shape[0], 1.0)
        local_idx = torch_cluster_fps(
            candidate_coord,
            ratio=ratio,
            random_start=True,
        ).long()
        if local_idx.numel() > count:
            local_idx = local_idx[:count]
        elif local_idx.numel() < count:
            selected_mask = torch.zeros(candidate_coord.shape[0], dtype=torch.bool)
            selected_mask[local_idx.cpu()] = True
            fill = _sample_from_pool(
                torch.arange(candidate_coord.shape[0], dtype=torch.long)[~selected_mask],
                count - local_idx.numel(),
            )
            local_idx = torch.cat([local_idx.cpu(), fill], dim=0)
        else:
            local_idx = local_idx.cpu()
    else:
        local_idx = _iterative_farthest_point_indices(candidate_coord, count)
    return best_candidates[local_idx]


def _read_binary_ply_fields(path):
    """Read simple binary PLY files used by the Urb3DCD-V2/Lyon dataset."""
    props = []
    count = None
    fmt = None
    with open(path, "rb") as handle:
        while True:
            line = handle.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                parts = line.split()
                count = int(parts[2])
            elif line.startswith("property"):
                parts = line.split()
                props.append((parts[1], parts[2]))
            elif line == "end_header":
                break
        if count is None or fmt is None:
            raise ValueError(f"Invalid PLY header: {path}")
        endian = ">" if fmt == "binary_big_endian" else "<"
        dtype_map = {
            "float": "f4",
            "float32": "f4",
            "double": "f8",
            "uchar": "u1",
            "uint8": "u1",
            "char": "i1",
            "int8": "i1",
        }
        dtype = np.dtype(
            [
                (
                    name,
                    dtype_map[prop_type]
                    if dtype_map[prop_type] in ("u1", "i1")
                    else endian + dtype_map[prop_type],
                )
                for prop_type, name in props
            ]
        )
        return np.fromfile(handle, dtype=dtype, count=count)


class CDDatasetUrb3DCDV2ForPipeline(Dataset):
    """
    Urb3DCD-V2/Lyon dataset adapter for engine.py.

    The source PLY stores x/y/z, RGB, label_ch and label_mono. ME-CPT supervises
    change labels from pointCloud1. This adapter converts them into the current
    3-class paired format:
      label0 == 1 -> removed
      label1 == 2 -> added
      otherwise 0 -> unchanged
    Removed T1 labels are projected to nearest T0 points so existing sampling
    and target construction can be reused.
    """

    SPLIT_DIR = {
        "train": "Train",
        "val": "Val",
        "test": "Test",
        "Train": "Train",
        "Val": "Val",
        "Test": "Test",
    }

    def __init__(
        self,
        data_root="data/IEEE_Dataset_V2_Lid05_MS",
        split="train",
        sensor="1-Lidar05",
        removed_labels="1",
        added_labels="2,3",
        ignore_labels="4,5,6",
        use_all_change_labels=False,
        center_coords=True,
        **_,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.sensor = sensor
        self.removed_labels = _parse_int_set(removed_labels)
        self.added_labels = _parse_int_set(added_labels)
        self.ignore_labels = _parse_int_set(ignore_labels)
        self.use_all_change_labels = use_all_change_labels
        self.center_coords = center_coords
        split_dir = self.SPLIT_DIR.get(split, split)
        self.root = self._resolve_split_root(data_root, split_dir)
        self.scenes = self._find_scenes()
        if not self.scenes:
            raise FileNotFoundError(f"No Urb3DCD-V2 scenes found under {self.root}")

    def _resolve_split_root(self, data_root, split_dir):
        candidates = []
        if self.sensor:
            candidates.append(os.path.join(data_root, self.sensor, split_dir))
        candidates.extend(
            [
                os.path.join(data_root, "1-Lidar05", split_dir),
                os.path.join(data_root, "5-MultiSensor", split_dir),
                os.path.join(data_root, split_dir),
            ]
        )
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        return candidates[0]

    def _find_scenes(self):
        scenes = []
        for name in sorted(os.listdir(self.root)):
            scene_dir = os.path.join(self.root, name)
            if not os.path.isdir(scene_dir):
                continue
            pc0 = os.path.join(scene_dir, "pointCloud0.ply")
            pc1 = os.path.join(scene_dir, "pointCloud1.ply")
            if os.path.isfile(pc0) and os.path.isfile(pc1):
                scenes.append((name, pc0, pc1))
        return scenes

    @staticmethod
    def _coords_and_feats(ply_array):
        coord = np.stack(
            [ply_array["x"], ply_array["y"], ply_array["z"]],
            axis=1,
        ).astype("float32")
        if all(name in ply_array.dtype.names for name in ("red", "green", "blue")):
            color_names = ("red", "green", "blue")
        elif all(name in ply_array.dtype.names for name in ("r", "g", "b")):
            color_names = ("r", "g", "b")
        else:
            color_names = None
        if color_names:
            feat = np.stack([ply_array[name] for name in color_names], axis=1).astype("float32") / 255.0
        else:
            feat = np.ones((coord.shape[0], 3), dtype="float32")
        return torch.from_numpy(coord), torch.from_numpy(feat)

    def _convert_labels(self, coord0, label_ch1, coord1):
        label0 = torch.zeros(coord0.shape[0], dtype=torch.long)
        label_ch1 = torch.from_numpy(label_ch1.astype("int64"))
        if self.use_all_change_labels:
            return label0.contiguous(), label_ch1.contiguous()

        label1 = torch.zeros(coord1.shape[0], dtype=torch.long)
        added_mask = torch.zeros_like(label_ch1, dtype=torch.bool)
        removed_mask = torch.zeros_like(label_ch1, dtype=torch.bool)
        ignore_mask = torch.zeros_like(label_ch1, dtype=torch.bool)
        for class_id in self.added_labels:
            added_mask |= label_ch1 == class_id
        for class_id in self.removed_labels:
            removed_mask |= label_ch1 == class_id
        for class_id in self.ignore_labels:
            ignore_mask |= label_ch1 == class_id

        label1[added_mask] = 2
        label1[ignore_mask] = -1

        if torch.any(removed_mask):
            tree = KDTree(coord0.detach().cpu().numpy())
            _, nn_idx = tree.query(coord1[removed_mask].detach().cpu().numpy(), k=1)
            label0[torch.from_numpy(nn_idx.reshape(-1)).long()] = 1
        return label0.contiguous(), label1.contiguous()

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        _, pc0_path, pc1_path = self.scenes[idx]
        pc0 = _read_binary_ply_fields(pc0_path)
        pc1 = _read_binary_ply_fields(pc1_path)
        coord0, feat0 = self._coords_and_feats(pc0)
        coord1, feat1 = self._coords_and_feats(pc1)
        if self.center_coords:
            joint_min = torch.minimum(coord0.min(dim=0)[0], coord1.min(dim=0)[0])
            coord0 = coord0 - joint_min
            coord1 = coord1 - joint_min
        label0, label1 = self._convert_labels(coord0, pc1["label_ch"], coord1)
        semantic0 = torch.from_numpy(pc0["label_mono"].astype("int64"))
        semantic1 = torch.from_numpy(pc1["label_mono"].astype("int64"))
        return {
            "coord0": coord0.contiguous(),
            "feat0": feat0.contiguous(),
            "label0": label0,
            "coord1": coord1.contiguous(),
            "feat1": feat1.contiguous(),
            "label1": label1,
            "semantic0": semantic0.contiguous(),
            "semantic1": semantic1.contiguous(),
        }


# Backward-compatible alias for older scripts.
CDDatasetNYCForPipeline = CDDatasetUrb3DCDV2ForPipeline


class CDDatasetHKSCDForPipeline(Dataset):
    """
    HKSCD adapter for the paired 3-class change-detection pipeline.

    HKSCD stores a binary cd_type mask in each time point. We convert direction
    from the time anchor:
      pointCloud0 cd_type == 1 -> removed
      pointCloud1 cd_type == 1 -> added
      otherwise -> unchanged
    The separate class field is kept as a 6-class semantic auxiliary label.
    """

    SPLIT_DIR = {
        "train": "Train",
        "val": "Val",
        "test": "Test",
        "Train": "Train",
        "Val": "Val",
        "Test": "Test",
    }

    def __init__(
        self,
        data_root="data/HKSCD",
        split="train",
        center_coords=True,
        **_,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.center_coords = center_coords
        split_dir = self.SPLIT_DIR.get(split, split)
        self.root = os.path.join(data_root, split_dir)
        self.scenes = self._find_scenes()
        if not self.scenes:
            raise FileNotFoundError(f"No HKSCD scenes found under {self.root}")

    def _find_scenes(self):
        scenes = []
        for name in sorted(os.listdir(self.root)):
            scene_dir = os.path.join(self.root, name)
            if not os.path.isdir(scene_dir):
                continue
            pc0 = os.path.join(scene_dir, "pointCloud0.ply")
            pc1 = os.path.join(scene_dir, "pointCloud1.ply")
            if os.path.isfile(pc0) and os.path.isfile(pc1):
                scenes.append((name, pc0, pc1))
        return scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        _, pc0_path, pc1_path = self.scenes[idx]
        pc0 = _read_binary_ply_fields(pc0_path)
        pc1 = _read_binary_ply_fields(pc1_path)
        coord0, feat0 = CDDatasetUrb3DCDV2ForPipeline._coords_and_feats(pc0)
        coord1, feat1 = CDDatasetUrb3DCDV2ForPipeline._coords_and_feats(pc1)
        if self.center_coords:
            joint_min = torch.minimum(coord0.min(dim=0)[0], coord1.min(dim=0)[0])
            coord0 = coord0 - joint_min
            coord1 = coord1 - joint_min

        cd0 = torch.from_numpy(pc0["cd_type"].astype("int64"))
        cd1 = torch.from_numpy(pc1["cd_type"].astype("int64"))
        label0 = torch.zeros(coord0.shape[0], dtype=torch.long)
        label1 = torch.zeros(coord1.shape[0], dtype=torch.long)
        label0[cd0 == 1] = 1
        label1[cd1 == 1] = 2

        semantic0 = torch.from_numpy(pc0["class"].astype("int64"))
        semantic1 = torch.from_numpy(pc1["class"].astype("int64"))
        return {
            "coord0": coord0.contiguous(),
            "feat0": feat0.contiguous(),
            "label0": label0.contiguous(),
            "coord1": coord1.contiguous(),
            "feat1": feat1.contiguous(),
            "label1": label1.contiguous(),
            "semantic0": semantic0.contiguous(),
            "semantic1": semantic1.contiguous(),
        }


def remap_sample_to_semantic(sample):
    updated = dict(sample)
    label0 = sample["label0"].long().clone()
    label1 = sample["label1"].long().clone()

    label0[(label0 < 0) | (label0 > 2)] = -1

    # Old caches use label1 in {2, 3}; new semantic data uses {0, 2}.
    if torch.any(label1 == 3):
        label1[label1 == 2] = 0
        label1[label1 == 3] = 2
    label1[(label1 < 0) | (label1 > 2)] = -1

    updated["label0"] = label0.contiguous()
    updated["label1"] = label1.contiguous()
    return updated


def sample_street_pair(sample, num_points, change_ratio=DEFAULT_CHANGE_RATIO):
    """Sample spatially matched T0/T1 pairs with change-aware anchor selection."""
    if num_points is None:
        return sample
    if not 0 <= change_ratio <= 1:
        raise ValueError(f"change_ratio must be in [0, 1], got {change_ratio}.")

    coord0, feat0, label0 = sample["coord0"], sample["feat0"], sample["label0"]
    coord1, feat1, label1 = sample["coord1"], sample["feat1"], sample["label1"]

    def sample_from_pool(pool, count):
        if count <= 0:
            return torch.empty(0, dtype=torch.long)
        if pool.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        if pool.numel() >= count:
            return pool[torch.randperm(pool.numel())[:count]]
        extra = pool[torch.randint(0, pool.numel(), (count - pool.numel(),))]
        return torch.cat([pool[torch.randperm(pool.numel())], extra], dim=0)

    def nearest_indices(src_coord, query_coord):
        if query_coord.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        tree = KDTree(src_coord.detach().cpu().numpy())
        _, nn_idx = tree.query(query_coord.detach().cpu().numpy(), k=1)
        return torch.from_numpy(nn_idx.reshape(-1)).long()

    def build_mutual_pairs():
        if coord0.shape[0] == 0 or coord1.shape[0] == 0:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        idx1_for_0 = nearest_indices(coord1, coord0)
        idx0_for_1 = nearest_indices(coord0, coord1)
        idx0 = torch.arange(coord0.shape[0], dtype=torch.long)
        mutual_mask = idx0_for_1[idx1_for_0] == idx0
        pair0 = idx0[mutual_mask]
        pair1 = idx1_for_0[mutual_mask]
        return pair0, pair1

    def build_changed_pairs(anchor_label, anchor_coord, other_coord, class_id, count):
        anchor_idx = torch.nonzero(anchor_label == class_id, as_tuple=False).view(-1)
        anchor_idx = sample_from_pool(anchor_idx, count)
        paired_idx = nearest_indices(other_coord, anchor_coord[anchor_idx])
        return anchor_idx, paired_idx

    pair0, pair1 = build_mutual_pairs()
    unchanged_pairs_mask = (label0[pair0] == 0) & (label1[pair1] == 0)
    unchanged_pair0 = pair0[unchanged_pairs_mask]
    unchanged_pair1 = pair1[unchanged_pairs_mask]

    num_changed = int(num_points * change_ratio)
    num_unchanged = max(num_points - num_changed, 0)
    num_removed = num_changed // 2
    num_added = num_changed - num_removed

    unchanged_pick = sample_from_pool(
        torch.arange(unchanged_pair0.numel(), dtype=torch.long), num_unchanged
    )
    idx0_unchanged = unchanged_pair0[unchanged_pick]
    idx1_unchanged = unchanged_pair1[unchanged_pick]

    idx0_removed, idx1_from_removed = build_changed_pairs(label0, coord0, coord1, 1, num_removed)
    idx1_added, idx0_from_added = build_changed_pairs(label1, coord1, coord0, 2, num_added)

    idx0 = torch.cat([idx0_unchanged, idx0_removed, idx0_from_added], dim=0)
    idx1 = torch.cat([idx1_unchanged, idx1_from_removed, idx1_added], dim=0)

    if idx0.numel() < num_points:
        fallback_count = num_points - idx0.numel()
        fallback_pairs = sample_from_pool(torch.arange(pair0.numel(), dtype=torch.long), fallback_count)
        idx0 = torch.cat([idx0, pair0[fallback_pairs]], dim=0)
        idx1 = torch.cat([idx1, pair1[fallback_pairs]], dim=0)

    if idx0.numel() == 0:
        raise ValueError("No valid T0/T1 point pairs could be sampled.")

    order = torch.randperm(idx0.numel())
    idx0 = idx0[order]
    idx1 = idx1[order]

    updated_sample = dict(sample)
    updated_sample["coord0"] = coord0[idx0].contiguous()
    updated_sample["feat0"] = feat0[idx0].contiguous()
    updated_sample["label0"] = label0[idx0].contiguous()
    updated_sample["coord1"] = coord1[idx1].contiguous()
    updated_sample["feat1"] = feat1[idx1].contiguous()
    updated_sample["label1"] = label1[idx1].contiguous()
    if "semantic0" in sample and "semantic1" in sample:
        updated_sample["semantic0"] = sample["semantic0"][idx0].contiguous()
        updated_sample["semantic1"] = sample["semantic1"][idx1].contiguous()
    return updated_sample


def sample_natural_bidirectional_street_pair(sample, num_points, voxel_size=0.0):
    """Build label-agnostic paired input from both temporal directions.

    Every measured T0 point contributes a T0->T1 nearest-neighbor pair and
    every measured T1 point contributes a T1->T0 pair. Uniform subsampling of
    their union preserves the observed branch distribution without consulting
    change labels while retaining the equal-length paired tensors expected by
    the three-class SLPCCD models.
    """
    if num_points is None:
        return sample

    coord0, feat0, label0 = sample["coord0"], sample["feat0"], sample["label0"]
    coord1, feat1, label1 = sample["coord1"], sample["feat1"], sample["label1"]
    semantic0 = sample.get("semantic0")
    semantic1 = sample.get("semantic1")

    keep0 = _voxelize_indices(coord0, voxel_size)
    keep1 = _voxelize_indices(coord1, voxel_size)
    coord0, feat0, label0 = coord0[keep0], feat0[keep0], label0[keep0]
    coord1, feat1, label1 = coord1[keep1], feat1[keep1], label1[keep1]
    if semantic0 is not None and semantic1 is not None:
        semantic0 = semantic0[keep0]
        semantic1 = semantic1[keep1]
    if coord0.shape[0] == 0 or coord1.shape[0] == 0:
        raise ValueError("Natural paired sampling requires non-empty T0 and T1.")

    tree0 = KDTree(coord0.detach().cpu().numpy())
    tree1 = KDTree(coord1.detach().cpu().numpy())
    _, nearest1_for_0 = tree1.query(coord0.detach().cpu().numpy(), k=1)
    _, nearest0_for_1 = tree0.query(coord1.detach().cpu().numpy(), k=1)

    anchors0 = torch.arange(coord0.shape[0], dtype=torch.long)
    anchors1 = torch.arange(coord1.shape[0], dtype=torch.long)
    pair0 = torch.cat(
        [
            anchors0,
            torch.from_numpy(nearest0_for_1.reshape(-1)).long(),
        ],
        dim=0,
    )
    pair1 = torch.cat(
        [
            torch.from_numpy(nearest1_for_0.reshape(-1)).long(),
            anchors1,
        ],
        dim=0,
    )

    pool_size = pair0.numel()
    if pool_size >= num_points:
        selected = torch.randperm(pool_size)[:num_points]
    else:
        selected = torch.cat(
            [
                torch.randperm(pool_size),
                torch.randint(0, pool_size, (num_points - pool_size,)),
            ],
            dim=0,
        )
        selected = selected[torch.randperm(selected.numel())]
    idx0 = pair0[selected]
    idx1 = pair1[selected]

    updated_sample = dict(sample)
    updated_sample["coord0"] = coord0[idx0].contiguous()
    updated_sample["feat0"] = feat0[idx0].contiguous()
    updated_sample["label0"] = label0[idx0].contiguous()
    updated_sample["coord1"] = coord1[idx1].contiguous()
    updated_sample["feat1"] = feat1[idx1].contiguous()
    updated_sample["label1"] = label1[idx1].contiguous()
    if semantic0 is not None and semantic1 is not None:
        updated_sample["semantic0"] = semantic0[idx0].contiguous()
        updated_sample["semantic1"] = semantic1[idx1].contiguous()
    return updated_sample


def sample_spatial_coverage_bidirectional_street_pair(
    sample,
    num_points,
    voxel_size=0.0,
    coverage_voxel_size=0.1,
):
    """Create label-agnostic, spatially covered bidirectional point pairs.

    Half of the requested pairs are anchored by spatially distributed T0
    points and the other half by spatially distributed T1 points. Each anchor
    is paired with its nearest measured point in the opposite epoch. This
    preserves equal-length tensors while ensuring that both temporal clouds
    contribute direct observations across the complete object extent.
    """
    if num_points is None:
        return sample
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}.")

    coord0, feat0, label0 = sample["coord0"], sample["feat0"], sample["label0"]
    coord1, feat1, label1 = sample["coord1"], sample["feat1"], sample["label1"]
    semantic0 = sample.get("semantic0")
    semantic1 = sample.get("semantic1")

    keep0 = _voxelize_indices(coord0, voxel_size)
    keep1 = _voxelize_indices(coord1, voxel_size)
    coord0, feat0, label0 = coord0[keep0], feat0[keep0], label0[keep0]
    coord1, feat1, label1 = coord1[keep1], feat1[keep1], label1[keep1]
    if semantic0 is not None and semantic1 is not None:
        semantic0 = semantic0[keep0]
        semantic1 = semantic1[keep1]
    if coord0.shape[0] == 0 or coord1.shape[0] == 0:
        raise ValueError("Spatial-coverage pairing requires non-empty T0 and T1.")

    num_anchor0 = num_points // 2
    num_anchor1 = num_points - num_anchor0
    anchor0 = _sample_spatial_coverage_indices(
        coord0, num_anchor0, coverage_voxel_size
    )
    anchor1 = _sample_spatial_coverage_indices(
        coord1, num_anchor1, coverage_voxel_size
    )

    tree0 = KDTree(coord0.detach().cpu().numpy())
    tree1 = KDTree(coord1.detach().cpu().numpy())
    _, nearest1 = tree1.query(coord0[anchor0].detach().cpu().numpy(), k=1)
    _, nearest0 = tree0.query(coord1[anchor1].detach().cpu().numpy(), k=1)
    nearest1 = torch.from_numpy(nearest1.reshape(-1)).long()
    nearest0 = torch.from_numpy(nearest0.reshape(-1)).long()

    idx0 = torch.cat([anchor0, nearest0], dim=0)
    idx1 = torch.cat([nearest1, anchor1], dim=0)
    order = torch.randperm(num_points)
    idx0, idx1 = idx0[order], idx1[order]

    updated_sample = dict(sample)
    updated_sample["coord0"] = coord0[idx0].contiguous()
    updated_sample["feat0"] = feat0[idx0].contiguous()
    updated_sample["label0"] = label0[idx0].contiguous()
    updated_sample["coord1"] = coord1[idx1].contiguous()
    updated_sample["feat1"] = feat1[idx1].contiguous()
    updated_sample["label1"] = label1[idx1].contiguous()
    if semantic0 is not None and semantic1 is not None:
        updated_sample["semantic0"] = semantic0[idx0].contiguous()
        updated_sample["semantic1"] = semantic1[idx1].contiguous()
    return updated_sample


def sample_fps_bidirectional_street_pair(
    sample,
    num_points,
    voxel_size=0.0,
    fps_candidate_voxel_size=0.1,
):
    """Sample FPS anchors from both epochs and construct bidirectional pairs."""
    if num_points is None:
        return sample
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}.")

    coord0, feat0, label0 = sample["coord0"], sample["feat0"], sample["label0"]
    coord1, feat1, label1 = sample["coord1"], sample["feat1"], sample["label1"]
    semantic0 = sample.get("semantic0")
    semantic1 = sample.get("semantic1")

    keep0 = _voxelize_indices(coord0, voxel_size)
    keep1 = _voxelize_indices(coord1, voxel_size)
    coord0, feat0, label0 = coord0[keep0], feat0[keep0], label0[keep0]
    coord1, feat1, label1 = coord1[keep1], feat1[keep1], label1[keep1]
    if semantic0 is not None and semantic1 is not None:
        semantic0 = semantic0[keep0]
        semantic1 = semantic1[keep1]
    if coord0.shape[0] == 0 or coord1.shape[0] == 0:
        raise ValueError("FPS pairing requires non-empty T0 and T1.")

    num_anchor0 = num_points // 2
    num_anchor1 = num_points - num_anchor0
    anchor0 = _sample_fps_indices(
        coord0, num_anchor0, candidate_voxel_size=fps_candidate_voxel_size
    )
    anchor1 = _sample_fps_indices(
        coord1, num_anchor1, candidate_voxel_size=fps_candidate_voxel_size
    )

    tree0 = KDTree(coord0.detach().cpu().numpy())
    tree1 = KDTree(coord1.detach().cpu().numpy())
    _, nearest1 = tree1.query(coord0[anchor0].detach().cpu().numpy(), k=1)
    _, nearest0 = tree0.query(coord1[anchor1].detach().cpu().numpy(), k=1)
    nearest1 = torch.from_numpy(nearest1.reshape(-1)).long()
    nearest0 = torch.from_numpy(nearest0.reshape(-1)).long()

    idx0 = torch.cat([anchor0, nearest0], dim=0)
    idx1 = torch.cat([nearest1, anchor1], dim=0)
    order = torch.randperm(num_points)
    idx0, idx1 = idx0[order], idx1[order]

    updated_sample = dict(sample)
    updated_sample["coord0"] = coord0[idx0].contiguous()
    updated_sample["feat0"] = feat0[idx0].contiguous()
    updated_sample["label0"] = label0[idx0].contiguous()
    updated_sample["coord1"] = coord1[idx1].contiguous()
    updated_sample["feat1"] = feat1[idx1].contiguous()
    updated_sample["label1"] = label1[idx1].contiguous()
    if semantic0 is not None and semantic1 is not None:
        updated_sample["semantic0"] = semantic0[idx0].contiguous()
        updated_sample["semantic1"] = semantic1[idx1].contiguous()
    return updated_sample


def sample_label1_multiclass_pair(sample, num_points, change_ratio=DEFAULT_CHANGE_RATIO):
    """Sample paired T0/T1 points while preserving all T1 label ids."""
    if num_points is None:
        return sample
    if not 0 <= change_ratio <= 1:
        raise ValueError(f"change_ratio must be in [0, 1], got {change_ratio}.")

    coord0, feat0, label0 = sample["coord0"], sample["feat0"], sample["label0"]
    coord1, feat1, label1 = sample["coord1"], sample["feat1"], sample["label1"]

    def sample_from_pool(pool, count):
        if count <= 0 or pool.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        if pool.numel() >= count:
            return pool[torch.randperm(pool.numel())[:count]]
        extra = pool[torch.randint(0, pool.numel(), (count - pool.numel(),))]
        return torch.cat([pool[torch.randperm(pool.numel())], extra], dim=0)

    def nearest_indices(src_coord, query_coord):
        if query_coord.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        tree = KDTree(src_coord.detach().cpu().numpy())
        _, nn_idx = tree.query(query_coord.detach().cpu().numpy(), k=1)
        return torch.from_numpy(nn_idx.reshape(-1)).long()

    idx1_for_0 = nearest_indices(coord1, coord0)
    idx0_for_1 = nearest_indices(coord0, coord1)
    all_idx0 = torch.arange(coord0.shape[0], dtype=torch.long)
    mutual_mask = idx0_for_1[idx1_for_0] == all_idx0
    pair0 = all_idx0[mutual_mask]
    pair1 = idx1_for_0[mutual_mask]

    unchanged_mask = (label0[pair0] == 0) & (label1[pair1] == 0)
    unchanged_pair0 = pair0[unchanged_mask]
    unchanged_pair1 = pair1[unchanged_mask]

    num_changed = int(num_points * change_ratio)
    num_unchanged = max(num_points - num_changed, 0)
    unchanged_pick = sample_from_pool(
        torch.arange(unchanged_pair0.numel(), dtype=torch.long),
        num_unchanged,
    )
    idx0_parts = [unchanged_pair0[unchanged_pick]]
    idx1_parts = [unchanged_pair1[unchanged_pick]]

    changed_labels = label1[label1 > 0].unique(sorted=True)
    if changed_labels.numel() > 0 and num_changed > 0:
        base_count = num_changed // changed_labels.numel()
        remainder = num_changed % changed_labels.numel()
        for label_pos, class_id in enumerate(changed_labels.tolist()):
            class_count = base_count + (1 if label_pos < remainder else 0)
            class_pool = torch.nonzero(label1 == class_id, as_tuple=False).view(-1)
            picked_idx1 = sample_from_pool(class_pool, class_count)
            picked_idx0 = nearest_indices(coord0, coord1[picked_idx1])
            idx0_parts.append(picked_idx0)
            idx1_parts.append(picked_idx1)

    idx0 = torch.cat(idx0_parts, dim=0)
    idx1 = torch.cat(idx1_parts, dim=0)

    if idx0.numel() < num_points:
        fallback_count = num_points - idx0.numel()
        fallback_pairs = sample_from_pool(torch.arange(pair0.numel(), dtype=torch.long), fallback_count)
        idx0 = torch.cat([idx0, pair0[fallback_pairs]], dim=0)
        idx1 = torch.cat([idx1, pair1[fallback_pairs]], dim=0)

    if idx0.numel() == 0:
        raise ValueError("No valid T0/T1 point pairs could be sampled.")

    order = torch.randperm(idx0.numel())
    idx0 = idx0[order]
    idx1 = idx1[order]

    updated_sample = dict(sample)
    updated_sample["coord0"] = coord0[idx0].contiguous()
    updated_sample["feat0"] = feat0[idx0].contiguous()
    updated_sample["label0"] = label0[idx0].contiguous()
    updated_sample["coord1"] = coord1[idx1].contiguous()
    updated_sample["feat1"] = feat1[idx1].contiguous()
    updated_sample["label1"] = label1[idx1].contiguous()
    if "semantic0" in sample and "semantic1" in sample:
        updated_sample["semantic0"] = sample["semantic0"][idx0].contiguous()
        updated_sample["semantic1"] = sample["semantic1"][idx1].contiguous()
    return updated_sample


def sample_label1_multiclass_cylinder_pair(
    sample,
    num_points,
    change_ratio=DEFAULT_CHANGE_RATIO,
    radius=50.0,
    voxel_size=0.3,
    min_crop_points=1024,
    center_label=None,
    center_xy=None,
):
    """ME-CPT-style Urb3DCD-V2 crop: label-balanced XY cylinder + voxelization."""
    coord0, feat0, label0 = sample["coord0"], sample["feat0"], sample["label0"]
    coord1, feat1, label1 = sample["coord1"], sample["feat1"], sample["label1"]
    valid_center_mask = label1 >= 0
    center_labels = label1[valid_center_mask]
    if center_labels.numel() == 0:
        return sample_label1_multiclass_pair(sample, num_points, change_ratio)

    if center_xy is None:
        labels, counts = center_labels.unique(sorted=True, return_counts=True)
        if center_label is not None and torch.any(labels == int(center_label)):
            chosen_label = labels[labels == int(center_label)][0]
        else:
            # ME-CPT balances crop centers by inverse sqrt class frequency.
            weights = torch.sqrt(counts.float().mean() / counts.float())
            weights = weights / weights.sum()
            chosen_label = labels[torch.multinomial(weights, 1).item()]
        center_pool = torch.nonzero(label1 == chosen_label, as_tuple=False).view(-1)
        center_idx = center_pool[torch.randint(0, center_pool.numel(), (1,)).item()]
        center_xy = coord1[center_idx, :2]
    else:
        center_xy = torch.as_tensor(
            center_xy,
            dtype=coord1.dtype,
            device=coord1.device,
        )

    dist0 = torch.norm(coord0[:, :2] - center_xy, dim=-1)
    dist1 = torch.norm(coord1[:, :2] - center_xy, dim=-1)
    idx0 = torch.nonzero(dist0 <= radius, as_tuple=False).view(-1)
    idx1 = torch.nonzero(dist1 <= radius, as_tuple=False).view(-1)
    if idx0.numel() < min_crop_points or idx1.numel() < min_crop_points:
        return sample_label1_multiclass_pair(sample, num_points, change_ratio)

    c0 = coord0[idx0].clone()
    c1 = coord1[idx1].clone()
    f0 = feat0[idx0]
    f1 = feat1[idx1]
    l0 = label0[idx0]
    l1 = label1[idx1]

    min0 = c0.min(dim=0)[0]
    c0 = c0 - min0
    c1 = c1 - min0

    vox0 = _voxelize_indices(c0, voxel_size)
    vox1 = _voxelize_indices(c1, voxel_size)
    if vox0.numel() < min_crop_points or vox1.numel() < min_crop_points:
        return sample_label1_multiclass_pair(sample, num_points, change_ratio)

    crop_sample = {
        "coord0": c0[vox0].contiguous(),
        "feat0": f0[vox0].contiguous(),
        "label0": l0[vox0].contiguous(),
        "coord1": c1[vox1].contiguous(),
        "feat1": f1[vox1].contiguous(),
        "label1": l1[vox1].contiguous(),
    }
    if "semantic0" in sample and "semantic1" in sample:
        crop_sample["semantic0"] = sample["semantic0"][idx0][vox0].contiguous()
        crop_sample["semantic1"] = sample["semantic1"][idx1][vox1].contiguous()
    return sample_label1_multiclass_pair(crop_sample, num_points, change_ratio)


def sample_pair_3class_cylinder(
    sample,
    num_points,
    change_ratio=DEFAULT_CHANGE_RATIO,
    radius=50.0,
    voxel_size=0.3,
    min_crop_points=1024,
    center_label=None,
):
    """Cylinder crop for paired 3-class datasets such as HKSCD."""
    coord0, feat0, label0 = sample["coord0"], sample["feat0"], sample["label0"]
    coord1, feat1, label1 = sample["coord1"], sample["feat1"], sample["label1"]

    centers = []
    center_labels = []
    if torch.any(label1 == 0):
        centers.append(coord1[label1 == 0])
        center_labels.append(torch.zeros((centers[-1].shape[0],), dtype=torch.long))
    if torch.any(label0 == 1):
        centers.append(coord0[label0 == 1])
        center_labels.append(torch.ones((centers[-1].shape[0],), dtype=torch.long))
    if torch.any(label1 == 2):
        centers.append(coord1[label1 == 2])
        center_labels.append(torch.full((centers[-1].shape[0],), 2, dtype=torch.long))
    if not centers:
        return sample_street_pair(sample, num_points, change_ratio)

    all_centers = torch.cat(centers, dim=0)
    all_center_labels = torch.cat(center_labels, dim=0)
    labels, counts = all_center_labels.unique(sorted=True, return_counts=True)
    if center_label is not None and torch.any(labels == int(center_label)):
        chosen_label = labels[labels == int(center_label)][0]
    else:
        weights = torch.sqrt(counts.float().mean() / counts.float())
        weights = weights / weights.sum()
        chosen_label = labels[torch.multinomial(weights, 1).item()]
    center_pool = torch.nonzero(all_center_labels == chosen_label, as_tuple=False).view(-1)
    center_idx = center_pool[torch.randint(0, center_pool.numel(), (1,)).item()]
    center_xy = all_centers[center_idx, :2]

    dist0 = torch.norm(coord0[:, :2] - center_xy, dim=-1)
    dist1 = torch.norm(coord1[:, :2] - center_xy, dim=-1)
    idx0 = torch.nonzero(dist0 <= radius, as_tuple=False).view(-1)
    idx1 = torch.nonzero(dist1 <= radius, as_tuple=False).view(-1)
    if idx0.numel() < min_crop_points or idx1.numel() < min_crop_points:
        return sample_street_pair(sample, num_points, change_ratio)

    c0 = coord0[idx0].clone()
    c1 = coord1[idx1].clone()
    f0 = feat0[idx0]
    f1 = feat1[idx1]
    l0 = label0[idx0]
    l1 = label1[idx1]
    min0 = c0.min(dim=0)[0]
    c0 = c0 - min0
    c1 = c1 - min0

    vox0 = _voxelize_indices(c0, voxel_size)
    vox1 = _voxelize_indices(c1, voxel_size)
    if vox0.numel() < min_crop_points or vox1.numel() < min_crop_points:
        return sample_street_pair(sample, num_points, change_ratio)

    crop_sample = {
        "coord0": c0[vox0].contiguous(),
        "feat0": f0[vox0].contiguous(),
        "label0": l0[vox0].contiguous(),
        "coord1": c1[vox1].contiguous(),
        "feat1": f1[vox1].contiguous(),
        "label1": l1[vox1].contiguous(),
    }
    if "semantic0" in sample and "semantic1" in sample:
        crop_sample["semantic0"] = sample["semantic0"][idx0][vox0].contiguous()
        crop_sample["semantic1"] = sample["semantic1"][idx1][vox1].contiguous()
    return sample_street_pair(crop_sample, num_points, change_ratio)


class CachedDataset(Dataset):
    def __init__(
        self,
        cache_path="train.pt",
        dataset_cls=CDDatasetStreet,
        num_points=DEFAULT_NUM_POINTS,
        change_ratio=DEFAULT_CHANGE_RATIO,
        remap_labels=True,
        sample_mode="street",
        virtual_sample_count=None,
        cylinder_radius=50.0,
        cylinder_voxel_size=0.3,
        street_voxel_size=0.0,
        street_coverage_voxel_size=0.1,
        require_semantic_labels=False,
        **dataset_kwargs,
    ):
        super().__init__()
        self.cache_path = cache_path
        self.dataset_cls = dataset_cls
        self.num_points = num_points
        self.change_ratio = change_ratio
        self.remap_labels = remap_labels
        self.sample_mode = sample_mode
        self.virtual_sample_count = virtual_sample_count
        self.cylinder_radius = cylinder_radius
        self.cylinder_voxel_size = cylinder_voxel_size
        self.street_voxel_size = street_voxel_size
        self.street_coverage_voxel_size = street_coverage_voxel_size
        self.require_semantic_labels = require_semantic_labels
        self.dataset_kwargs = dataset_kwargs
        self.samples = self._load_or_build()
        self._mecpt_grid_centers = None
        self._mecpt_grid_labels = None
        self._mecpt_grid_class_probs = None
        if self.sample_mode in (
            "label1_multiclass_mecpt_grid_cylinder",
            "label1_multiclass_mecpt_grid_paired_cylinder",
            "label1_multiclass_mecpt_grid_independent_uniform_cylinder",
        ):
            self._build_mecpt_grid_centers()

    def _load_or_build(self):
        if os.path.exists(self.cache_path):
            print("Loading cached dataset...")
            samples = torch.load(self.cache_path, map_location="cpu")
            has_semantic_labels = all(
                "semantic0" in sample and "semantic1" in sample
                for sample in samples
            )
            if not self.require_semantic_labels or has_semantic_labels:
                if self.remap_labels:
                    return [remap_sample_to_semantic(sample) for sample in samples]
                return samples
            print(
                "Cached samples do not contain label_mono semantic labels; "
                "rebuilding this cache for semantic auxiliary supervision."
            )

        print("Building dataset and saving...")
        dataset = self.dataset_cls(**self.dataset_kwargs)
        samples = self._build_samples(dataset)
        self._save_samples(samples)
        return samples

    def _build_samples(self, dataset):
        label_counter = torch.zeros(NUM_CLASSES, dtype=torch.long)
        samples = []
        for idx in tqdm(range(len(dataset)), desc="Caching dataset"):
            # Cache the raw sample so training can resample points dynamically.
            sample = dataset[idx]
            if self.remap_labels:
                sample = remap_sample_to_semantic(sample)
            samples.append(sample)
            label0 = sample["label0"].view(-1)
            label1 = sample["label1"].view(-1)

            labels = torch.cat([label0, label1], dim=0)
            labels = labels[labels >= 0]
            if labels.numel() > 0:
                required_classes = int(labels.max().item()) + 1
                if required_classes > label_counter.numel():
                    grown_counter = torch.zeros(required_classes, dtype=torch.long)
                    grown_counter[: label_counter.numel()] = label_counter
                    label_counter = grown_counter
                label_counter += torch.bincount(
                    labels,
                    minlength=label_counter.numel(),
                )[: label_counter.numel()]

        print("\n========== Dataset Label Statistics ==========")
        if label_counter.numel() == len(STREET_CLASS_LABELS):
            label_names = STREET_CLASS_LABELS
        else:
            label_names = [f"class_{class_id}" for class_id in range(label_counter.numel())]
        for class_id, class_name in enumerate(label_names):
            print(f"Class {class_id} ({class_name}):", int(label_counter[class_id]))
        return samples

    def _save_samples(self, samples):
        cache_dir = os.path.dirname(os.path.abspath(self.cache_path))
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save(samples, self.cache_path)

    def _build_mecpt_grid_centers(self):
        """Precompute ME-CPT-style 5 m T1 grid centers with majority labels."""
        grid_size = float(self.cylinder_radius) / 10.0
        center_records = []
        for sample_idx, sample in enumerate(self.samples):
            coord1 = sample["coord1"].float()
            label1 = sample["label1"].long()
            valid = label1 >= 0
            if not torch.any(valid):
                continue
            coord_valid = coord1[valid]
            label_valid = label1[valid]
            grid = torch.round(coord_valid / grid_size).long()
            unique_grid, inverse = torch.unique(
                grid,
                dim=0,
                sorted=True,
                return_inverse=True,
            )
            for cell_idx in range(unique_grid.shape[0]):
                member_mask = inverse == cell_idx
                labels = label_valid[member_mask]
                if labels.numel() == 0:
                    continue
                majority_label = int(torch.bincount(labels).argmax().item())
                center_xy = coord_valid[member_mask, :2].mean(dim=0)
                center_records.append(
                    [
                        float(sample_idx),
                        float(center_xy[0].item()),
                        float(center_xy[1].item()),
                        float(majority_label),
                    ]
                )
        if not center_records:
            raise ValueError("No valid ME-CPT grid centers were constructed.")
        self._mecpt_grid_centers = torch.tensor(center_records, dtype=torch.float32)
        self._mecpt_grid_labels = self._mecpt_grid_centers[:, 3].long()
        labels, counts = self._mecpt_grid_labels.unique(sorted=True, return_counts=True)
        probs = torch.sqrt(counts.float().mean() / counts.float())
        probs = probs / probs.sum()
        self._mecpt_grid_class_labels = labels
        self._mecpt_grid_class_probs = probs
        summary = ", ".join(
            f"{int(label)}:{int(count)}"
            for label, count in zip(labels.tolist(), counts.tolist())
        )
        print(
            "ME-CPT grid centers "
            f"(grid={grid_size:g} m, total={len(center_records)}): {summary}"
        )

    def _sample_mecpt_grid_center(self, center_label=None):
        if self._mecpt_grid_centers is None:
            self._build_mecpt_grid_centers()
        if center_label is None:
            class_pos = torch.multinomial(self._mecpt_grid_class_probs, 1).item()
            chosen_label = int(self._mecpt_grid_class_labels[class_pos].item())
        else:
            chosen_label = int(center_label)
        candidates = torch.nonzero(
            self._mecpt_grid_labels == chosen_label,
            as_tuple=False,
        ).view(-1)
        if candidates.numel() == 0:
            raise ValueError(f"No ME-CPT grid center for label {chosen_label}.")
        row = self._mecpt_grid_centers[
            candidates[torch.randint(0, candidates.numel(), (1,)).item()]
        ]
        return int(row[0].item()), row[1:3]

    def __len__(self):
        return self.virtual_sample_count or len(self.samples)

    def __getitem__(self, idx):
        # Apply random point sampling at access time to preserve augmentation
        # variability across epochs instead of freezing one sampled version.
        
        sample = self.samples[idx % len(self.samples)]
        if self.sample_mode == "natural_bidirectional_pair":
            return sample_natural_bidirectional_street_pair(
                sample,
                self.num_points,
                voxel_size=self.street_voxel_size,
            )
        if self.sample_mode == "spatial_coverage_bidirectional_pair":
            return sample_spatial_coverage_bidirectional_street_pair(
                sample,
                self.num_points,
                voxel_size=self.street_voxel_size,
                coverage_voxel_size=self.street_coverage_voxel_size,
            )
        if self.sample_mode == "fps_bidirectional_pair":
            return sample_fps_bidirectional_street_pair(
                sample,
                self.num_points,
                voxel_size=self.street_voxel_size,
                fps_candidate_voxel_size=self.street_coverage_voxel_size,
            )
        if self.sample_mode == "mecpt_full_pair":
            return sample_mecpt_full_scene(
                sample,
                voxel_size=self.street_voxel_size,
            )
        if self.sample_mode == "label1_multiclass_cylinder":
            return sample_label1_multiclass_cylinder_pair(
                sample,
                self.num_points,
                self.change_ratio,
                radius=self.cylinder_radius,
                voxel_size=self.cylinder_voxel_size,
            )
        if self.sample_mode == "label1_multiclass_independent_cylinder":
            return sample_independent_label1_multiclass_cylinder(
                sample,
                self.num_points,
                self.change_ratio,
                radius=self.cylinder_radius,
                voxel_size=self.cylinder_voxel_size,
            )
        if self.sample_mode == "label1_multiclass_independent_uniform_cylinder":
            return sample_independent_uniform_label1_multiclass_cylinder(
                sample,
                self.num_points,
                self.change_ratio,
                radius=self.cylinder_radius,
                voxel_size=self.cylinder_voxel_size,
            )
        if self.sample_mode == "label1_multiclass_variable_cylinder":
            return sample_variable_label1_multiclass_cylinder(
                sample,
                self.num_points,
                self.change_ratio,
                radius=self.cylinder_radius,
                voxel_size=self.cylinder_voxel_size,
            )
        if self.sample_mode == "label1_multiclass_mecpt_grid_cylinder":
            scene_idx, center_xy = self._sample_mecpt_grid_center()
            return sample_variable_label1_multiclass_cylinder(
                self.samples[scene_idx],
                self.num_points,
                self.change_ratio,
                radius=self.cylinder_radius,
                voxel_size=self.cylinder_voxel_size,
                center_xy=center_xy,
            )
        if self.sample_mode == "label1_multiclass_mecpt_grid_paired_cylinder":
            scene_idx, center_xy = self._sample_mecpt_grid_center()
            return sample_label1_multiclass_cylinder_pair(
                self.samples[scene_idx],
                self.num_points,
                self.change_ratio,
                radius=self.cylinder_radius,
                voxel_size=self.cylinder_voxel_size,
                center_xy=center_xy,
            )
        if self.sample_mode == "label1_multiclass_mecpt_grid_independent_uniform_cylinder":
            scene_idx, center_xy = self._sample_mecpt_grid_center()
            return sample_independent_uniform_label1_multiclass_cylinder(
                self.samples[scene_idx],
                self.num_points,
                self.change_ratio,
                radius=self.cylinder_radius,
                voxel_size=self.cylinder_voxel_size,
                center_xy=center_xy,
                normalize_to_t0_min=True,
            )
        if self.sample_mode == "mecpt_full_scene":
            return sample_mecpt_full_scene(
                sample,
                voxel_size=self.cylinder_voxel_size,
            )
        if self.sample_mode == "label1_multiclass_natural_cylinder":
            return sample_natural_independent_label1_multiclass_cylinder(
                sample,
                self.num_points,
                self.change_ratio,
                radius=self.cylinder_radius,
                voxel_size=self.cylinder_voxel_size,
            )
        if self.sample_mode == "label1_multiclass":
            return sample_label1_multiclass_pair(sample, self.num_points, self.change_ratio)
        if self.sample_mode == "pair_3class_cylinder":
            return sample_pair_3class_cylinder(
                sample,
                self.num_points,
                self.change_ratio,
                radius=self.cylinder_radius,
                voxel_size=self.cylinder_voxel_size,
            )
        return sample_street_pair(sample, self.num_points, self.change_ratio)

    def available_center_labels(self):
        if self.sample_mode == "pair_3class_cylinder":
            return [0, 1, 2]
        if self.sample_mode not in (
            "label1_multiclass_cylinder",
            "label1_multiclass_independent_cylinder",
            "label1_multiclass_independent_uniform_cylinder",
            "label1_multiclass_natural_cylinder",
            "label1_multiclass_variable_cylinder",
            "label1_multiclass_mecpt_grid_cylinder",
            "label1_multiclass_mecpt_grid_paired_cylinder",
            "label1_multiclass_mecpt_grid_independent_uniform_cylinder",
            "mecpt_full_scene",
        ):
            return []
        if self.sample_mode in (
            "label1_multiclass_mecpt_grid_cylinder",
            "label1_multiclass_mecpt_grid_paired_cylinder",
            "label1_multiclass_mecpt_grid_independent_uniform_cylinder",
        ):
            if self._mecpt_grid_labels is None:
                self._build_mecpt_grid_centers()
            return sorted(int(value) for value in self._mecpt_grid_labels.unique().tolist())
        labels = set()
        for sample in self.samples:
            labels.update(int(value) for value in sample["label1"].unique().tolist() if value >= 0)
        return sorted(labels)

    def sample_for_center_label(self, center_label):
        if self.sample_mode == "pair_3class_cylinder":
            candidates = []
            for sample in self.samples:
                label_id = int(center_label)
                has_label = torch.any(sample["label1"] == label_id)
                if label_id == 1:
                    has_label = torch.any(sample["label0"] == 1)
                if has_label:
                    candidates.append(sample)
        elif self.sample_mode in (
            "label1_multiclass_mecpt_grid_cylinder",
            "label1_multiclass_mecpt_grid_paired_cylinder",
            "label1_multiclass_mecpt_grid_independent_uniform_cylinder",
        ):
            scene_idx, center_xy = self._sample_mecpt_grid_center(
                center_label=center_label
            )
            if self.sample_mode == "label1_multiclass_mecpt_grid_independent_uniform_cylinder":
                return sample_independent_uniform_label1_multiclass_cylinder(
                    self.samples[scene_idx],
                    self.num_points,
                    self.change_ratio,
                    radius=self.cylinder_radius,
                    voxel_size=self.cylinder_voxel_size,
                    center_xy=center_xy,
                    normalize_to_t0_min=True,
                )
            if self.sample_mode == "label1_multiclass_mecpt_grid_paired_cylinder":
                return sample_label1_multiclass_cylinder_pair(
                    self.samples[scene_idx],
                    self.num_points,
                    self.change_ratio,
                    radius=self.cylinder_radius,
                    voxel_size=self.cylinder_voxel_size,
                    center_xy=center_xy,
                )
            return sample_variable_label1_multiclass_cylinder(
                self.samples[scene_idx],
                self.num_points,
                self.change_ratio,
                radius=self.cylinder_radius,
                voxel_size=self.cylinder_voxel_size,
                center_xy=center_xy,
            )
        else:
            candidates = [
                sample
                for sample in self.samples
                if torch.any(sample["label1"] == int(center_label))
            ]
        if not candidates:
            raise ValueError(f"No cached sample contains center label {center_label}.")
        sample = candidates[torch.randint(0, len(candidates), (1,)).item()]
        if self.sample_mode == "pair_3class_cylinder":
            return sample_pair_3class_cylinder(
                sample,
                self.num_points,
                self.change_ratio,
                radius=self.cylinder_radius,
                voxel_size=self.cylinder_voxel_size,
                center_label=center_label,
            )
        sampler = (
            sample_independent_label1_multiclass_cylinder
            if self.sample_mode == "label1_multiclass_independent_cylinder"
            else (
                sample_independent_uniform_label1_multiclass_cylinder
                if self.sample_mode
                == "label1_multiclass_independent_uniform_cylinder"
                else (
                    sample_natural_independent_label1_multiclass_cylinder
                    if self.sample_mode == "label1_multiclass_natural_cylinder"
                    else (
                        sample_variable_label1_multiclass_cylinder
                        if self.sample_mode == "label1_multiclass_variable_cylinder"
                        else sample_label1_multiclass_cylinder_pair
                    )
                )
            )
        )
        return sampler(
            sample,
            self.num_points,
            self.change_ratio,
            radius=self.cylinder_radius,
            voxel_size=self.cylinder_voxel_size,
            center_label=center_label,
        )


class MECPTFullCoverageTileDataset(Dataset):
    """Deterministic full-coverage test tiles for memory-limited GPUs.

    T1 core tiles form a disjoint partition of each complete voxelized scene.
    A spatial halo from both epochs supplies context, while halo T1 labels are
    ignored so every official test point contributes to metrics exactly once.
    Each tile is translated by its T0-context minimum to match the coordinate
    normalization used by ME-CPT-style training cylinders. Translation does
    not change neighborhoods, correspondence distances, or evaluated points.
    """

    def __init__(self, dataset, voxel_size=0.3, tile_size=200.0, halo=20.0):
        if tile_size <= 0 or halo < 0:
            raise ValueError("tile_size must be positive and halo non-negative.")
        self.tile_size = float(tile_size)
        self.halo = float(halo)
        self.scenes = []
        self.tiles = []
        for scene_idx, raw_sample in enumerate(dataset.samples):
            scene = sample_mecpt_full_scene(raw_sample, voxel_size=voxel_size)
            origin = scene["coord1"][:, :2].amin(dim=0)
            keys = torch.floor(
                (scene["coord1"][:, :2] - origin) / self.tile_size
            ).long()
            unique_keys = torch.unique(keys, dim=0)
            self.scenes.append((scene, origin, keys))
            self.tiles.extend(
                (scene_idx, key.clone()) for key in unique_keys
            )

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        scene_idx, tile_key = self.tiles[idx]
        scene, origin, keys1 = self.scenes[scene_idx]
        low = origin + tile_key.to(origin.dtype) * self.tile_size
        high = low + self.tile_size
        context_low = low - self.halo
        context_high = high + self.halo

        coord0_xy = scene["coord0"][:, :2]
        coord1_xy = scene["coord1"][:, :2]
        keep0 = torch.all(
            (coord0_xy >= context_low) & (coord0_xy < context_high), dim=1
        )
        keep1 = torch.all(
            (coord1_xy >= context_low) & (coord1_xy < context_high), dim=1
        )
        idx0 = torch.nonzero(keep0, as_tuple=False).view(-1)
        idx1 = torch.nonzero(keep1, as_tuple=False).view(-1)
        if idx0.numel() == 0 or idx1.numel() == 0:
            raise ValueError(f"Empty temporal context for full-coverage tile {idx}.")

        core1 = torch.all(keys1[idx1] == tile_key, dim=1)
        label1 = scene["label1"][idx1].clone()
        label1[~core1] = -1
        coord0 = scene["coord0"][idx0].clone()
        coord1 = scene["coord1"][idx1].clone()
        tile_min0 = coord0.amin(dim=0)
        coord0 -= tile_min0
        coord1 -= tile_min0
        result = {
            "coord0": coord0.contiguous(),
            "feat0": scene["feat0"][idx0].contiguous(),
            "label0": scene["label0"][idx0].contiguous(),
            "coord1": coord1.contiguous(),
            "feat1": scene["feat1"][idx1].contiguous(),
            "label1": label1.contiguous(),
        }
        if "semantic0" in scene and "semantic1" in scene:
            result["semantic0"] = scene["semantic0"][idx0].contiguous()
            result["semantic1"] = scene["semantic1"][idx1].contiguous()
        return result



def build_cached_street_dataloader(
    batch_size=4,
    shuffle=True,
    num_workers=0,
    cache_path="train.pt",
    dataset_cls=CDDatasetStreet,
    num_points=DEFAULT_NUM_POINTS,
    change_ratio=DEFAULT_CHANGE_RATIO,
    sampler=None,
    **dataset_kwargs,
):
    dataset = CachedDataset(
        cache_path=cache_path,
        dataset_cls=dataset_cls,
        num_points=num_points,
        change_ratio=change_ratio,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        num_workers=num_workers,
    )



def main():
    parser = argparse.ArgumentParser(description="Build or load a cached CDDatasetStreet dataset.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/SLPCCD/SHREC2020-CD",
        help="Path to the Street dataset root, e.g. data/SLPCCD/SHREC2020-CD"   
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to cache.")
    parser.add_argument("--voxel-size", type=float, default=0.5, help="Voxel size for CDDatasetStreet.")
    parser.add_argument("--cache-path", type=str, default="train.pt", help="Path to the cached dataset file.")
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS, help="Number of points to sample per point cloud.")
    parser.add_argument("--change-ratio", type=float, default=DEFAULT_CHANGE_RATIO, help="Target ratio of sampled changed points per branch.")
    args = parser.parse_args()

    dataset = CachedDataset(
        cache_path=args.cache_path,
        data_root=args.data_root,
        split=args.split,
        voxel_size=args.voxel_size,
        num_points=args.num_points,
        change_ratio=args.change_ratio,
    )
    print(f"Dataset ready with {len(dataset)} samples.")
    # =====================
    # val
    # =====================
    dataset_val =CachedDataset(
        cache_path="val.pt",
        data_root=args.data_root,
        split="val",
        voxel_size=args.voxel_size,
        num_points=args.num_points,
        change_ratio=args.change_ratio,
    )
    print(f"Dataset ready with {len(dataset_val)} samples.")
    # =====================
    # test
    # =====================
    dataset_test=CachedDataset(
        cache_path="test.pt",
        data_root=args.data_root,
        split="test",
        voxel_size=args.voxel_size,
        num_points=args.num_points,
        change_ratio=args.change_ratio,
    )
    print(f"Dataset ready with {len(dataset_test)} samples.")
if __name__ == "__main__":
    main()
