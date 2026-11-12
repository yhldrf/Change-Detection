"""Independent T0/T1 cylinder sampling for temporal point-cloud experiments.

The sampler deliberately never constructs a T0 sample by querying nearest
neighbors of selected T1 points. Both epochs retain their own measured
geometry; cross-temporal alignment is left to the model's soft matcher.
"""

import numpy as np
import torch


def _voxelize_indices(coord, voxel_size):
    if voxel_size is None or voxel_size <= 0 or coord.numel() == 0:
        return torch.arange(coord.shape[0], dtype=torch.long)
    grid = torch.floor(coord / float(voxel_size)).to(torch.int64).cpu().numpy()
    _, unique_idx = np.unique(grid, axis=0, return_index=True)
    return torch.from_numpy(np.sort(unique_idx)).long()


def _mecpt_voxelize_indices(coord, voxel_size):
    """Select one random point per voxel following ME-CPT's test loader."""
    if voxel_size is None or voxel_size <= 0 or coord.numel() == 0:
        return torch.arange(coord.shape[0], dtype=torch.long)

    discrete = np.floor(
        coord.detach().cpu().numpy() / float(voxel_size)
    ).astype(np.uint64, copy=False)
    keys = np.full(discrete.shape[0], np.uint64(14695981039346656037))
    for axis in range(discrete.shape[1]):
        keys *= np.uint64(1099511628211)
        keys = np.bitwise_xor(keys, discrete[:, axis])

    sorted_indices = np.argsort(keys)
    sorted_keys = keys[sorted_indices]
    _, counts = np.unique(sorted_keys, return_counts=True)
    starts = np.cumsum(np.insert(counts, 0, 0)[:-1])
    offsets = np.random.randint(0, counts.max(), counts.size) % counts
    return torch.from_numpy(sorted_indices[starts + offsets].copy()).long()


def _sample_from_pool(pool, count):
    if count <= 0 or pool.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    order = torch.randperm(pool.numel())
    if pool.numel() >= count:
        return pool[order[:count]]
    extra = pool[torch.randint(0, pool.numel(), (count - pool.numel(),))]
    return torch.cat([pool[order], extra], dim=0)


def _sample_uniform(point_count, num_points):
    if point_count <= 0:
        raise ValueError("Cannot sample from an empty temporal point cloud.")
    pool = torch.arange(point_count, dtype=torch.long)
    return _sample_from_pool(pool, num_points)


def _sample_label1_balanced(label1, num_points, change_ratio):
    """Preserve the existing T1 class-focused policy without pairing T0."""
    valid = torch.nonzero(label1 >= 0, as_tuple=False).view(-1)
    if valid.numel() == 0:
        return _sample_uniform(label1.numel(), num_points)

    unchanged = torch.nonzero(label1 == 0, as_tuple=False).view(-1)
    changed_labels = label1[label1 > 0].unique(sorted=True)
    num_changed = int(round(num_points * float(change_ratio)))
    num_changed = min(max(num_changed, 0), num_points)
    if changed_labels.numel() == 0:
        num_changed = 0

    parts = [_sample_from_pool(unchanged, num_points - num_changed)]
    if changed_labels.numel() > 0 and num_changed > 0:
        base = num_changed // changed_labels.numel()
        remainder = num_changed % changed_labels.numel()
        for pos, class_id in enumerate(changed_labels.tolist()):
            count = base + int(pos < remainder)
            pool = torch.nonzero(label1 == class_id, as_tuple=False).view(-1)
            parts.append(_sample_from_pool(pool, count))

    selected = torch.cat([part for part in parts if part.numel() > 0], dim=0)
    if selected.numel() < num_points:
        selected = torch.cat(
            [selected, _sample_from_pool(valid, num_points - selected.numel())],
            dim=0,
        )
    return selected[torch.randperm(selected.numel())[:num_points]]


def _choose_center(label1, center_label=None):
    valid_labels = label1[label1 >= 0]
    if valid_labels.numel() == 0:
        raise ValueError("T1 contains no valid change labels for center sampling.")
    labels, counts = valid_labels.unique(sorted=True, return_counts=True)
    if center_label is not None and torch.any(labels == int(center_label)):
        chosen_label = labels[labels == int(center_label)][0]
    else:
        weights = torch.sqrt(counts.float().mean() / counts.float())
        chosen_label = labels[torch.multinomial(weights / weights.sum(), 1).item()]
    center_pool = torch.nonzero(label1 == chosen_label, as_tuple=False).view(-1)
    return center_pool[torch.randint(0, center_pool.numel(), (1,)).item()]


def sample_independent_label1_multiclass_cylinder(
    sample,
    num_points,
    change_ratio=0.5,
    radius=50.0,
    voxel_size=0.3,
    min_crop_points=1024,
    center_label=None,
    center_xy=None,
    max_center_attempts=8,
    balance_t1=True,
    label_aware_center=True,
    normalize_to_t0_min=False,
):
    """Sample equal-size but index-independent T0 and T1 cylinder subsets.

    T0 is sampled uniformly after voxelization so its original structures are
    preserved without using T1 coordinates or labels. T1 retains the existing
    label-balanced point policy because the main seven-class target is T1.
    Equal output sizes keep the current dense batch interface, but equal array
    indices explicitly do not denote correspondence.
    """
    if num_points is None:
        raise ValueError("Independent dense sampling requires num_points.")
    if not 0 <= change_ratio <= 1:
        raise ValueError(f"change_ratio must be in [0, 1], got {change_ratio}.")

    coord0, feat0, label0 = sample["coord0"], sample["feat0"], sample["label0"]
    coord1, feat1, label1 = sample["coord1"], sample["feat1"], sample["label1"]

    crop_indices = None
    for _ in range(max(int(max_center_attempts), 1)):
        if center_xy is not None:
            current_center_xy = torch.as_tensor(
                center_xy, dtype=coord1.dtype, device=coord1.device
            )
        elif label_aware_center:
            center_idx = _choose_center(label1, center_label=center_label)
            current_center_xy = coord1[center_idx, :2]
        else:
            center_idx = torch.randint(0, coord1.shape[0], (1,)).item()
            current_center_xy = coord1[center_idx, :2]
        idx0 = torch.nonzero(
            torch.norm(coord0[:, :2] - current_center_xy, dim=-1) <= radius,
            as_tuple=False,
        ).view(-1)
        idx1 = torch.nonzero(
            torch.norm(coord1[:, :2] - current_center_xy, dim=-1) <= radius,
            as_tuple=False,
        ).view(-1)
        if idx0.numel() >= min_crop_points and idx1.numel() >= min_crop_points:
            crop_indices = (idx0, idx1)
            break
        if center_xy is not None:
            break

    # A full independently sampled scene is safer than falling back to the old
    # hard-pair sampler when a sparse cylinder cannot satisfy the size limit.
    if crop_indices is None:
        idx0 = torch.arange(coord0.shape[0], dtype=torch.long)
        idx1 = torch.arange(coord1.shape[0], dtype=torch.long)
    else:
        idx0, idx1 = crop_indices

    c0, c1 = coord0[idx0].clone(), coord1[idx1].clone()
    reference_min = c0.min(dim=0)[0] if normalize_to_t0_min else torch.minimum(
        c0.min(dim=0)[0], c1.min(dim=0)[0]
    )
    c0 -= reference_min
    c1 -= reference_min

    vox0 = _voxelize_indices(c0, voxel_size)
    vox1 = _voxelize_indices(c1, voxel_size)
    c0, c1 = c0[vox0], c1[vox1]
    f0, f1 = feat0[idx0][vox0], feat1[idx1][vox1]
    l0, l1 = label0[idx0][vox0], label1[idx1][vox1]

    pick0 = _sample_uniform(c0.shape[0], int(num_points))
    if balance_t1:
        pick1 = _sample_label1_balanced(l1, int(num_points), change_ratio)
    else:
        pick1 = _sample_uniform(c1.shape[0], int(num_points))

    result = dict(sample)
    result.update(
        coord0=c0[pick0].contiguous(),
        feat0=f0[pick0].contiguous(),
        label0=l0[pick0].contiguous(),
        coord1=c1[pick1].contiguous(),
        feat1=f1[pick1].contiguous(),
        label1=l1[pick1].contiguous(),
    )
    if "semantic0" in sample and "semantic1" in sample:
        s0 = sample["semantic0"][idx0][vox0]
        s1 = sample["semantic1"][idx1][vox1]
        result["semantic0"] = s0[pick0].contiguous()
        result["semantic1"] = s1[pick1].contiguous()
    return result


def sample_independent_uniform_label1_multiclass_cylinder(
    sample,
    num_points,
    change_ratio=0.5,
    radius=50.0,
    voxel_size=0.3,
    min_crop_points=None,
    center_label=None,
    center_xy=None,
    max_center_attempts=8,
    normalize_to_t0_min=False,
):
    """Label-agnostic fixed-size approximation of ME-CPT cylinder input.

    Cylinder centers can retain the benchmark's class-aware center policy, but
    point selection inside each cylinder is uniform for both epochs. Requiring
    at least ``num_points`` voxelized crop points avoids replacement in normal
    operation and preserves the natural test-label distribution.
    """
    if min_crop_points is None:
        min_crop_points = int(num_points)
    return sample_independent_label1_multiclass_cylinder(
        sample,
        num_points,
        change_ratio=change_ratio,
        radius=radius,
        voxel_size=voxel_size,
        min_crop_points=min_crop_points,
        center_label=center_label,
        center_xy=center_xy,
        max_center_attempts=max_center_attempts,
        balance_t1=False,
        normalize_to_t0_min=normalize_to_t0_min,
    )


def sample_variable_label1_multiclass_cylinder(
    sample,
    num_points=None,
    change_ratio=0.5,
    radius=50.0,
    voxel_size=0.3,
    min_crop_points=1024,
    center_label=None,
    center_xy=None,
    max_center_attempts=16,
):
    """ME-CPT-style variable-size cylinder without point-level resampling.

    A change-aware T1 center selects one shared spatial cylinder. T0 and T1
    are cropped and voxelized independently, and every surviving point is
    retained. ``num_points`` and ``change_ratio`` are accepted only for the
    common dataset interface and deliberately do not alter point retention.
    """
    del num_points, change_ratio
    coord0, feat0, label0 = sample["coord0"], sample["feat0"], sample["label0"]
    coord1, feat1, label1 = sample["coord1"], sample["feat1"], sample["label1"]

    crop_indices = None
    fallback_indices = None
    for _ in range(max(int(max_center_attempts), 1)):
        if center_xy is None:
            center_idx = _choose_center(label1, center_label=center_label)
            current_center_xy = coord1[center_idx, :2]
        else:
            current_center_xy = torch.as_tensor(
                center_xy,
                dtype=coord1.dtype,
                device=coord1.device,
            )
        idx0 = torch.nonzero(
            torch.norm(coord0[:, :2] - current_center_xy, dim=-1) <= radius,
            as_tuple=False,
        ).view(-1)
        idx1 = torch.nonzero(
            torch.norm(coord1[:, :2] - current_center_xy, dim=-1) <= radius,
            as_tuple=False,
        ).view(-1)
        if idx0.numel() > 0 and idx1.numel() > 0:
            fallback_indices = (idx0, idx1)
        if idx0.numel() > min_crop_points and idx1.numel() > min_crop_points:
            crop_indices = (idx0, idx1)
            break
        if center_xy is not None:
            break
    if crop_indices is None:
        crop_indices = fallback_indices
    if crop_indices is None:
        raise ValueError("Could not construct a non-empty paired cylinder.")

    idx0, idx1 = crop_indices
    c0, c1 = coord0[idx0].clone(), coord1[idx1].clone()
    # Match ME-CPT normalization: use the T0 minimum for both epochs.
    min0 = c0.min(dim=0)[0]
    c0 -= min0
    c1 -= min0

    # ME-CPT training randomly retains one representative from each occupied
    # voxel instead of always selecting the first point in storage order.
    vox0 = _mecpt_voxelize_indices(c0, voxel_size)
    vox1 = _mecpt_voxelize_indices(c1, voxel_size)
    if vox0.numel() == 0 or vox1.numel() == 0:
        raise ValueError("Voxelization produced an empty temporal cylinder.")

    result = dict(sample)
    result.update(
        coord0=c0[vox0].contiguous(),
        feat0=feat0[idx0][vox0].contiguous(),
        label0=label0[idx0][vox0].contiguous(),
        coord1=c1[vox1].contiguous(),
        feat1=feat1[idx1][vox1].contiguous(),
        label1=label1[idx1][vox1].contiguous(),
    )
    if "semantic0" in sample and "semantic1" in sample:
        result["semantic0"] = sample["semantic0"][idx0][vox0].contiguous()
        result["semantic1"] = sample["semantic1"][idx1][vox1].contiguous()
    return result


def sample_mecpt_full_scene(
    sample,
    num_points=None,
    change_ratio=0.5,
    radius=50.0,
    voxel_size=0.3,
    **_,
):
    """Prepare one complete scene using ME-CPT's Urb3DCD-V2 test protocol.

    No cylinder center or class label participates in evaluation. T0 and T1
    are normalized with the T0 minimum, voxelized independently, and all voxel
    representatives are retained. The model is evaluated on every retained T1
    point exactly once when each official scene is loaded once.
    """
    del num_points, change_ratio, radius
    coord0 = sample["coord0"].clone()
    coord1 = sample["coord1"].clone()
    if coord0.numel() == 0 or coord1.numel() == 0:
        raise ValueError("ME-CPT full-scene evaluation requires two non-empty clouds.")

    min0 = coord0.min(dim=0)[0]
    coord0 -= min0
    coord1 -= min0
    vox0 = _mecpt_voxelize_indices(coord0, voxel_size)
    vox1 = _mecpt_voxelize_indices(coord1, voxel_size)

    result = dict(sample)
    result.update(
        coord0=coord0[vox0].contiguous(),
        feat0=sample["feat0"][vox0].contiguous(),
        label0=sample["label0"][vox0].contiguous(),
        coord1=coord1[vox1].contiguous(),
        feat1=sample["feat1"][vox1].contiguous(),
        label1=sample["label1"][vox1].contiguous(),
    )
    if "semantic0" in sample and "semantic1" in sample:
        result["semantic0"] = sample["semantic0"][vox0].contiguous()
        result["semantic1"] = sample["semantic1"][vox1].contiguous()
    return result


def sample_natural_independent_label1_multiclass_cylinder(
    sample,
    num_points,
    change_ratio=0.5,
    radius=50.0,
    voxel_size=0.3,
    min_crop_points=None,
    center_label=None,
    max_center_attempts=8,
):
    """Uniform-center, uniform-point evaluation without label-based sampling.

    ``change_ratio`` and ``center_label`` are accepted for compatibility with
    the cached-dataset interface but do not affect sample selection.
    """
    del center_label
    if min_crop_points is None:
        min_crop_points = int(num_points)
    return sample_independent_label1_multiclass_cylinder(
        sample,
        num_points,
        change_ratio=change_ratio,
        radius=radius,
        voxel_size=voxel_size,
        min_crop_points=min_crop_points,
        max_center_attempts=max_center_attempts,
        balance_t1=False,
        label_aware_center=False,
    )
