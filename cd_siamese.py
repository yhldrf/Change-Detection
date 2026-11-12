import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CDDatasetStreet(Dataset):
    """Siamese-style SLPCCD dataset with 3 semantic change classes.

    Semantic labels:
    0 = unchanged, 1 = removed, 2 = added.
    """

    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        pre_split="PRE",
        radius=50,
        voxel_size=0.2,
        nameInPly="vertex",
        sample_num=1000,
        num_classes=3,
        loop=1,
        test_mode=False,
        test_cfg=None,
        cache=False,
    ):
        super().__init__()
        self.file_root = os.path.join(data_root, split)
        self.txt_path = os.path.join(data_root, split) + ".txt"
        self.voxel_size = voxel_size

        with open(self.txt_path, "r") as f:
            self.list = f.readlines()
        self.file_size = len(self.list)

    def _resolve_sample_path(self, relative_path):
        normalized_path = os.path.normpath(relative_path.strip().replace("\\", os.sep))
        candidates = [
            os.path.join(self.file_root, normalized_path),
            os.path.join(os.path.dirname(self.file_root), normalized_path),
            os.path.join(os.path.dirname(self.file_root), "train", normalized_path),
            os.path.join(os.path.dirname(self.file_root), "test", normalized_path),
            os.path.join(os.path.dirname(self.file_root), "data", normalized_path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def txt2sample(self, path):
        index = ["X", "Y", "Z", "Rf", "Gf", "Bf", "label"]
        with open(path, "r") as f:
            lines = f.readlines()
            head = lines[0][2:].strip("\n").split(" ")
            ids = tuple(head.index(i) for i in index)
        return np.loadtxt(path, skiprows=2, usecols=ids)

    def __getitem__(self, idx):
        sample_paths = self.list[idx].split()
        p0_path = self._resolve_sample_path(sample_paths[0])
        p1_path = self._resolve_sample_path(sample_paths[1])

        vertex0 = self.txt2sample(p0_path).astype(np.float32)
        vertex1 = self.txt2sample(p1_path).astype(np.float32)

        coord0 = torch.from_numpy(vertex0[:, :3])
        coord1 = torch.from_numpy(vertex1[:, :3])
        feat0 = torch.from_numpy(vertex0[:, 3:6])
        feat1 = torch.from_numpy(vertex1[:, 3:6])

        label0 = torch.from_numpy(vertex0[:, 6]).long()
        label1 = torch.from_numpy(vertex1[:, 6]).long()

        label0 = torch.where(label0 == 0, torch.zeros_like(label0), label0)
        label0 = torch.where(label0 == 1, torch.ones_like(label0), label0)
        label1 = torch.where(label1 == 0, torch.zeros_like(label1), label1)
        label1 = torch.where(label1 == 1, torch.full_like(label1, 2), label1)

        label0[(label0 < 0) | (label0 > 2)] = -1
        label1[(label1 < 0) | (label1 > 2)] = -1

        min0 = coord0.min(dim=0, keepdim=True)[0]
        coord0 = coord0 - min0
        coord1 = coord1 - min0

        return dict(
            coord0=coord0.contiguous(),
            feat0=feat0.contiguous(),
            label0=label0.contiguous(),
            coord1=coord1.contiguous(),
            feat1=feat1.contiguous(),
            label1=label1.contiguous(),
            scene_id=int(idx),
        )

    def __len__(self):
        return self.file_size
