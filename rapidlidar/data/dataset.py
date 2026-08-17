import os

import numpy as np
import yaml
from natsort import natsorted
from torch.utils.data import Dataset

from rapidlidar.data.transforms import (
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
    rotate_point_cloud,
)
import open3d as o3d


class SemanticKITTI(Dataset):
    def __init__(self, config_path, split="train", mode="coarse"):
        super().__init__()

        assert mode in ("coarse", "refine")
        self.mode = mode

        config = yaml.safe_load(open(config_path))["data"]
        self.data_dir = os.path.expanduser(config["data_dir"])
        split_map = {"train": "train", "validation": "validation", "test": "validation"}
        self.split = split_map[split]
        self.seqs = config[self.split]

        self.num_points = config["num_points"]
        self.x_range = config["x_range"]
        self.y_range = config["y_range"]
        self.z_range = config["z_range"]

        self._build_datapath_list()
        self.nr_data = len(self.points_datapath)

    def _build_datapath_list(self):
        self.points_datapath = []
        for seq in self.seqs:
            seq_path = os.path.join(self.data_dir, seq)
            gt_files = natsorted(os.listdir(os.path.join(seq_path, "gt")))
            for name in gt_files:
                self.points_datapath.append(os.path.join(seq_path, "gt", name))

    def _augment(self, points: np.ndarray) -> np.ndarray:
        points = points.copy()
        points[:, :3] = rotate_point_cloud(points[:, :3])
        points[:, :3] = rotate_perturbation_point_cloud(points[:, :3])
        points[:, :3] = random_scale_point_cloud(points[:, :3])
        points[:, :3] = random_flip_point_cloud(points[:, :3])
        return points

    def __getitem__(self, index):
        if self.mode == "refine":
            return self._getitem_refine(index)
        return self._getitem_coarse(index)

    def _getitem_coarse(self, index):
        gt_path = self.points_datapath[index]
        p_full = np.load(gt_path).reshape((-1, 3))
        p_part = np.load(gt_path.replace("gt", "input")).reshape((-1, 3))

        if self.split == "train":
            n_full = len(p_full)
            p_concat = self._augment(np.concatenate((p_full, p_part), axis=0))
            p_full = p_concat[:n_full]
            p_part = p_concat[n_full:]

        return p_full.astype(np.float32), p_part.astype(np.float32)

    def _getitem_refine(self, index):

        gt_path = self.points_datapath[index]
        p_full_pcd = o3d.io.read_point_cloud(gt_path)
        p_part_pcd = o3d.io.read_point_cloud(gt_path.replace("gt", "input"))

        if self.split == "train":
            p_full = np.asarray(p_full_pcd.voxel_down_sample(0.1).points, dtype=np.float32)
            p_part = np.asarray(p_part_pcd.points, dtype=np.float32)

            n_full = len(p_full)
            p_concat = self._augment(np.concatenate((p_full, p_part), axis=0))
            p_full = p_concat[:n_full]
            p_part = p_concat[n_full:]

            num_target = self.num_points * 2
            replace = len(p_full) < num_target
            p_full = p_full[np.random.choice(len(p_full), num_target, replace=replace)]
        else:
            p_full = np.asarray(p_full_pcd.points, dtype=np.float32)
            p_part = np.asarray(p_part_pcd.points, dtype=np.float32)

        return p_full.astype(np.float32), p_part.astype(np.float32)

    def __len__(self):
        return self.nr_data
