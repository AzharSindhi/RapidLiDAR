from typing import NamedTuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from rapidlidar.data.dataset import SemanticKITTI
from rapidlidar.models.adaptive_init import AdaptiveInitializationModule
from rapidlidar.models.feature_extraction import FeatureExtractionModule
from rapidlidar.models.reconstruction import MultiScaleReconstructionModule

pl.seed_everything(42)

# Imported lazily so inference (and `from_pretrained`) don't require the Chamfer
# CUDA extension, which is only needed for the training/eval loss.
sqrt_chamfer_distance = None


def get_chamfer_distance():
    global sqrt_chamfer_distance
    if sqrt_chamfer_distance is None:
        from rapidlidar.losses.chamfer import sqrt_chamfer_distance as _cd
        sqrt_chamfer_distance = _cd
    return sqrt_chamfer_distance


class RapidLiDAROutput(NamedTuple):
    points: torch.Tensor
    point_features: torch.Tensor
    voxel_features: list
    bev_maps: list
    projected_bev_patches: list


class RapidLiDAR(pl.LightningModule):
    def __init__(self, config_path="configs/rapidlidar.yaml",
                 learning_rate=1e-4, batch_size=4,
                 num_reconstruction_rounds=2,
                 predict_residual=True,
                 init_noise_std=0.1,
                 displacement_scale=50.0,
                 voxel_size=0.3,
                 x_range=None, y_range=None, z_range=None, num_points=None):
        super().__init__()
        self.save_hyperparameters()

        self.config_path = config_path
        # Data ranges can be passed explicitly (so `from_pretrained` works without
        # the config file on disk); otherwise they are read from `config_path`.
        if None in (x_range, y_range, z_range, num_points):
            with open(config_path, "r") as f:
                import yaml
                cfg = yaml.safe_load(f)["data"]
            x_range = cfg["x_range"] if x_range is None else x_range
            y_range = cfg["y_range"] if y_range is None else y_range
            z_range = cfg["z_range"] if z_range is None else z_range
            num_points = cfg["num_points"] if num_points is None else num_points

        self.num_workers = 4
        self.predict_residual = predict_residual
        self.init_noise_std = init_noise_std
        self.displacement_scale = displacement_scale

        self.num_points = num_points
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.latent_dim = 512
        self.voxel_size = voxel_size

        self.feature_extractor = FeatureExtractionModule(
            x_range=self.x_range, y_range=self.y_range, z_range=self.z_range,
            voxel_size=voxel_size, latent_dim=self.latent_dim,
        )
        self.adaptive_init = AdaptiveInitializationModule(feature_dim=self.latent_dim + 3)
        self.reconstruction = MultiScaleReconstructionModule(
            feature_dims=self.feature_extractor.feature_dims,
            latent_dim=self.latent_dim, num_layers=num_reconstruction_rounds,
            x_range=self.x_range, y_range=self.y_range,
        )

    def build_initial_queries(self, x_partial: torch.Tensor, num_points: int) -> torch.Tensor:
        return AdaptiveInitializationModule.repeat_and_perturb(x_partial, num_points, self.init_noise_std)

    def clamp_to_bbox(self, points: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            points[..., 0].clamp(self.x_range[0], self.x_range[1]),
            points[..., 1].clamp(self.y_range[0], self.y_range[1]),
            points[..., 2].clamp(self.z_range[0], self.z_range[1]),
        ], dim=-1)

    def forward(self, x_partial: torch.Tensor, up_factor: int = 10) -> RapidLiDAROutput:
        num_points = x_partial.shape[1] * up_factor

        voxel_features, bev_dense = self.feature_extractor.extract(x_partial)
        bev_multiscale = self.feature_extractor.project_voxels_to_bev(voxel_features)
        grids = voxel_features + [bev_dense]
        bev_maps = bev_multiscale + [bev_dense]

        tilde_p = self.build_initial_queries(x_partial, num_points)
        tilde_p = self.clamp_to_bbox(tilde_p)

        point_features = self.feature_extractor.interpolate(grids, tilde_p)
        p_init = self.adaptive_init(tilde_p, torch.cat([point_features, tilde_p], dim=-1), self.displacement_scale)
        p_init = self.clamp_to_bbox(p_init)

        point_features = self.feature_extractor.interpolate(grids, p_init)
        points, point_features, projected_bev_patches = self.reconstruction(p_init, point_features, bev_maps)

        return RapidLiDAROutput(points, point_features, grids, bev_maps, projected_bev_patches)

    @torch.no_grad()
    def _log_point_clouds(self, tag_points: dict, step: int):
        import open3d as o3d
        import wandb

        objects = {}
        for name, pts in tag_points.items():
            if pts is None or pts.shape[0] == 0:
                continue
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=5.0)
            pcd = pcd.select_by_index(ind)
            pcd.estimate_normals()
            objects[name] = wandb.Object3D(np.asarray(pcd.points))

        if objects and isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(objects, step=step)

    def training_step(self, batch, batch_idx):
        p_full, p_part = batch
        batch_size = p_full.shape[0]

        output = self(p_part)
        points = output.points
        loss = get_chamfer_distance()(points, p_full)

        if batch_idx == 0:
            self._log_point_clouds({
                "train/p_part": p_part[0].detach().cpu().float().numpy(),
                "train/p_full": p_full[0].detach().cpu().float().numpy(),
                "train/reconstructed": points[0].detach().cpu().float().numpy(),
            }, step=self.global_step)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        p_full, p_part = batch
        batch_size = p_full.shape[0]

        output = self(p_part)
        cd = get_chamfer_distance()(output.points, p_full)
        self.log("val_cd", cd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        if batch_idx == 0:
            self._log_point_clouds({
                "val/p_part": p_part[0].detach().cpu().float().numpy(),
                "val/p_full": p_full[0].detach().cpu().float().numpy(),
                "val/reconstructed": output.points[0].detach().cpu().float().numpy(),
            }, step=self.global_step)
        return cd

    def test_step(self, batch, batch_idx):
        p_full, p_part = batch
        batch_size = p_full.shape[0]
        output = self(p_part)
        cd = get_chamfer_distance()(output.points, p_full)
        self.log("test_cd", cd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return cd

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def _make_dataloader(self, split, shuffle):
        dataset = SemanticKITTI(self.config_path, split=split, mode="coarse")
        overfit = self.trainer.overfit_batches > 0
        num_workers = 0 if overfit else self.num_workers
        prefetch = 2 if num_workers > 0 else None
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False if overfit else shuffle,
            num_workers=num_workers, pin_memory=not overfit,
            persistent_workers=num_workers > 0, prefetch_factor=prefetch,
        )

    def train_dataloader(self):
        return self._make_dataloader("train", shuffle=True)

    def val_dataloader(self):
        split = "train" if self.trainer.overfit_batches > 0 else "validation"
        return self._make_dataloader(split, shuffle=True)

    def test_dataloader(self):
        return self._make_dataloader("validation", shuffle=False)
