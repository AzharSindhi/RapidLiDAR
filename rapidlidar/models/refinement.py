import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from rapidlidar.data.dataset import SemanticKITTI
from rapidlidar.losses.chamfer import sqrt_chamfer_distance
from rapidlidar.models.feature_extraction import project_to_bev
from rapidlidar.models.rapidlidar import RapidLiDAR

pl.seed_everything(42)


class RefineFFN(nn.Module):
    def __init__(self, embed_dims, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dims // 2, embed_dims),
        )
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, attn_out):
        x = self.norm1(attn_out)
        return self.norm2(x + self.ffn(x))


class RefinementNetwork(pl.LightningModule):
    def __init__(self, base_model_checkpoint, config_path="configs/rapidlidar_refine.yaml",
                 learning_rate=1e-5, batch_size=2,
                 num_stages=4, kappa=6,
                 x_range=None, y_range=None, z_range=None,
                 attn_chunk_size=131072, use_grad_checkpoint=True,
                 fast_exp=False, fast_exp_factor=10,
                 completion_indices_file=None):
        super().__init__()
        self.save_hyperparameters()

        try:
            from mmcv.ops import MultiScaleDeformableAttention
        except ImportError as exc:
            raise ImportError("mmcv is required for RefinementNetwork") from exc

        self.base_model = self._load_base_model(base_model_checkpoint)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        self.config_path = config_path
        with open(config_path, "r") as f:
            import yaml
            cfg = yaml.safe_load(f)["data"]

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.fast_exp = fast_exp
        self.fast_exp_factor = fast_exp_factor if fast_exp else None
        self.num_workers = 8 if fast_exp else 4
        self.completion_indices_file = completion_indices_file

        self.x_range = x_range or cfg["x_range"]
        self.y_range = y_range or cfg["y_range"]
        self.z_range = z_range or cfg["z_range"]

        self.latent_dim = self.base_model.latent_dim
        self.num_levels = len(self.base_model.feature_extractor.feature_dims)
        self.num_stages = num_stages
        self.kappa = kappa

        self.attn_chunk_size = attn_chunk_size
        self.use_grad_checkpoint = use_grad_checkpoint

        self.deform_attn_layers = nn.ModuleList()
        self.refine_ffns = nn.ModuleList()
        for _ in range(num_stages):
            attn = MultiScaleDeformableAttention(
                embed_dims=self.latent_dim, num_heads=8, num_levels=self.num_levels,
                num_points=4, im2col_step=64, dropout=0.0, batch_first=True,
            )
            attn.init_weights()
            self.deform_attn_layers.append(attn)
            self.refine_ffns.append(RefineFFN(self.latent_dim))

        self.residual_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 3 * kappa),
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    def _load_base_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams")
        if not hparams:
            raise RuntimeError(
                f"Base model checkpoint '{ckpt_path}' has no saved hyper_parameters; "
                "refinement requires the exact training configuration of the frozen base model."
            )
        model = RapidLiDAR(**dict(hparams))
        model.load_state_dict(ckpt["state_dict"])
        return model

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def _deformable_cross_attention(self, bev_patches, query_features, query_points, deform_attn):
        B, N, _ = query_features.shape
        spatial_shapes, reference_points = [], []
        for patches, bev_map in bev_patches:
            patch_shape = (bev_map.shape[2], bev_map.shape[3])
            coords = project_to_bev(query_points, self.x_range, self.y_range, patch_shape)
            spatial_shapes.append(patch_shape)
            reference_points.append(coords)

        values = torch.cat([patches for patches, _ in bev_patches], dim=1)
        reference_points = torch.cat(reference_points, dim=1).view(B, N, -1, 2)
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=query_points.device)
        level_start_index = torch.cat([
            torch.tensor([0], device=query_points.device),
            torch.cumsum(spatial_shapes[:, 0] * spatial_shapes[:, 1], dim=0)[:-1],
        ])

        chunk = self.attn_chunk_size if self.attn_chunk_size and self.attn_chunk_size > 0 else N

        def _call(q, ref):
            return deform_attn(
                query=q, key=values, value=values, identity=q, query_pos=None,
                reference_points=ref, spatial_shapes=spatial_shapes, level_start_index=level_start_index,
            )

        def _run(q, ref):
            if self.use_grad_checkpoint and self.training and q.requires_grad:
                return torch.utils.checkpoint.checkpoint(_call, q, ref, use_reentrant=False)
            return _call(q, ref)

        if chunk >= N:
            return _run(query_features, reference_points)

        outputs = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            outputs.append(_run(query_features[:, start:end], reference_points[:, start:end]))
        return torch.cat(outputs, dim=1)

    def _apply_residual_mlp(self, point_features):
        B, N, _ = point_features.shape
        chunk = self.attn_chunk_size if self.attn_chunk_size and self.attn_chunk_size > 0 else N

        def _head(x):
            if self.use_grad_checkpoint and self.training and x.requires_grad:
                return torch.utils.checkpoint.checkpoint(self.residual_mlp, x, use_reentrant=False)
            return self.residual_mlp(x)

        if chunk >= N:
            return _head(point_features)

        outputs = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            outputs.append(_head(point_features[:, start:end]))
        return torch.cat(outputs, dim=1)

    def forward(self, x_partial: torch.Tensor):
        with torch.no_grad():
            base_output = self.base_model(x_partial, up_factor=10)
            coarse_points = base_output.points.detach()
            point_features = base_output.point_features.detach()
            bev_maps = [b.detach() for b in base_output.bev_maps]
            projected_bev_patches = [p.detach() for p in base_output.projected_bev_patches]

        bev_patches = list(zip(projected_bev_patches, bev_maps))
        for i in range(self.num_stages):
            attn_out = self._deformable_cross_attention(bev_patches, point_features, coarse_points, self.deform_attn_layers[i])
            point_features = self.refine_ffns[i](attn_out)

        B, M, _ = coarse_points.shape
        delta = self._apply_residual_mlp(point_features).view(B, M, self.kappa, 3)
        refined_points = (coarse_points.unsqueeze(2) + delta).view(B, M * self.kappa, 3)
        return refined_points, coarse_points

    def sqrt_cd(self, pred, gt):
        return sqrt_chamfer_distance(pred, gt)

    def training_step(self, batch, batch_idx):
        p_full, p_part = batch
        batch_size = p_full.shape[0]
        refined_points, _ = self(p_part)
        loss = self.sqrt_cd(refined_points, p_full)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        p_full, p_part = batch
        batch_size = p_full.shape[0]
        refined_points, _ = self(p_part)
        cd = self.sqrt_cd(refined_points, p_full)
        self.log("val_cd", cd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return cd

    def test_step(self, batch, batch_idx):
        p_full, p_part = batch
        batch_size = p_full.shape[0]
        refined_points, _ = self(p_part)
        cd = self.sqrt_cd(refined_points, p_full)
        self.log("test_cd", cd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return cd

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def _make_dataloader(self, split, shuffle, completion_indices_file=None):
        dataset = SemanticKITTI(self.config_path, split=split, mode="refine")
        overfit = self.trainer.overfit_batches > 0
        num_workers = 0 if overfit else self.num_workers
        prefetch = (4 if self.fast_exp else 2) if num_workers > 0 else None
        return DataLoader(
            dataset, batch_size=self.batch_size if split == "train" else 1,
            shuffle=False if overfit else shuffle, num_workers=num_workers,
            pin_memory=not overfit, persistent_workers=num_workers > 0, prefetch_factor=prefetch,
        )

    def train_dataloader(self):
        return self._make_dataloader("train", shuffle=True)

    def val_dataloader(self):
        split = "train" if self.trainer.overfit_batches > 0 else "validation"
        cif = None if self.trainer.overfit_batches > 0 else self.completion_indices_file
        return self._make_dataloader(split, shuffle=True, completion_indices_file=cif)

    def test_dataloader(self):
        return self._make_dataloader("validation", shuffle=False)
