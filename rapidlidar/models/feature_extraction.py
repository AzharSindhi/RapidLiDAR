import torch
import torch.nn as nn
import torch.nn.functional as F

from rapidlidar.models.backbones.voxel_unet3d import VoxelFeatureEncoder


def voxelize(points, x_range, y_range, z_range, voxel_size):
    B, N, _ = points.shape
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    W = max(1, int(round((x_max - x_min) / voxel_size)))
    H = max(1, int(round((y_max - y_min) / voxel_size)))
    D = max(1, int(round((z_max - z_min) / voxel_size)))

    voxels = torch.zeros((B, 1, D, H, W), device=points.device, dtype=torch.float32)

    xi = ((points[..., 0] - x_min) / voxel_size).long().clamp(0, W - 1)
    yi = ((points[..., 1] - y_min) / voxel_size).long().clamp(0, H - 1)
    zi = ((points[..., 2] - z_min) / voxel_size).long().clamp(0, D - 1)

    for b in range(B):
        voxels[b, 0, zi[b], yi[b], xi[b]] = 1.0
    return voxels


def sample_grid_features(grid, points, x_range, y_range, z_range):
    if grid.dim() == 5:
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range
        x_norm = ((points[..., 0] - x_min) / (x_max - x_min) * 2.0 - 1.0).clamp(-1, 1)
        y_norm = ((points[..., 1] - y_min) / (y_max - y_min) * 2.0 - 1.0).clamp(-1, 1)
        z_norm = ((points[..., 2] - z_min) / (z_max - z_min) * 2.0 - 1.0).clamp(-1, 1)
        grid_coords = torch.stack([x_norm, y_norm, z_norm], dim=-1).unsqueeze(2).unsqueeze(2)
        sampled = F.grid_sample(grid, grid_coords, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.squeeze(-1).squeeze(-1).transpose(1, 2)
    else:
        x_min, x_max = x_range
        y_min, y_max = y_range
        x_norm = ((points[..., 0] - x_min) / (x_max - x_min) * 2.0 - 1.0).clamp(-1, 1)
        y_norm = ((points[..., 1] - y_min) / (y_max - y_min) * 2.0 - 1.0).clamp(-1, 1)
        grid_coords = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(grid, grid_coords, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.squeeze(-1).transpose(1, 2)


def project_to_bev(points, x_range, y_range, spatial_shape):
    x_min, x_max = x_range
    y_min, y_max = y_range
    H, W = spatial_shape
    x_norm = (((points[..., 0] - x_min) / (x_max - x_min) * W).clamp(0, W) / W).clamp(0, 1)
    y_norm = (((points[..., 1] - y_min) / (y_max - y_min) * H).clamp(0, H) / H).clamp(0, 1)
    return torch.stack([x_norm, y_norm], dim=-1)


class BEVHead(nn.Module):
    def __init__(self, in_channels: int, depth: int, out_channels: int, num_heads: int = 4):
        super().__init__()
        cat_channels = in_channels * max(depth, 1)
        self.project = nn.Sequential(
            nn.Conv2d(cat_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )
        self.context_attn = nn.MultiheadAttention(out_channels, num_heads, batch_first=True, dropout=0.0)
        self.context_norm = nn.LayerNorm(out_channels)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxel_feature: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = voxel_feature.shape
        x = voxel_feature.reshape(B, C * D, H, W)
        x = self.project(x)
        out_channels = x.shape[1]
        tokens = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.context_attn(tokens, tokens, tokens)
        tokens = self.context_norm(tokens + attn_out)
        x = tokens.transpose(1, 2).reshape(B, out_channels, H, W)
        return x + self.refine(x)


class BEVProjection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, None, None)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce(x).squeeze(2)


class FeatureExtractionModule(nn.Module):
    def __init__(self, x_range, y_range, z_range, voxel_size=0.3,
                 f_maps=(32, 64, 128, 256), latent_dim=512, bev_head_heads=4):
        super().__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.voxel_size = voxel_size
        self.latent_dim = latent_dim

        self.voxel_encoder = VoxelFeatureEncoder(in_channels=1, f_maps=f_maps)
        self.feature_dims = list(f_maps) + [latent_dim]

        z_min, z_max = z_range
        depth_full = max(1, int(round((z_max - z_min) / voxel_size)))
        last_stride = 2 ** (len(f_maps) - 1)
        depth_last = max(1, depth_full // last_stride)
        self.bev_head = BEVHead(
            in_channels=f_maps[-1], depth=depth_last, out_channels=latent_dim, num_heads=bev_head_heads
        )
        self.bev_projections = nn.ModuleList([BEVProjection(dim) for dim in f_maps])

        self.feature_proj = nn.Linear(sum(self.feature_dims), latent_dim)

    def extract(self, points: torch.Tensor):
        x = voxelize(points, self.x_range, self.y_range, self.z_range, self.voxel_size)
        voxel_features = self.voxel_encoder(x)
        bev_dense = self.bev_head(voxel_features[-1])
        return voxel_features, bev_dense

    def project_voxels_to_bev(self, voxel_features):
        bev_multiscale = []
        for i, feat in enumerate(voxel_features):
            bev_multiscale.append(self.bev_projections[i](feat))
        return bev_multiscale

    def interpolate(self, grids, points: torch.Tensor) -> torch.Tensor:
        sampled = [sample_grid_features(grid, points, self.x_range, self.y_range, self.z_range) for grid in grids]
        return self.feature_proj(torch.cat(sampled, dim=-1))
