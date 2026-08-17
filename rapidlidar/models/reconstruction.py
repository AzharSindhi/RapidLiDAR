import torch
import torch.nn as nn

from rapidlidar.models.feature_extraction import project_to_bev


class MultiScaleReconstructionModule(nn.Module):
    def __init__(self, feature_dims, latent_dim=512, num_layers=2, num_heads=8,
                 num_points=4, x_range=None, y_range=None):
        super().__init__()
        try:
            from mmcv.ops import MultiScaleDeformableAttention
        except ImportError as exc:
            raise ImportError(
                "mmcv is required for MultiScaleReconstructionModule (pip install mmcv with ops support)"
            ) from exc

        self.x_range = x_range
        self.y_range = y_range
        self.num_layers = num_layers
        self.num_levels = len(feature_dims)

        self.deform_attn_layers = nn.ModuleList()
        self.value_projs = nn.ModuleList()
        for _ in range(num_layers):
            attn = MultiScaleDeformableAttention(
                embed_dims=latent_dim,
                num_heads=num_heads,
                num_levels=self.num_levels,
                num_points=num_points,
                im2col_step=64,
                dropout=0.0,
                batch_first=True,
            )
            if hasattr(attn, "sampling_offsets"):
                nn.init.constant_(attn.sampling_offsets.weight, 0.0)
                nn.init.constant_(attn.sampling_offsets.bias, 0.0)
            self.deform_attn_layers.append(attn)
            self.value_projs.append(nn.ModuleList([nn.Linear(dim, latent_dim) for dim in feature_dims]))

        self.residual_mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def deformable_cross_attention(self, bev_maps, query_features, query_points, value_projs, deform_attn):
        B, N, C = query_features.shape
        values, spatial_shapes, reference_points, projected_bev_patches = [], [], [], []
        for i, bev_map in enumerate(bev_maps):
            patch_shape = (bev_map.shape[2], bev_map.shape[3])
            patches = bev_map.flatten(2).transpose(1, 2)
            patches = value_projs[i](patches)
            coords = project_to_bev(query_points, self.x_range, self.y_range, patch_shape)
            values.append(patches)
            spatial_shapes.append(patch_shape)
            reference_points.append(coords)
            projected_bev_patches.append(patches)

        values = torch.cat(values, dim=1)
        reference_points = torch.cat(reference_points, dim=1).view(B, N, -1, 2)
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=query_points.device)
        level_start_index = torch.cat([
            torch.tensor([0], device=query_points.device),
            torch.cumsum(spatial_shapes[:, 0] * spatial_shapes[:, 1], dim=0)[:-1],
        ])

        output = deform_attn(
            query=query_features,
            key=values,
            value=values,
            identity=query_features,
            query_pos=None,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        return output, projected_bev_patches

    def forward(self, p_init: torch.Tensor, point_features: torch.Tensor, bev_maps):
        projected_bev_patches = None
        for i in range(self.num_layers):
            point_features, projected_bev_patches = self.deformable_cross_attention(
                bev_maps, point_features, p_init, self.value_projs[i], self.deform_attn_layers[i]
            )
        delta = self.residual_mlp(point_features)
        return p_init + delta, point_features, projected_bev_patches
