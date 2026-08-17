import torch
import torch.nn as nn


class DisplacementMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        last = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 0.0)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            last,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class AdaptiveInitializationModule(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.displacement_mlp = DisplacementMLP(feature_dim=feature_dim)

    @staticmethod
    def repeat_and_perturb(x_partial: torch.Tensor, num_points: int, init_noise_std: float) -> torch.Tensor:
        B, N, _ = x_partial.shape
        if num_points == 0:
            return x_partial.new_zeros(B, 0, 3)
        idx = torch.randint(0, N, (B, num_points), device=x_partial.device)
        repeated = torch.gather(x_partial, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        if init_noise_std > 0:
            repeated = repeated + torch.randn_like(repeated) * init_noise_std
        return repeated

    def forward(self, tilde_p: torch.Tensor, point_features: torch.Tensor, displacement_scale: float) -> torch.Tensor:
        logits = self.displacement_mlp(point_features)
        delta = torch.tanh(logits) * displacement_scale
        return tilde_p + delta
