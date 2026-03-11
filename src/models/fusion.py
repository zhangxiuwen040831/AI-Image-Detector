import torch
from torch import nn


class ConcatFusionMLP(nn.Module):
    def __init__(
        self,
        rgb_dim: int,
        noise_dim: int,
        freq_dim: int,
        fused_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = rgb_dim + noise_dim + freq_dim
        self.fused_dim = fused_dim
        self.rgb_proj = nn.Linear(rgb_dim, fused_dim)
        self.noise_proj = nn.Linear(noise_dim, fused_dim)
        self.freq_proj = nn.Linear(freq_dim, fused_dim)
        self.weight_predictor = nn.Sequential(
            nn.Linear(self.input_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 3),
        )

    def forward(
        self,
        rgb_feat: torch.Tensor,
        noise_feat: torch.Tensor,
        freq_feat: torch.Tensor,
    ) -> torch.Tensor:
        concat_feat = torch.cat([rgb_feat, noise_feat, freq_feat], dim=1)
        weights = torch.softmax(self.weight_predictor(concat_feat), dim=1)
        rgb_proj = self.rgb_proj(rgb_feat)
        noise_proj = self.noise_proj(noise_feat)
        freq_proj = self.freq_proj(freq_feat)
        stacked = torch.stack([rgb_proj, noise_proj, freq_proj], dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return fused
