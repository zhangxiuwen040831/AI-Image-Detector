from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OSDConfig:
    """
    正交子空间分解配置：
    - proj_dim: 子空间维度
    - lambda_orth: 正交性约束权重（用于 loss）
    """

    proj_dim: int = 128
    lambda_orth: float = 1e-3


class OrthogonalSubspaceProjector(nn.Module):
    """
    将特征投影到两个近似正交的子空间（real / fake）。
    训练时通过额外的正则项鼓励两个投影矩阵正交。
    """

    def __init__(self, feat_dim: int, cfg: OSDConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.proj_real = nn.Linear(feat_dim, cfg.proj_dim, bias=False)
        self.proj_fake = nn.Linear(feat_dim, cfg.proj_dim, bias=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        返回拼接后的特征 [real_proj, fake_proj]。
        """
        real_proj = self.proj_real(feats)
        fake_proj = self.proj_fake(feats)
        return torch.cat([real_proj, fake_proj], dim=-1)

    def orthogonality_loss(self) -> torch.Tensor:
        """
        计算两个子空间投影矩阵之间的正交性损失：
        L = || P_r^T P_f ||_F^2
        """
        pr = self.proj_real.weight  # (proj_dim, feat_dim)
        pf = self.proj_fake.weight  # (proj_dim, feat_dim)
        m = pr @ pf.T  # (proj_dim, proj_dim)
        return (m ** 2).sum()

