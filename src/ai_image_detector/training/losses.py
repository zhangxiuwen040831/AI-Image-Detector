import torch
import torch.nn.functional as F
from torch import nn


class DetectionLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.7,
        focal_weight: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        lambda_rgb: float = 0.2,
        lambda_freq: float = 0.2,
        lambda_spatial: float = 0.2,
    ) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.lambda_rgb = lambda_rgb
        self.lambda_freq = lambda_freq
        self.lambda_spatial = lambda_spatial

    def _main_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bce = self.criterion(logits, labels)
        focal = self.focal(logits, labels)
        return self.bce_weight * bce + self.focal_weight * focal

    def forward(self, logits, labels: torch.Tensor):
        labels = labels.float().view(-1, 1)
        if isinstance(logits, dict):
            main_loss = self._main_loss(logits["logit"], labels)
            rgb_loss = self.criterion(logits["rgb_logit"], labels)
            freq_loss = self.criterion(logits["freq_logit"], labels)
            spatial_loss = self.criterion(logits["spatial_logit"], labels)
            total_loss = (
                main_loss
                + self.lambda_rgb * rgb_loss
                + self.lambda_freq * freq_loss
                + self.lambda_spatial * spatial_loss
            )
            return {
                "total_loss": total_loss,
                "main_loss": main_loss,
                "rgb_loss": rgb_loss,
                "freq_loss": freq_loss,
                "spatial_loss": spatial_loss,
            }
        return self._main_loss(logits, labels)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float().view(-1, 1)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1.0 - pt).pow(self.gamma) * bce
        return focal.mean()
