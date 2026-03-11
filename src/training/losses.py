import torch
import torch.nn.functional as F
from torch import nn


class DetectionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float().view(-1, 1)
        return self.criterion(logits, labels)


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

