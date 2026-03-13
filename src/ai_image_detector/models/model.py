import torch
from torch import nn

from .detector import MultiBranchDetector


class AIGCImageDetector(nn.Module):
    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        self.detector = MultiBranchDetector(
            rgb_backbone=model_cfg.get("backbone", "resnet18"),
            rgb_pretrained=bool(model_cfg.get("rgb_pretrained", True)),
            noise_pretrained=bool(model_cfg.get("noise_pretrained", False)),
            freq_pretrained=bool(model_cfg.get("freq_pretrained", False)),
            fused_dim=int(model_cfg.get("fused_dim", 512)),
            classifier_hidden_dim=int(model_cfg.get("classifier_hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )

    def forward(self, x: torch.Tensor):
        return self.detector(x)
