from typing import Tuple

import timm
import torch
from torch import nn


SUPPORTED_BACKBONES = {"convnext_tiny", "efficientnet_b0", "resnet18"}


class RGBSpatialBranch(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone_name = backbone_name
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.out_dim = self._infer_dim()

    def _infer_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.encoder(dummy)
        return int(feat.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        if feat.dim() != 2:
            feat = feat.flatten(1)
        return feat

    def output_shape(self, batch_size: int) -> Tuple[int, int]:
        return batch_size, self.out_dim
