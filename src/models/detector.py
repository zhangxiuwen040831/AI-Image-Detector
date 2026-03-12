from typing import Dict

import torch
from torch import nn

from .freq_branch import FrequencyBranch
from .fusion import ConcatFusionMLP
from .noise_branch import NoiseResidualBranch
from .rgb_branch import RGBSpatialBranch


class MultiBranchDetector(nn.Module):
    def __init__(
        self,
        rgb_backbone: str = "resnet18",
        rgb_pretrained: bool = True,
        noise_pretrained: bool = False,
        freq_pretrained: bool = False,
        fused_dim: int = 512,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.rgb_branch = RGBSpatialBranch(
            backbone_name=rgb_backbone,
            pretrained=rgb_pretrained,
        )
        self.noise_branch = NoiseResidualBranch(pretrained=noise_pretrained)
        self.freq_branch = FrequencyBranch(pretrained=freq_pretrained)
        self.fusion_module = ConcatFusionMLP(
            rgb_dim=self.rgb_branch.out_dim,
            noise_dim=self.noise_branch.out_dim,
            freq_dim=self.freq_branch.out_dim,
            fused_dim=fused_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb_feat = self.rgb_branch(x)
        noise_feat = self.noise_branch(x)
        freq_feat = self.freq_branch(x)
        fused_feat = self.fusion_module(rgb_feat, noise_feat, freq_feat)
        logit = self.classifier(fused_feat)
        probability = torch.sigmoid(logit)
        return {
            "logit": logit,
            "probability": probability,
            "rgb_feat": rgb_feat,
            "noise_feat": noise_feat,
            "freq_feat": freq_feat,
            "fused_feat": fused_feat,
        }
    
    def forward_with_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播并返回特征图"""
        # 使用RGB分支的forward_with_features获取特征图
        rgb_feat, rgb_feature_map = self.rgb_branch.forward_with_features(x)
        noise_feat = self.noise_branch(x)
        freq_feat = self.freq_branch(x)
        fused_feat = self.fusion_module(rgb_feat, noise_feat, freq_feat)
        logit = self.classifier(fused_feat)
        probability = torch.sigmoid(logit)
        return {
            "logit": logit,
            "probability": probability,
            "rgb_feat": rgb_feat,
            "noise_feat": noise_feat,
            "freq_feat": freq_feat,
            "fused_feat": fused_feat,
            "rgb_feature_map": rgb_feature_map,
        }
