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
    
    def get_last_conv_layer(self):
        """获取最后一个卷积层，用于Grad-CAM"""
        if self.backbone_name == "resnet18":
            # ResNet18的最后一个卷积层是layer4
            return self.encoder.layer4
        elif self.backbone_name == "efficientnet_b0":
            # EfficientNet的最后一个卷积层
            return self.encoder.conv_head
        elif self.backbone_name == "convnext_tiny":
            # ConvNeXt的最后一个卷积层
            return self.encoder.stages[3]
        else:
            raise ValueError(f"Unsupported backbone for Grad-CAM: {self.backbone_name}")
    
    def forward_with_features(self, x: torch.Tensor):
        """前向传播并返回特征图和最终特征"""
        # 保存特征图
        feature_maps = []
        
        # 定义钩子函数
        def hook(module, input, output):
            feature_maps.append(output)
        
        # 获取最后一个卷积层
        last_conv = self.get_last_conv_layer()
        # 注册钩子
        handle = last_conv.register_forward_hook(hook)
        
        # 前向传播
        feat = self.encoder(x)
        if feat.dim() != 2:
            feat = feat.flatten(1)
        
        # 移除钩子
        handle.remove()
        
        return feat, feature_maps[0]

    def output_shape(self, batch_size: int) -> Tuple[int, int]:
        return batch_size, self.out_dim
