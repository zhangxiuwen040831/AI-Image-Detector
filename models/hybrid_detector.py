
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .clip_backbone import CLIPBackbone, build_clip_backbone
from .classifier_head import BinaryClassifierHead
from .detector_model import DetectorModelConfig
from .lora import LoRAConfig, inject_lora
from .osd import OSDConfig, OrthogonalSubspaceProjector

class FrequencyBranch(nn.Module):
    """
    频域特征提取分支
    输入：(B, C, H, W) 图像
    输出：(B, embed_dim) 频域特征
    """
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        # 使用简单的 CNN 处理 DCT 变换后的频谱图
        # 假设输入 224x224，DCT 后也是 224x224
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), # 112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算 DCT (离散余弦变换)
        # 这里使用 FFT 作为近似替代，取幅值谱
        # 实际 DCT 需要更复杂的实现，FFT 幅值也能反映频域特征
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_amp = torch.abs(fft)
        # 将低频移到中心
        fft_amp = torch.fft.fftshift(fft_amp, dim=(-2, -1))
        # 对数变换增强细节
        fft_amp = torch.log(fft_amp + 1e-6)
        
        features = self.conv_layers(fft_amp)
        features = features.flatten(1)
        return self.fc(features)

class SpatialBranch(nn.Module):
    """
    空域纹理特征提取分支 (轻量级 CNN)
    """
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        # 简单的 ResNet-like 结构或纹理提取网络
        # 关注局部纹理，使用较小的感受野
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.flatten(1)
        return self.fc(feat)

class HybridDetectorModel(nn.Module):
    """
    多模态/多分支检测模型
    1. Global Branch: CLIP ViT (Optional LoRA)
    2. Frequency Branch: DCT/FFT + CNN
    3. Spatial Branch: Texture CNN
    """
    def __init__(self, cfg: DetectorModelConfig):
        super().__init__()
        
        # 1. Global Branch (ViT)
        self.backbone = build_clip_backbone(
            model_name=cfg.backbone_name,
            pretrained=cfg.backbone_pretrained,
            device=cfg.device,
            train_backbone=cfg.train_backbone,
        )
        
        # 可选：在 backbone 中注入 LoRA
        self.lora_modules = None
        if cfg.use_lora:
            lora_cfg = LoRAConfig(
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
            )
            self.lora_modules = inject_lora(self.backbone.model.visual, lora_cfg)

        self.global_dim = self.backbone.embed_dim
        
        # 2. Frequency Branch
        self.freq_dim = 256
        self.freq_branch = FrequencyBranch(embed_dim=self.freq_dim)
        
        # 3. Spatial Branch
        self.spatial_dim = 256
        self.spatial_branch = SpatialBranch(embed_dim=self.spatial_dim)
        
        # Fusion
        total_dim = self.global_dim + self.freq_dim + self.spatial_dim
        
        # Attention Fusion or Concat
        # 这里使用简单的 Concat + MLP
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        feat_dim = 512
        
        # 可选：OSD (Applied after fusion)
        self.osd_projector: Optional[OrthogonalSubspaceProjector] = None
        if cfg.use_osd:
            osd_cfg = OSDConfig(
                proj_dim=cfg.osd_proj_dim,
                lambda_orth=cfg.osd_lambda_orth,
            )
            self.osd_projector = OrthogonalSubspaceProjector(feat_dim, osd_cfg)
            feat_dim = cfg.osd_proj_dim * 2

        self.head = BinaryClassifierHead(
            in_dim=feat_dim,
            hidden_dim=cfg.head_hidden_dim,
            dropout=cfg.head_dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Global features
        global_feat = self.backbone.encode_image(x)
        
        # Frequency features
        freq_feat = self.freq_branch(x)
        
        # Spatial features
        spatial_feat = self.spatial_branch(x)
        
        # Fusion
        concat_feat = torch.cat([global_feat, freq_feat, spatial_feat], dim=1)
        fused_feat = self.fusion_layer(concat_feat)
        
        # OSD Projection
        if self.osd_projector is not None:
            fused_feat = self.osd_projector(fused_feat)
        
        logits = self.head(fused_feat)
        probs = torch.sigmoid(logits).view(-1)
        
        return {
            "logits": logits, 
            "probs": probs, 
            "features": fused_feat,
            "global_feat": global_feat,
            "freq_feat": freq_feat,
            "spatial_feat": spatial_feat
        }

def build_hybrid_detector_from_config(cfg: DetectorModelConfig) -> HybridDetectorModel:
    if isinstance(cfg, dict):
        # Convert dict to config object if necessary, but caller usually passes dict.
        # Wait, build_detector_from_config takes arguments, not config object directly unless it constructs it.
        # But here I see HybridDetectorModel takes cfg: DetectorModelConfig.
        # My build_model in train_phase2.py passes a dict? No, it calls build_hybrid_detector_from_config(model_cfg).
        # But model_cfg is a dict.
        # So I should construct DetectorModelConfig here.
        pass
    
    # If cfg is dict, convert it.
    if isinstance(cfg, dict):
        cfg_obj = DetectorModelConfig(
            backbone_name=cfg.get("backbone_name", "ViT-L-14"),
            backbone_pretrained=cfg.get("backbone_pretrained", "openai"),
            device=cfg.get("device", "cpu"),
            train_backbone=cfg.get("train_backbone", False),
            head_hidden_dim=cfg.get("head_hidden_dim"),
            head_dropout=cfg.get("head_dropout", 0.0),
            use_lora=cfg.get("use_lora", False),
            lora_rank=cfg.get("lora_rank", 8),
            lora_alpha=cfg.get("lora_alpha", 16.0),
            lora_dropout=cfg.get("lora_dropout", 0.0),
            use_osd=cfg.get("use_osd", False),
            osd_proj_dim=cfg.get("osd_proj_dim", 128),
            osd_lambda_orth=cfg.get("osd_lambda_orth", 1e-3),
        )
        return HybridDetectorModel(cfg_obj)
    
    return HybridDetectorModel(cfg)
