from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .clip_backbone import CLIPBackbone, build_clip_backbone
from .classifier_head import BinaryClassifierHead
from .lora import LoRAConfig, inject_lora
from .osd import OSDConfig, OrthogonalSubspaceProjector


@dataclass
class DetectorModelConfig:
    backbone_name: str = "ViT-L-14"
    backbone_pretrained: str = "openai"
    device: str = "cpu"
    train_backbone: bool = False
    head_hidden_dim: Optional[int] = None
    head_dropout: float = 0.0
    # LoRA
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    # OSD
    use_osd: bool = False
    osd_proj_dim: int = 128
    osd_lambda_orth: float = 1e-3


class DetectorModel(nn.Module):
    """
    CLIP 图像编码器 + 二分类头。
    输出：
    - logits: (batch, 1)
    - probs:  (batch,) 概率（预测为 fake 的概率）
    """

    def __init__(self, cfg: DetectorModelConfig) -> None:
        super().__init__()

        self.backbone: CLIPBackbone = build_clip_backbone(
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

        feat_dim = self.backbone.embed_dim

        # 可选：正交子空间分解，在特征前增加 projector
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        feats = self.backbone.encode_image(x)
        if self.osd_projector is not None:
            feats = self.osd_projector(feats)
        logits = self.head(feats)
        probs = torch.sigmoid(logits).view(-1)
        return {"logits": logits, "probs": probs, "features": feats}


def build_detector_from_config(
    device: str = "cpu",
    train_backbone: bool = False,
    head_hidden_dim: Optional[int] = None,
    head_dropout: float = 0.0,
    backbone_name: str = "ViT-L-14",
    backbone_pretrained: str = "openai",
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    use_osd: bool = False,
    osd_proj_dim: int = 128,
    osd_lambda_orth: float = 1e-3,
) -> DetectorModel:
    cfg = DetectorModelConfig(
        backbone_name=backbone_name,
        backbone_pretrained=backbone_pretrained,
        device=device,
        train_backbone=train_backbone,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_osd=use_osd,
        osd_proj_dim=osd_proj_dim,
        osd_lambda_orth=osd_lambda_orth,
    )
    return DetectorModel(cfg)

