from dataclasses import dataclass
from typing import Optional
import os
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch import nn

try:
    import open_clip
except ImportError as e:  # pragma: no cover - 仅在未安装 open_clip 时触发
    open_clip = None  # type: ignore[assignment]


@dataclass
class CLIPBackboneConfig:
    model_name: str = "ViT-L-14"
    pretrained: str = "openai"
    device: str = "cpu"
    train_backbone: bool = False


class CLIPBackbone(nn.Module):
    """
    基于 open_clip 的 CLIP ViT 图像特征提取骨干。

    提供统一的 encode_image 接口，并暴露特征维度 embed_dim。
    """

    def __init__(self, cfg: CLIPBackboneConfig) -> None:
        super().__init__()
        if open_clip is None:
            raise RuntimeError(
                "open_clip 未安装，请先运行 `pip install open_clip_torch`。"
            )

        max_retries = 5
        model = None
        for attempt in range(max_retries):
            try:
                print(f'Attempt {attempt+1}/{max_retries} to load model...')
                model, _, _ = open_clip.create_model_and_transforms(
                    cfg.model_name,
                    pretrained=cfg.pretrained,
                )
                print('Model loaded successfully!')
                break
            except Exception as e:
                print(f'Error loading model: {e}')
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f'Waiting {wait_time} seconds before retry...')
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f'Failed to load model after {max_retries} attempts') from e
        
        self.model = model
        self.device = torch.device(cfg.device)
        self.to(self.device)

        if not cfg.train_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    @property
    def embed_dim(self) -> int:
        # 大多数 CLIP ViT 模型提供 visual.output_dim
        return getattr(self.model.visual, "output_dim", self.model.text_projection.shape[1])  # type: ignore[no-any-return]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.encode_image(x)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        feats = self.model.encode_image(x)  # type: ignore[arg-type]
        # 规范化特征有助于稳定训练
        return feats.float()


def build_clip_backbone(
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: str = "cpu",
    train_backbone: bool = False,
) -> CLIPBackbone:
    cfg = CLIPBackboneConfig(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        train_backbone=train_backbone,
    )
    return CLIPBackbone(cfg)

