from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ai_image_detector.datasets.transforms import build_val_transforms
from ai_image_detector.models.detector_model import DetectorModel


@dataclass
class ExplainConfig:
    image_size: int = 224


class AttentionRollout:
    """
    基于 ViT 多层注意力的 attention rollout，实现简易 heatmap。
    """

    def __init__(self, model: DetectorModel) -> None:
        self.model = model
        self.attentions = []
        self.handles = []

        visual = self.model.backbone.model.visual  # type: ignore[attr-defined]
        blocks = getattr(visual, "transformer", visual).resblocks  # type: ignore[attr-defined]

        for block in blocks:
            handle = block.attn.register_forward_hook(self._hook_attn)  # type: ignore[attr-defined]
            self.handles.append(handle)

    def _hook_attn(self, module, input, output):
        # output: attn_out, attn_weights
        if isinstance(output, tuple) and len(output) == 2:
            attn = output[1]
        else:
            # 兼容 fallback：不返回注意力时，直接跳过
            return
            
        if attn is None:
            # 部分模型实现（如某些优化过的 ViT）可能不返回注意力权重
            return
            
        self.attentions.append(attn.detach())

    def clear(self) -> None:
        self.attentions = []

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    @torch.no_grad()
    def generate(self, images: torch.Tensor, image_size: int) -> np.ndarray:
        """
        对单张图像生成 heatmap，返回 [H, W] 0-1 numpy 数组。
        """
        self.clear()
        # 前向传播以收集注意力
        _ = self.model(images)

        if not self.attentions:
            # 无法获取注意力时，返回均匀图
            return np.ones((image_size, image_size), dtype=np.float32)

        # 假设注意力形状为 (B, heads, tokens, tokens)
        attn = torch.stack(self.attentions, dim=0)  # (L, B, H, T, T)
        attn = attn.mean(dim=2)  # 平均各个 head -> (L, B, T, T)

        # attention rollout（参考论文）
        eye = torch.eye(attn.size(-1), device=attn.device).unsqueeze(0).unsqueeze(0)
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True)

        rollout = attn[0]  # 取第一层开始
        for i in range(1, attn.size(0)):
            rollout = attn[i] @ rollout

        # CLS token 对其他 token 的注意力
        mask = rollout[0, 0, 1:]  # (T-1,)
        num_patches = mask.size(0)
        grid_size = int(num_patches ** 0.5)
        mask = mask.reshape(1, 1, grid_size, grid_size)
        mask = F.interpolate(
            mask, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        mask = mask.squeeze().cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask.astype(np.float32)


def explain_image(
    model: DetectorModel,
    pil_img: Image.Image,
    cfg: ExplainConfig,
) -> Dict[str, Any]:
    """
    对单张 PIL 图像生成：
    - 预测结果
    - heatmap (H, W) 数组
    - 简单文字解释
    """
    device = next(model.parameters()).device

    transform = build_val_transforms(cfg.image_size)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    rollout = AttentionRollout(model)
    with torch.no_grad():
        outputs = model(tensor)
        # 模型输出的是类别 1 (REAL) 的概率，因此 prob_fake = 1 - probs
        prob_real = float(outputs["probs"].item())
        prob_fake = 1.0 - prob_real
    heatmap = rollout.generate(tensor, cfg.image_size)
    rollout.remove()

    # 简单规则生成文字解释
    high_region = (heatmap > 0.7).mean()
    
    # 计算重心以确定关注区域
    y_indices, x_indices = np.where(heatmap > 0.5)
    if len(y_indices) > 0:
        center_y, center_x = np.mean(y_indices), np.mean(x_indices)
        h, w = heatmap.shape
        if center_y < h/3: v_pos = "顶部"
        elif center_y > 2*h/3: v_pos = "底部"
        else: v_pos = "中部"
        
        if center_x < w/3: h_pos = "左侧"
        elif center_x > 2*w/3: h_pos = "右侧"
        else: h_pos = "中间"
        
        position_desc = f"{v_pos}{h_pos}"
    else:
        position_desc = "分散"

    if prob_fake > 0.5:
        confidence_level = "极高" if prob_fake > 0.9 else "较高"
        if high_region > 0.3:
            reason = f"模型以{confidence_level}置信度在图像{position_desc}大面积区域检测到潜在的生成痕迹（如异常纹理或不自然的光影一致性）。"
        else:
            reason = f"模型以{confidence_level}置信度在图像{position_desc}局部区域捕捉到高频细节异常，这通常是 AI 生成图像的特征。"
    else:
        confidence_level = "极高" if prob_fake < 0.1 else "较高"
        if high_region > 0.3:
            reason = f"尽管模型关注了图像{position_desc}较大区域，但分析后认为其纹理特征符合自然摄影规律，判定为真实照片的可能性{confidence_level}。"
        else:
            reason = f"模型主要聚焦于{position_desc}的关键物体特征，未发现显著的合成伪影，因此判定为真实照片（置信度{confidence_level}）。"

    return {
        "prob_fake": prob_fake,
        "heatmap": heatmap,
        "explanation": reason,
    }

