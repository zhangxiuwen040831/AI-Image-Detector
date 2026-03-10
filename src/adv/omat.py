from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from src.training.losses import DetectionLoss


@dataclass
class OMATConfig:
    """
    简化版流形上对抗训练（OMAT）配置：
    - num_steps: 特征空间对抗迭代步数
    - step_size: 每步更新步长
    - epsilon: 特征扰动半径（L2 范数约束）
    - adv_weight: 对抗损失权重
    """

    num_steps: int = 3
    step_size: float = 1e-1
    epsilon: float = 1.0
    adv_weight: float = 1.0


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.view(x.size(0), -1).norm(p=2, dim=1, keepdim=True) + eps).view(
        x.size(0), *([1] * (x.dim() - 1))
    )


def omat_adversarial_logits(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    cfg: OMATConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    在特征流形上生成对抗特征，并返回对应 logits。

    假设模型具有：
    - model.backbone.encode_image
    - model.head
    """
    criterion = DetectionLoss().to(device)

    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        feats = model.backbone.encode_image(images)  # type: ignore[attr-defined]

    delta = torch.zeros_like(feats, device=device, requires_grad=True)

    for _ in range(cfg.num_steps):
        adv_feats = feats + delta
        logits = model.head(adv_feats).view(-1)  # type: ignore[attr-defined]
        loss = criterion(logits=logits, labels=labels)["loss"]

        loss.backward()
        with torch.no_grad():
            grad = delta.grad
            if grad is None:
                break
            grad = _l2_normalize(grad)
            delta += cfg.step_size * grad

            # 投影到 L2 ball 内
            delta = torch.clamp(
                _l2_normalize(delta) * torch.clamp(
                    delta.view(delta.size(0), -1).norm(p=2, dim=1, keepdim=True),
                    max=cfg.epsilon,
                ).view(delta.size(0), *([1] * (delta.dim() - 1))),
                min=-cfg.epsilon,
                max=cfg.epsilon,
            )
        delta.grad.zero_()

    adv_feats = feats + delta.detach()
    logits_adv = model.head(adv_feats).view(-1)  # type: ignore[attr-defined]
    return logits_adv


def omat_loss_dict(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    cfg: OMATConfig,
) -> Dict[str, torch.Tensor]:
    """
    返回包含 clean / adv / total 的损失字典，便于训练循环使用。
    """
    criterion = DetectionLoss().to(device)

    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    logits = outputs["logits"]
    clean_loss_dict = criterion(logits=logits, labels=labels)

    adv_logits = omat_adversarial_logits(
        model=model,
        images=images,
        labels=labels,
        cfg=cfg,
        device=device,
    )
    adv_loss = criterion(logits=adv_logits, labels=labels)["loss"]

    total_loss = clean_loss_dict["loss"] + cfg.adv_weight * adv_loss

    return {
        "loss": total_loss,
        "loss_clean": clean_loss_dict["loss"].detach(),
        "loss_adv": adv_loss.detach(),
    }

