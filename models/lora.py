from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = ("attn", "to_q", "to_k", "to_v")


class LoRALinear(nn.Module):
    """
    在线性层上添加 LoRA 模块：y = Wx + scale * BAx
    """

    def __init__(self, base: nn.Linear, cfg: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.rank = cfg.rank
        self.scale = cfg.alpha / cfg.rank
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        in_dim = base.in_features
        out_dim = base.out_features

        self.lora_A = nn.Linear(in_dim, cfg.rank, bias=False)
        self.lora_B = nn.Linear(cfg.rank, out_dim, bias=False)

        # 初始化为接近零的扰动
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        # 冻结原始权重
        for p in self.base.parameters():
            p.requires_grad = False

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base_out + lora_out


def _iter_named_linear_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def inject_lora(
    model: nn.Module,
    cfg: LoRAConfig,
) -> List[str]:
    """
    在给定模型中，将名称包含 target_modules 关键字的 Linear 层替换为 LoRALinear。

    返回所有被注入 LoRA 的模块名称列表。
    """
    replaced: List[str] = []
    for name, module in list(_iter_named_linear_modules(model)):
        if not any(key in name for key in cfg.target_modules):
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model

        lora_layer = LoRALinear(module, cfg)
        setattr(parent, child_name, lora_layer)
        replaced.append(name)

    return replaced

