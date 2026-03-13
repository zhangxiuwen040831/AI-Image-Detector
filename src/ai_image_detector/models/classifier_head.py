from typing import Optional

import torch
from torch import nn


class BinaryClassifierHead(nn.Module):
    """
    简单的二分类头：
    - 输入为特征向量 (batch, dim)
    - 输出为 logits (batch, 1)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers = []
        if hidden_dim is not None and hidden_dim > 0:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

