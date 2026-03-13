from __future__ import annotations

import torch
from torch import nn


class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor([init_temperature], dtype=torch.float32))
        )

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(min=1e-3, max=100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    @torch.no_grad()
    def value(self) -> float:
        return float(self.temperature.item())

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
    ) -> float:
        logits = logits.detach().view(-1, 1).float()
        labels = labels.detach().view(-1, 1).float()
        if logits.numel() == 0 or labels.numel() == 0:
            return self.value()
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            return self.value()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            scaled = self.forward(logits)
            loss = criterion(scaled, labels)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
            if torch.isnan(self.log_temperature).any():
                raise RuntimeError("temperature became NaN after LBFGS")
        except Exception:
            best_temp = self._grid_search_temperature(logits=logits, labels=labels, criterion=criterion)
            with torch.no_grad():
                self.log_temperature.copy_(torch.log(torch.tensor([best_temp], device=self.log_temperature.device)))
        return self.value()

    @torch.no_grad()
    def _grid_search_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> float:
        candidates = torch.linspace(0.25, 5.0, steps=40, device=logits.device)
        best_temp = 1.0
        best_loss = float("inf")
        for temp in candidates:
            scaled = logits / temp.clamp(min=1e-3)
            loss = criterion(scaled, labels).item()
            if loss < best_loss:
                best_loss = loss
                best_temp = float(temp.item())
        return best_temp
