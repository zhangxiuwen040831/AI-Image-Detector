import torch
from torch import nn


class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor([init_temperature], dtype=torch.float32)))

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
        if logits.numel() == 0 or labels.numel() == 0:
            return self.value()
        logits = logits.detach().float().view(-1, 1)
        labels = labels.detach().float().view(-1, 1)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)

        def closure():
            optimizer.zero_grad(set_to_none=True)
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.value()
