import timm
import torch
from torch import nn


class FrequencyBranch(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="avg",
        )
        self.out_dim = self._infer_dim()

    def _infer_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.encoder(dummy)
        return int(feat.shape[1])

    @staticmethod
    def fft_log_magnitude(x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        log_mag = torch.log1p(torch.abs(fft_shift))
        mean = log_mag.mean(dim=(-2, -1), keepdim=True)
        std = log_mag.std(dim=(-2, -1), keepdim=True)
        log_mag = (log_mag - mean) / (std + 1e-6)
        return log_mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self.fft_log_magnitude(x)
        feat = self.encoder(freq)
        if feat.dim() != 2:
            feat = feat.flatten(1)
        return feat

    def extract_frequency_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.fft_log_magnitude(x)
