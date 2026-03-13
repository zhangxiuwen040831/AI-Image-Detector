import timm
import torch
from torch import nn


class SRMConv2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=30,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        kernels = self._build_srm30_kernels()
        with torch.no_grad():
            self.conv.weight.copy_(kernels)
        self.conv.weight.requires_grad_(False)

    @staticmethod
    def _build_srm30_kernels() -> torch.Tensor:
        base_1 = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ) / 4.0
        base_2 = torch.tensor(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=torch.float32,
        ) / 12.0
        base_3 = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ) / 2.0
        bases = [base_1, base_2, base_3]
        kernels_2d = []
        for base in bases:
            transforms = [
                base,
                torch.rot90(base, 1, dims=(0, 1)),
                torch.rot90(base, 2, dims=(0, 1)),
                torch.rot90(base, 3, dims=(0, 1)),
                torch.flip(base, dims=(0,)),
                torch.flip(base, dims=(1,)),
                torch.flip(torch.rot90(base, 1, dims=(0, 1)), dims=(0,)),
                torch.flip(torch.rot90(base, 1, dims=(0, 1)), dims=(1,)),
                torch.roll(base, shifts=1, dims=0),
                torch.roll(base, shifts=-1, dims=1),
            ]
            kernels_2d.extend(transforms)

        kernels = torch.stack(kernels_2d[:30], dim=0).unsqueeze(1)
        kernels = kernels.repeat(1, 3, 1, 1) / 3.0
        return kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv(x)
        residual = torch.clamp(residual, min=-3.0, max=3.0)
        return residual


class NoiseResidualBranch(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.srm = SRMConv2d()
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            in_chans=30,
            num_classes=0,
            global_pool="avg",
        )
        if hasattr(self.encoder, "maxpool"):
            self.encoder.maxpool = nn.Identity()

        self.out_dim = self._infer_dim()

    def _infer_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 30, 224, 224)
            feat = self.encoder(dummy)
        return int(feat.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.srm(x)
        feat = self.encoder(residual)
        if feat.dim() != 2:
            feat = feat.flatten(1)
        return feat

    def extract_residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.srm(x)
