from __future__ import annotations

from typing import Dict, Tuple

import timm
import torch
import torch.nn.functional as F
from torch import nn


def _infer_feature_dim(encoder: nn.Module, image_size: int = 224) -> int:
    with torch.no_grad():
        x = torch.zeros(1, 3, image_size, image_size)
        y = encoder(x)
        if y.dim() > 2:
            y = y.flatten(1)
    return int(y.shape[1])


class GlobalSemanticBranch(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_clip_224.openai",
        pretrained: bool = True,
        trainable_layers: int = 0,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        try:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            self.backbone_name = backbone_name
        except Exception:
            self.encoder = timm.create_model(
                "resnet18",
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            self.backbone_name = "resnet18"
        self.out_dim = _infer_feature_dim(self.encoder, image_size=image_size)
        self._set_trainable_layers(trainable_layers=trainable_layers)

    def _set_trainable_layers(self, trainable_layers: int) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        if trainable_layers <= 0:
            return
        module_list = list(self.encoder.children())
        for m in module_list[-trainable_layers:]:
            for p in m.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x)
        if y.dim() > 2:
            y = y.flatten(1)
        return y


class FrequencyBranch(nn.Module):
    def __init__(self, out_dim: int = 256, use_global_pool: bool = True) -> None:
        super().__init__()
        self.use_global_pool = use_global_pool
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if use_global_pool else nn.Identity()
        self.proj = nn.Linear(128, out_dim) if use_global_pool else nn.Conv2d(128, out_dim, kernel_size=1)
        self.out_dim = out_dim

    def _log_fft(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray, norm="ortho")
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        amp = torch.log1p(torch.abs(fft))
        amp = amp / (amp.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return amp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self._log_fft(x)
        z = self.encoder(f)
        z = self.pool(z)
        if self.use_global_pool:
            return self.proj(z.flatten(1))
        return self.proj(z)


class NoiseArtifactBranch(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        kernels = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("srm_kernels", kernels)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)
        self.out_dim = out_dim

    def _srm_residual(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        y = F.conv2d(gray, self.srm_kernels, stride=1, padding=1)
        y = torch.clamp(y, min=-3.0, max=3.0)
        y = (y + 3.0) / 6.0
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self._srm_residual(x)
        z = self.encoder(residual).flatten(1)
        return self.proj(z)


class GatedFusion(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int, int],
        fused_dim: int = 512,
        dropout: float = 0.2,
        gate_input_dropout: float = 0.1,
        feature_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d0, d1, d2 = in_dims
        self.rgb_proj = nn.Linear(d0, fused_dim)
        self.freq_proj = nn.Linear(d1, fused_dim)
        self.noise_proj = nn.Linear(d2, fused_dim)
        self.gate_input_dropout = nn.Dropout(gate_input_dropout)
        self.rgb_feature_dropout = nn.Dropout(feature_dropout)
        self.freq_feature_dropout = nn.Dropout(feature_dropout)
        self.noise_feature_dropout = nn.Dropout(feature_dropout)
        self.gate = nn.Sequential(
            nn.Linear(d0 + d1 + d2, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 3),
        )
        self.norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        rgb_feat: torch.Tensor,
        freq_feat: torch.Tensor,
        noise_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_in = self.rgb_feature_dropout(rgb_feat)
        freq_in = self.freq_feature_dropout(freq_feat)
        noise_in = self.noise_feature_dropout(noise_feat)
        fusion_input = self.gate_input_dropout(torch.cat([rgb_in, freq_in, noise_in], dim=1))
        weights = torch.softmax(self.gate(fusion_input), dim=1)
        rgb_z = self.rgb_proj(rgb_feat)
        freq_z = self.freq_proj(freq_feat)
        noise_z = self.noise_proj(noise_feat)
        fused = (
            weights[:, 0:1] * rgb_z
            + weights[:, 1:2] * freq_z
            + weights[:, 2:3] * noise_z
        )
        return self.norm(fused), weights


class HybridAIGCDetector(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_clip_224.openai",
        pretrained_backbone: bool = True,
        backbone_trainable_layers: int = 0,
        image_size: int = 224,
        fused_dim: int = 512,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
        use_aux_heads: bool = True,
        frequency_use_global_pool: bool = True,
        fusion_gate_input_dropout: float = 0.1,
        fusion_feature_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.semantic_branch = GlobalSemanticBranch(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            trainable_layers=backbone_trainable_layers,
            image_size=image_size,
        )
        self.frequency_branch = FrequencyBranch(out_dim=256, use_global_pool=frequency_use_global_pool)
        self.noise_branch = NoiseArtifactBranch(out_dim=256)
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        self.fusion = GatedFusion(
            in_dims=(self.semantic_branch.out_dim, self.frequency_branch.out_dim, self.noise_branch.out_dim),
            fused_dim=fused_dim,
            dropout=dropout,
            gate_input_dropout=fusion_gate_input_dropout,
            feature_dropout=fusion_feature_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )
        self.use_aux_heads = use_aux_heads
        if use_aux_heads:
            self.semantic_head = nn.Linear(self.semantic_branch.out_dim, 1)
            self.frequency_head = nn.Linear(self.frequency_branch.out_dim, 1)
            self.noise_head = nn.Linear(self.noise_branch.out_dim, 1)

    def _denormalize_for_artifacts(self, x_norm: torch.Tensor) -> torch.Tensor:
        # Denormalize to [0, 1] then scale to [0, 255] for SRM/FFT stability
        x = x_norm * self.input_std + self.input_mean
        return torch.clamp(x * 255.0, min=0.0, max=255.0)

    @staticmethod
    def _to_feature_vector(feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 4:
            return F.adaptive_avg_pool2d(feat, output_size=(1, 1)).flatten(1)
        return feat

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_norm = x
        x_art = self._denormalize_for_artifacts(x_norm)
        semantic_feat = self.semantic_branch(x_norm)
        freq_feat_raw = self.frequency_branch(x_art)
        noise_feat_raw = self.noise_branch(x_art)
        freq_feat = self._to_feature_vector(freq_feat_raw)
        noise_feat = self._to_feature_vector(noise_feat_raw)
        fused_feat, fusion_weights = self.fusion(semantic_feat, freq_feat, noise_feat)
        logit = self.classifier(fused_feat)
        out: Dict[str, torch.Tensor] = {
            "logit": logit,
            "probability": torch.sigmoid(logit),
            "semantic_feat": semantic_feat,
            "freq_feat": freq_feat,
            "noise_feat": noise_feat,
            "freq_feat_raw": freq_feat_raw,
            "noise_feat_raw": noise_feat_raw,
            "fused_feat": fused_feat,
            "fusion_weights": fusion_weights,
        }
        if self.use_aux_heads:
            out["semantic_logit"] = self.semantic_head(semantic_feat)
            out["freq_logit"] = self.frequency_head(freq_feat)
            out["noise_logit"] = self.noise_head(noise_feat)
        return out
