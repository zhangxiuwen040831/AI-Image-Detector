from __future__ import annotations

from typing import Dict, List, Tuple

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
        trainable_layers: int = -1,
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
            try:
                self.encoder = timm.create_model(
                    backbone_name,
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                )
                self.backbone_name = backbone_name
            except Exception:
                self.encoder = timm.create_model(
                    "resnet18",
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                )
                self.backbone_name = "resnet18"
        self.out_dim = _infer_feature_dim(self.encoder, image_size=image_size)
        self.trainable_layers = int(trainable_layers)
        self._set_trainable_layers(trainable_layers=trainable_layers)

    def _set_trainable_layers(self, trainable_layers: int) -> None:
        self.trainable_layers = int(trainable_layers)
        for p in self.encoder.parameters():
            p.requires_grad = False
        if trainable_layers == 0:
            return
        if trainable_layers < 0:
            for p in self.encoder.parameters():
                p.requires_grad = True
            return
        blocks_module = getattr(self.encoder, "blocks", None)
        if isinstance(blocks_module, (nn.ModuleList, nn.Sequential)):
            blocks = list(blocks_module)
            for block in blocks[-trainable_layers:]:
                for p in block.parameters():
                    p.requires_grad = True
            for norm_name in ("norm", "fc_norm", "head_norm"):
                norm_module = getattr(self.encoder, norm_name, None)
                if norm_module is None:
                    continue
                for p in norm_module.parameters():
                    p.requires_grad = True
            return
        if hasattr(self.encoder, "layer4"):
            residual_layers: List[nn.Module] = []
            for layer_name in ("layer4", "layer3", "layer2", "layer1"):
                layer_module = getattr(self.encoder, layer_name, None)
                if layer_module is not None:
                    residual_layers.append(layer_module)
            for module in residual_layers[:trainable_layers]:
                for p in module.parameters():
                    p.requires_grad = True
            return
        module_list = list(self.encoder.children())
        for m in module_list[-trainable_layers:]:
            for p in m.parameters():
                p.requires_grad = True

    def set_trainable_layers(self, trainable_layers: int) -> None:
        self._set_trainable_layers(trainable_layers=trainable_layers)

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
        max_semantic_weight: float = 0.45,
        min_semantic_weight: float = 0.0,
        min_noise_weight: float = 0.20,
        noise_branch_dropout_prob: float = 0.20,
        noise_branch_dropout_scale: float = 0.50,
        use_weight_regularization: bool = True,
        weight_reg_lambda: float = 0.01,
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
        
        # Weight constraints
        if not 0.0 <= max_semantic_weight <= 1.0:
            raise ValueError("max_semantic_weight must be in [0, 1]")
        if not 0.0 <= min_noise_weight <= 1.0:
            raise ValueError("min_noise_weight must be in [0, 1]")
        self.max_semantic_weight = max_semantic_weight
        self.min_semantic_weight = float(min_semantic_weight)
        self.min_noise_weight = min_noise_weight
        self.noise_branch_dropout_prob = max(0.0, min(float(noise_branch_dropout_prob), 1.0))
        self.noise_branch_dropout_scale = max(0.0, min(float(noise_branch_dropout_scale), 1.0))
        self.use_weight_regularization = use_weight_regularization
        self.weight_reg_lambda = weight_reg_lambda
        self._debug_printed = False

    def apply_noise_branch_dropout(self, weights: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.noise_branch_dropout_prob <= 0.0:
            return weights
        drop_mask = (
            torch.rand(weights.shape[0], 1, device=weights.device) < self.noise_branch_dropout_prob
        ).to(weights.dtype)
        if not torch.any(drop_mask > 0):
            return weights
        semantic = weights[:, 0:1]
        frequency = weights[:, 1:2]
        noise = weights[:, 2:3] * (1.0 - drop_mask + drop_mask * self.noise_branch_dropout_scale)
        total = (semantic + frequency + noise).clamp_min(1e-6)
        return torch.cat([semantic / total, frequency / total, noise / total], dim=1)

    def apply_semantic_floor(self, weights: torch.Tensor) -> torch.Tensor:
        # Deprecated in V7: semantic participation must be learned rather than
        # injected via a hard floor. Keep this helper as a no-op for compatibility.
        return weights

    def constrain_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply constraints to fusion weights."""
        # V7: semantic is not hard-clamped to a minimum. We only keep
        # semantic under the configured maximum and force a minimum noise share.
        w0 = torch.clamp(weights[:, 0:1], min=0.0, max=self.max_semantic_weight)
        w2 = torch.clamp(weights[:, 2:3], min=self.min_noise_weight, max=1.0)
        overflow = torch.clamp(w0 + w2 - 1.0, min=0.0)
        if torch.any(overflow > 0):
            w0 = torch.clamp(w0 - overflow, min=0.0, max=self.max_semantic_weight)
        w1 = torch.clamp(1.0 - w0 - w2, min=0.0)
        return torch.cat([w0, w1, w2], dim=1)

    @staticmethod
    def _branch_stats(weights: torch.Tensor) -> Dict[str, Tuple[float, float, float]]:
        names = ("semantic", "frequency", "noise")
        stats: Dict[str, Tuple[float, float, float]] = {}
        for idx, name in enumerate(names):
            column = weights[:, idx]
            stats[name] = (
                float(column.mean().item()),
                float(column.min().item()),
                float(column.max().item()),
            )
        return stats

    def _debug_print(self, tag: str, weights: torch.Tensor) -> None:
        stats = self._branch_stats(weights)
        print(
            f"[GatedFusion] {tag} semantic(mean/min/max)="
            f"{stats['semantic'][0]:.4f}/{stats['semantic'][1]:.4f}/{stats['semantic'][2]:.4f} "
            f"frequency={stats['frequency'][0]:.4f}/{stats['frequency'][1]:.4f}/{stats['frequency'][2]:.4f} "
            f"noise={stats['noise'][0]:.4f}/{stats['noise'][1]:.4f}/{stats['noise'][2]:.4f}"
        )

    def compute_weight_regularization(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute L2 regularization on weights to encourage balanced distribution."""
        if not self.use_weight_regularization:
            return torch.tensor(0.0, device=weights.device)
        
        # Target: more balanced distribution (0.33, 0.33, 0.33)
        target = torch.tensor([0.33, 0.33, 0.34], device=weights.device)
        reg = F.mse_loss(weights, target.unsqueeze(0).expand_as(weights))
        return self.weight_reg_lambda * reg

    def forward(
        self,
        rgb_feat: torch.Tensor,
        freq_feat: torch.Tensor,
        noise_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_in = self.rgb_feature_dropout(rgb_feat)
        freq_in = self.freq_feature_dropout(freq_feat)
        noise_in = self.noise_feature_dropout(noise_feat)
        fusion_input = self.gate_input_dropout(torch.cat([rgb_in, freq_in, noise_in], dim=1))
        weights_raw = torch.softmax(self.gate(fusion_input), dim=1)
        weights_dropped = self.apply_noise_branch_dropout(weights_raw)
        weights = self.constrain_weights(weights_dropped)
        
        # Debug print (first batch only)
        if self.training and (not self._debug_printed) and weights.shape[0] > 0:
            self._debug_print("Raw gate", weights_raw)
            self._debug_print("Noise-drop", weights_dropped)
            self._debug_print("Constrained fusion", weights)
            self._debug_printed = True
        
        # Compute regularization loss
        reg_loss = self.compute_weight_regularization(weights)
        
        rgb_z = self.rgb_proj(rgb_feat)
        freq_z = self.freq_proj(freq_feat)
        noise_z = self.noise_proj(noise_feat)
        fused = (
            weights[:, 0:1] * rgb_z
            + weights[:, 1:2] * freq_z
            + weights[:, 2:3] * noise_z
        )
        return self.norm(fused), weights, reg_loss, weights_raw


class PrimaryFusionSF(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        frequency_dim: int,
        fused_dim: int = 512,
        dropout: float = 0.2,
        gate_input_dropout: float = 0.1,
        feature_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rgb_proj = nn.Linear(semantic_dim, fused_dim)
        self.freq_proj = nn.Linear(frequency_dim, fused_dim)
        self.gate_input_dropout = nn.Dropout(gate_input_dropout)
        self.rgb_feature_dropout = nn.Dropout(feature_dropout)
        self.freq_feature_dropout = nn.Dropout(feature_dropout)
        self.gate = nn.Sequential(
            nn.Linear(semantic_dim + frequency_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 2),
        )
        self.norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        semantic_feat: torch.Tensor,
        frequency_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic_in = self.rgb_feature_dropout(semantic_feat)
        frequency_in = self.freq_feature_dropout(frequency_feat)
        gate_input = self.gate_input_dropout(torch.cat([semantic_in, frequency_in], dim=1))
        weights_raw = torch.softmax(self.gate(gate_input), dim=1)
        semantic_proj = self.rgb_proj(semantic_feat)
        frequency_proj = self.freq_proj(frequency_feat)
        fused = weights_raw[:, 0:1] * semantic_proj + weights_raw[:, 1:2] * frequency_proj
        return self.norm(fused), weights_raw, weights_raw


class NoiseController(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        frequency_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        alpha_max: float = 0.35,
    ) -> None:
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.controller_uses_noise = False
        self.controller_input_dim = semantic_dim + frequency_dim + 3
        self.net = nn.Sequential(
            nn.Linear(self.controller_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        semantic_feat: torch.Tensor,
        frequency_feat: torch.Tensor,
        semantic_logit: torch.Tensor,
        frequency_logit: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        controller_input = torch.cat(
            [
                semantic_feat,
                frequency_feat,
                semantic_logit.view(-1, 1),
                frequency_logit.view(-1, 1),
                base_logit.view(-1, 1),
            ],
            dim=1,
        )
        alpha_raw = self.net(controller_input)
        alpha = torch.sigmoid(alpha_raw) * self.alpha_max
        return alpha, alpha_raw


class HybridAIGCDetector(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_clip_224.openai",
        pretrained_backbone: bool = True,
        backbone_trainable_layers: int = -1,
        image_size: int = 224,
        fused_dim: int = 512,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
        use_aux_heads: bool = True,
        frequency_use_global_pool: bool = True,
        fusion_gate_input_dropout: float = 0.1,
        fusion_feature_dropout: float = 0.1,
        max_semantic_weight: float = 0.45,
        min_semantic_weight: float = 0.0,
        min_noise_weight: float = 0.20,
        noise_branch_dropout_prob: float = 0.20,
        noise_branch_dropout_scale: float = 0.50,
        use_weight_regularization: bool = True,
        weight_reg_lambda: float = 0.01,
        enable_base_residual_fusion: bool = False,
        v8_alpha_max: float = 0.35,
        v8_stage: str = "residual_finetune",
    ) -> None:
        super().__init__()
        self.enable_base_residual_fusion = bool(enable_base_residual_fusion)
        self.v8_alpha_max = float(v8_alpha_max)
        self.v8_stage = str(v8_stage)
        self.inference_mode = "hybrid" if self.enable_base_residual_fusion else "legacy"
        self.v81_controller_only = False
        self._v81_saved_requires_grad: Dict[str, bool] = {}
        self.v9_base_debias = False
        self._v9_saved_requires_grad: Dict[str, bool] = {}
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
        if self.enable_base_residual_fusion:
            self.fusion = PrimaryFusionSF(
                semantic_dim=self.semantic_branch.out_dim,
                frequency_dim=self.frequency_branch.out_dim,
                fused_dim=fused_dim,
                dropout=dropout,
                gate_input_dropout=fusion_gate_input_dropout,
                feature_dropout=fusion_feature_dropout,
            )
            self.noise_proj = nn.Linear(self.noise_branch.out_dim, fused_dim)
            self.noise_delta_norm = nn.LayerNorm(fused_dim)
            self.noise_controller = NoiseController(
                semantic_dim=self.semantic_branch.out_dim,
                frequency_dim=self.frequency_branch.out_dim,
                hidden_dim=max(head_hidden_dim, 64),
                dropout=dropout,
                alpha_max=self.v8_alpha_max,
            )
        else:
            self.fusion = GatedFusion(
                in_dims=(self.semantic_branch.out_dim, self.frequency_branch.out_dim, self.noise_branch.out_dim),
                fused_dim=fused_dim,
                dropout=dropout,
                gate_input_dropout=fusion_gate_input_dropout,
                feature_dropout=fusion_feature_dropout,
                max_semantic_weight=max_semantic_weight,
                min_semantic_weight=min_semantic_weight,
                min_noise_weight=min_noise_weight,
                noise_branch_dropout_prob=noise_branch_dropout_prob,
                noise_branch_dropout_scale=noise_branch_dropout_scale,
                use_weight_regularization=use_weight_regularization,
                weight_reg_lambda=weight_reg_lambda,
            )
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )
        if self.enable_base_residual_fusion:
            self.noise_delta_head = nn.Sequential(
                nn.Linear(fused_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1),
            )
        self.use_aux_heads = use_aux_heads
        if use_aux_heads:
            self.semantic_head = nn.Linear(self.semantic_branch.out_dim, 1)
            self.frequency_head = nn.Linear(self.frequency_branch.out_dim, 1)
            if self.enable_base_residual_fusion:
                self.noise_head = nn.Linear(self.noise_branch.out_dim, 1)
            else:
                self.noise_head = nn.Linear(self.noise_branch.out_dim, 1)

    def set_v8_stage(self, stage: str) -> None:
        self.v8_stage = str(stage)

    def set_inference_mode(self, mode: str) -> None:
        mode = str(mode)
        if self.enable_base_residual_fusion:
            if mode not in {"hybrid", "base_only"}:
                raise ValueError("inference_mode must be 'hybrid' or 'base_only'")
        else:
            if mode != "legacy":
                raise ValueError("legacy fusion model only supports inference_mode='legacy'")
        self.inference_mode = mode

    def get_v81_base_module_names(self) -> Tuple[str, ...]:
        return (
            "semantic_branch",
            "frequency_branch",
            "fusion",
            "classifier",
            "semantic_head",
            "frequency_head",
        )

    def get_v81_residual_module_names(self) -> Tuple[str, ...]:
        return (
            "noise_branch",
            "noise_proj",
            "noise_delta_norm",
            "noise_delta_head",
            "noise_controller",
        )

    def get_v9_noise_module_names(self) -> Tuple[str, ...]:
        return (
            "noise_branch",
            "noise_proj",
            "noise_delta_norm",
            "noise_delta_head",
            "noise_controller",
            "noise_head",
        )

    def get_v9_base_tunable_module_names(self) -> Tuple[str, ...]:
        return (
            "fusion",
            "classifier",
            "semantic_head",
            "frequency_head",
        )

    def configure_v81_controller_only(self, enabled: bool = True) -> Dict[str, object]:
        enabled = bool(enabled)
        frozen_modules = []
        trainable_modules = []
        if not self.enable_base_residual_fusion:
            return {
                "enabled": enabled,
                "frozen_modules": frozen_modules,
                "trainable_modules": trainable_modules,
            }

        if enabled and (not self.v81_controller_only):
            self._v81_saved_requires_grad = {
                name: bool(param.requires_grad)
                for name, param in self.named_parameters()
            }

        self.v81_controller_only = enabled
        if enabled:
            for _, param in self.named_parameters():
                param.requires_grad = False

            for module_name in self.get_v81_base_module_names():
                if getattr(self, module_name, None) is not None:
                    frozen_modules.append(module_name)

            extra_frozen_modules = ("noise_head",)
            for module_name in extra_frozen_modules:
                module = getattr(self, module_name, None)
                if module is None:
                    continue
                frozen_modules.append(module_name)
                for param in module.parameters():
                    param.requires_grad = False

            for module_name in self.get_v81_residual_module_names():
                module = getattr(self, module_name, None)
                if module is None:
                    continue
                trainable_modules.append(module_name)
                for param in module.parameters():
                    param.requires_grad = True

            self.apply_v81_training_mode()
        else:
            saved_states = self._v81_saved_requires_grad
            if saved_states:
                for name, param in self.named_parameters():
                    if name in saved_states:
                        param.requires_grad = bool(saved_states[name])
            self._v81_saved_requires_grad = {}
        return {
            "enabled": self.v81_controller_only,
            "frozen_modules": frozen_modules,
            "trainable_modules": trainable_modules,
        }

    def apply_v81_training_mode(self) -> None:
        if (not self.enable_base_residual_fusion) or (not self.v81_controller_only):
            return
        for module_name in self.get_v81_base_module_names():
            module = getattr(self, module_name, None)
            if module is not None:
                module.eval()
        for module_name in self.get_v81_residual_module_names():
            module = getattr(self, module_name, None)
            if module is not None:
                module.train()

    def configure_v9_base_debias(self, semantic_trainable_layers: int = 2) -> Dict[str, object]:
        if not self.enable_base_residual_fusion:
            raise RuntimeError("V9 base-debias requires base-residual fusion to be enabled.")
        if (not self.v9_base_debias) and (not self._v9_saved_requires_grad):
            self._v9_saved_requires_grad = {
                name: bool(param.requires_grad)
                for name, param in self.named_parameters()
            }
        self.v9_base_debias = True
        self.v81_controller_only = False
        self.set_inference_mode("base_only")
        for _, param in self.named_parameters():
            param.requires_grad = False

        self.semantic_branch.set_trainable_layers(int(semantic_trainable_layers))
        for module_name in self.get_v9_base_tunable_module_names():
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = True

        frozen_modules = []
        trainable_modules = []
        for module_name in self.get_v9_noise_module_names():
            module = getattr(self, module_name, None)
            if module is not None:
                frozen_modules.append(module_name)
        for module_name in ("frequency_branch",):
            module = getattr(self, module_name, None)
            if module is not None:
                frozen_modules.append(module_name)
        trainable_modules.append("semantic_branch_last_blocks")
        for module_name in self.get_v9_base_tunable_module_names():
            module = getattr(self, module_name, None)
            if module is not None:
                trainable_modules.append(module_name)

        self.apply_v9_training_mode()
        trainable_param_names = [
            name for name, param in self.named_parameters() if param.requires_grad
        ]
        return {
            "enabled": True,
            "semantic_trainable_layers": int(semantic_trainable_layers),
            "inference_mode": self.inference_mode,
            "frozen_modules": frozen_modules,
            "trainable_modules": trainable_modules,
            "trainable_param_count": int(sum(param.numel() for param in self.parameters() if param.requires_grad)),
            "trainable_param_names_preview": trainable_param_names[:50],
        }

    def apply_v9_training_mode(self) -> None:
        if (not self.enable_base_residual_fusion) or (not self.v9_base_debias):
            return
        self.semantic_branch.train()
        self.fusion.train()
        self.classifier.train()
        if getattr(self, "semantic_head", None) is not None:
            self.semantic_head.train()
        if getattr(self, "frequency_head", None) is not None:
            self.frequency_head.train()
        if getattr(self, "frequency_branch", None) is not None:
            self.frequency_branch.eval()
        for module_name in self.get_v9_noise_module_names():
            module = getattr(self, module_name, None)
            if module is not None:
                module.eval()

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
        out: Dict[str, torch.Tensor] = {
            "semantic_feat": semantic_feat,
            "freq_feat": freq_feat,
            "noise_feat": noise_feat,
            "freq_feat_raw": freq_feat_raw,
            "noise_feat_raw": noise_feat_raw,
        }
        if self.use_aux_heads:
            semantic_logit = self.semantic_head(semantic_feat)
            frequency_logit = self.frequency_head(freq_feat)
            out["semantic_logit"] = semantic_logit
            out["freq_logit"] = frequency_logit
            out["noise_logit"] = self.noise_head(noise_feat)
        else:
            semantic_logit = semantic_feat.new_zeros((semantic_feat.shape[0], 1))
            frequency_logit = semantic_feat.new_zeros((semantic_feat.shape[0], 1))

        if self.enable_base_residual_fusion:
            base_feat, sf_weights, sf_weights_raw = self.fusion(semantic_feat, freq_feat)
            base_logit = self.classifier(base_feat)
            noise_proj = self.noise_proj(noise_feat)
            noise_delta_feat = self.noise_delta_norm(noise_proj)
            noise_delta_logit = self.noise_delta_head(noise_delta_feat)
            alpha, alpha_raw = self.noise_controller(
                semantic_feat=semantic_feat,
                frequency_feat=freq_feat,
                semantic_logit=semantic_logit.view(-1),
                frequency_logit=frequency_logit.view(-1),
                base_logit=base_logit.view(-1),
            )
            if self.v8_stage == "debias_base":
                alpha_used = torch.zeros_like(alpha)
            else:
                alpha_used = alpha
            hybrid_logit = base_logit + alpha_used * noise_delta_logit
            if self.inference_mode == "base_only":
                final_logit = base_logit
                active_alpha = torch.zeros_like(alpha_used)
            else:
                final_logit = hybrid_logit
                active_alpha = alpha_used
            alpha_ratio_raw = (alpha / max(self.v8_alpha_max, 1e-6)).clamp(0.0, 1.0)
            alpha_ratio_used = (alpha_used / max(self.v8_alpha_max, 1e-6)).clamp(0.0, 1.0)
            alpha_ratio_active = (active_alpha / max(self.v8_alpha_max, 1e-6)).clamp(0.0, 1.0)
            contribution_weights_raw = torch.cat(
                [
                    (1.0 - alpha_ratio_raw) * sf_weights_raw[:, 0:1],
                    (1.0 - alpha_ratio_raw) * sf_weights_raw[:, 1:2],
                    alpha_ratio_raw,
                ],
                dim=1,
            )
            contribution_weights = torch.cat(
                [
                    (1.0 - alpha_ratio_used) * sf_weights[:, 0:1],
                    (1.0 - alpha_ratio_used) * sf_weights[:, 1:2],
                    alpha_ratio_used,
                ],
                dim=1,
            )
            contribution_weights_active = torch.cat(
                [
                    (1.0 - alpha_ratio_active) * sf_weights[:, 0:1],
                    (1.0 - alpha_ratio_active) * sf_weights[:, 1:2],
                    alpha_ratio_active,
                ],
                dim=1,
            )
            base_only_weights = torch.cat(
                [
                    sf_weights[:, 0:1],
                    sf_weights[:, 1:2],
                    torch.zeros_like(sf_weights[:, 0:1]),
                ],
                dim=1,
            )
            out.update(
                {
                    "logit": final_logit,
                    "final_logit": final_logit,
                    "probability": torch.sigmoid(final_logit),
                    "hybrid_logit": hybrid_logit,
                    "hybrid_probability": torch.sigmoid(hybrid_logit),
                    "base_only_logit": base_logit,
                    "base_only_probability": torch.sigmoid(base_logit),
                    "base_logit": base_logit,
                    "base_probability": torch.sigmoid(base_logit),
                    "noise_delta_logit": noise_delta_logit,
                    "noise_delta_probability": torch.sigmoid(noise_delta_logit),
                    "base_feat": base_feat,
                    "fused_feat": base_feat,
                    "noise_delta_feat": noise_delta_feat,
                    "sf_weights": sf_weights,
                    "sf_weights_raw": sf_weights_raw,
                    "alpha": alpha,
                    "alpha_used": alpha_used,
                    "alpha_raw": alpha_raw,
                    "alpha_active": active_alpha,
                    "alpha_ratio": alpha_ratio_raw,
                    "alpha_ratio_used": alpha_ratio_used,
                    "alpha_ratio_active": alpha_ratio_active,
                    "fusion_weights": contribution_weights_active,
                    "fusion_weights_hybrid": contribution_weights,
                    "fusion_weights_base_only": base_only_weights,
                    "fusion_weights_raw": contribution_weights_raw,
                    "fusion_reg_loss": final_logit.new_zeros(()),
                    "active_inference_mode": self.inference_mode,
                }
            )
            out["noise_logit"] = noise_delta_logit
            return out

        fused_feat, fusion_weights, fusion_reg_loss, fusion_weights_raw = self.fusion(
            semantic_feat,
            freq_feat,
            noise_feat,
        )
        logit = self.classifier(fused_feat)
        out.update(
            {
                "logit": logit,
                "probability": torch.sigmoid(logit),
                "fused_feat": fused_feat,
                "fusion_weights": fusion_weights,
                "fusion_weights_raw": fusion_weights_raw,
                "fusion_reg_loss": fusion_reg_loss,
            }
        )
        return out
