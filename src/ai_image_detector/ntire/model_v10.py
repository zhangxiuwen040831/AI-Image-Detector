from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .model import FrequencyBranch, GlobalSemanticBranch, NoiseArtifactBranch


class PrimaryFusionSFV10(nn.Module):
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
        self.semantic_proj = nn.Linear(semantic_dim, fused_dim)
        self.frequency_proj = nn.Linear(frequency_dim, fused_dim)
        self.semantic_feature_dropout = nn.Dropout(feature_dropout)
        self.frequency_feature_dropout = nn.Dropout(feature_dropout)
        self.gate_input_dropout = nn.Dropout(gate_input_dropout)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        semantic_in = self.semantic_feature_dropout(semantic_feat)
        frequency_in = self.frequency_feature_dropout(frequency_feat)
        gate_input = self.gate_input_dropout(torch.cat([semantic_in, frequency_in], dim=1))
        sf_weights = torch.softmax(self.gate(gate_input), dim=1)
        fused = (
            sf_weights[:, 0:1] * self.semantic_proj(semantic_feat)
            + sf_weights[:, 1:2] * self.frequency_proj(frequency_feat)
        )
        return self.norm(fused), sf_weights


class TriBranchGatedFusion(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        frequency_dim: int,
        noise_dim: int,
        fused_dim: int = 512,
        dropout: float = 0.2,
        gate_input_dropout: float = 0.1,
        feature_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.semantic_proj = nn.Linear(semantic_dim, fused_dim)
        self.frequency_proj = nn.Linear(frequency_dim, fused_dim)
        self.noise_proj = nn.Linear(noise_dim, fused_dim)
        self.semantic_feature_dropout = nn.Dropout(feature_dropout)
        self.frequency_feature_dropout = nn.Dropout(feature_dropout)
        self.noise_feature_dropout = nn.Dropout(feature_dropout)
        self.gate_input_dropout = nn.Dropout(gate_input_dropout)
        self.gate = nn.Sequential(
            nn.Linear(semantic_dim + frequency_dim + noise_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 3),
        )
        self.norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        semantic_feat: torch.Tensor,
        frequency_feat: torch.Tensor,
        noise_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        semantic_in = self.semantic_feature_dropout(semantic_feat)
        frequency_in = self.frequency_feature_dropout(frequency_feat)
        noise_in = self.noise_feature_dropout(noise_feat)
        gate_input = self.gate_input_dropout(torch.cat([semantic_in, frequency_in, noise_in], dim=1))
        weights = torch.softmax(self.gate(gate_input), dim=1)
        fused = (
            weights[:, 0:1] * self.semantic_proj(semantic_feat)
            + weights[:, 1:2] * self.frequency_proj(frequency_feat)
            + weights[:, 2:3] * self.noise_proj(noise_feat)
        )
        return self.norm(fused), weights


class NoiseControllerV10(nn.Module):
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
        self.input_dim = semantic_dim + frequency_dim + 3
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
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


class V10CompetitionResetModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_clip_224.openai",
        pretrained_backbone: bool = True,
        semantic_trainable_layers: int = 0,
        image_size: int = 224,
        frequency_dim: int = 256,
        noise_dim: int = 256,
        fused_dim: int = 512,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
        fusion_gate_input_dropout: float = 0.1,
        fusion_feature_dropout: float = 0.1,
        alpha_max: float = 0.35,
        enable_noise_expert: bool = True,
    ) -> None:
        super().__init__()
        self.enable_noise_expert = bool(enable_noise_expert)
        self.inference_mode = "deploy_safe_tri_branch"
        self.use_trained_tri_fusion = True
        self.semantic_branch = GlobalSemanticBranch(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            trainable_layers=semantic_trainable_layers,
            image_size=image_size,
        )
        self.frequency_branch = FrequencyBranch(out_dim=frequency_dim, use_global_pool=True)
        self.noise_branch = NoiseArtifactBranch(out_dim=noise_dim)
        self.primary_fusion = PrimaryFusionSFV10(
            semantic_dim=self.semantic_branch.out_dim,
            frequency_dim=self.frequency_branch.out_dim,
            fused_dim=fused_dim,
            dropout=dropout,
            gate_input_dropout=fusion_gate_input_dropout,
            feature_dropout=fusion_feature_dropout,
        )
        self.base_classifier = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )
        self.tri_fusion = TriBranchGatedFusion(
            semantic_dim=self.semantic_branch.out_dim,
            frequency_dim=self.frequency_branch.out_dim,
            noise_dim=self.noise_branch.out_dim,
            fused_dim=fused_dim,
            dropout=dropout,
            gate_input_dropout=fusion_gate_input_dropout,
            feature_dropout=fusion_feature_dropout,
        )
        self.tri_classifier = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )
        self.semantic_head = nn.Linear(self.semantic_branch.out_dim, 1)
        self.frequency_head = nn.Linear(self.frequency_branch.out_dim, 1)
        self.register_buffer(
            "input_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )
        if self.enable_noise_expert:
            self.noise_proj = nn.Linear(self.noise_branch.out_dim, fused_dim)
            self.noise_delta_norm = nn.LayerNorm(fused_dim)
            self.noise_delta_head = nn.Sequential(
                nn.Linear(fused_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1),
            )
            self.noise_controller = NoiseControllerV10(
                semantic_dim=self.semantic_branch.out_dim,
                frequency_dim=self.frequency_branch.out_dim,
                hidden_dim=max(head_hidden_dim, 64),
                dropout=dropout,
                alpha_max=alpha_max,
            )
            self.noise_head = nn.Linear(self.noise_branch.out_dim, 1)
        else:
            self.noise_proj = None
            self.noise_delta_norm = None
            self.noise_delta_head = None
            self.noise_controller = None
            self.noise_head = None

    def set_inference_mode(self, mode: str) -> None:
        mode = str(mode)
        allowed = {"deploy_safe_tri_branch", "trained_tri_fusion", "tri_fusion", "base_only"}
        if self.enable_noise_expert:
            allowed.add("hybrid_optional")
        if mode not in allowed:
            raise ValueError(f"inference mode must be one of {sorted(allowed)}")
        if mode in {"base_only", "hybrid_optional"}:
            mode = "deploy_safe_tri_branch"
        if mode == "tri_fusion":
            mode = "trained_tri_fusion" if self.use_trained_tri_fusion else "deploy_safe_tri_branch"
        self.inference_mode = mode

    def set_tri_fusion_decision_enabled(self, enabled: bool) -> None:
        self.use_trained_tri_fusion = bool(enabled)
        self.set_inference_mode("trained_tri_fusion" if enabled else "deploy_safe_tri_branch")

    def set_semantic_trainable_layers(self, trainable_layers: int) -> None:
        self.semantic_branch.set_trainable_layers(int(trainable_layers))

    def _set_module_requires_grad(self, module_names: Tuple[str, ...], enabled: bool) -> None:
        for module_name in module_names:
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = bool(enabled)

    def _phase_summary(self, phase: str, semantic_trainable_layers: int) -> Dict[str, object]:
        trainable_names = [name for name, param in self.named_parameters() if param.requires_grad]
        return {
            "phase": str(phase),
            "inference_mode": self.inference_mode,
            "semantic_trainable_layers": int(semantic_trainable_layers),
            "trainable_param_count": int(sum(param.numel() for param in self.parameters() if param.requires_grad)),
            "total_param_count": int(sum(param.numel() for param in self.parameters())),
            "trainable_param_names_preview": trainable_names[:80],
        }

    def configure_phase(
        self,
        phase: str,
        semantic_trainable_layers: int = 0,
    ) -> Dict[str, object]:
        phase = str(phase)
        for _, param in self.named_parameters():
            param.requires_grad = False
        self.set_semantic_trainable_layers(int(semantic_trainable_layers))
        if phase == "phase1_warmup":
            self.set_inference_mode("trained_tri_fusion")
            self._set_module_requires_grad(
                ("frequency_branch", "noise_branch", "tri_fusion", "tri_classifier", "semantic_head", "frequency_head", "noise_head"),
                True,
            )
        elif phase in {"phase2_curriculum", "phase4_final_polish"}:
            self.set_inference_mode("trained_tri_fusion")
            self._set_module_requires_grad(
                ("frequency_branch", "noise_branch", "tri_fusion", "tri_classifier", "semantic_head", "frequency_head", "noise_head"),
                True,
            )
        elif phase == "phase3_competition":
            self.set_inference_mode("trained_tri_fusion")
            self._set_module_requires_grad(
                ("frequency_branch", "noise_branch", "tri_fusion", "tri_classifier", "semantic_head", "frequency_head", "noise_head"),
                True,
            )
            if self.enable_noise_expert:
                self._set_module_requires_grad(
                    ("noise_proj", "noise_delta_norm", "noise_delta_head", "noise_controller"),
                    True,
                )
        else:
            raise ValueError(f"Unsupported V10 phase: {phase}")
        return self._phase_summary(phase=phase, semantic_trainable_layers=semantic_trainable_layers)

    def count_parameters(self) -> Dict[str, int]:
        total = int(sum(param.numel() for param in self.parameters()))
        trainable = int(sum(param.numel() for param in self.parameters() if param.requires_grad))
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    def _denormalize_for_artifacts(self, x_norm: torch.Tensor) -> torch.Tensor:
        x = x_norm * self.input_std + self.input_mean
        return torch.clamp(x * 255.0, min=0.0, max=255.0)

    @staticmethod
    def _to_feature_vector(feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 4:
            return F.adaptive_avg_pool2d(feat, output_size=(1, 1)).flatten(1)
        return feat

    def _compose_fusion_weights(
        self,
        sf_weights: torch.Tensor,
        alpha_ratio: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                sf_weights[:, 0:1] * (1.0 - alpha_ratio),
                sf_weights[:, 1:2] * (1.0 - alpha_ratio),
                alpha_ratio,
            ],
            dim=1,
        )

    @staticmethod
    def _compose_decision_weights(sf_weights: torch.Tensor, noise_weight: float = 0.0) -> torch.Tensor:
        noise = sf_weights.new_full((sf_weights.shape[0], 1), float(noise_weight))
        scale = sf_weights.new_tensor(max(1.0 - float(noise_weight), 0.0))
        return torch.cat([sf_weights[:, 0:1] * scale, sf_weights[:, 1:2] * scale, noise], dim=1)

    @staticmethod
    def _compose_evidence_weights(
        semantic_logit: torch.Tensor,
        frequency_logit: torch.Tensor,
        noise_logit: torch.Tensor,
    ) -> torch.Tensor:
        branch_logits = torch.cat(
            [
                semantic_logit.view(-1, 1),
                frequency_logit.view(-1, 1),
                noise_logit.view(-1, 1),
            ],
            dim=1,
        )
        return torch.softmax(torch.abs(branch_logits), dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_norm = x
        x_art = self._denormalize_for_artifacts(x_norm)
        semantic_feat = self.semantic_branch(x_norm)
        frequency_feat = self._to_feature_vector(self.frequency_branch(x_art))
        noise_feat = self._to_feature_vector(self.noise_branch(x_art))
        semantic_logit = self.semantic_head(semantic_feat)
        frequency_logit = self.frequency_head(frequency_feat)
        noise_logit = self.noise_head(noise_feat) if self.noise_head is not None else semantic_logit.new_zeros(semantic_logit.shape)
        fused_feat, tri_weights = self.tri_fusion(semantic_feat, frequency_feat, noise_feat)
        tri_fusion_logit = self.tri_classifier(fused_feat)

        # Deprecated legacy path kept only so old checkpoints/API clients can still read debug fields.
        base_feat, sf_weights = self.primary_fusion(semantic_feat, frequency_feat)
        stable_sf_logit = self.base_classifier(base_feat)
        alpha = stable_sf_logit.new_zeros(stable_sf_logit.shape)
        alpha_raw = stable_sf_logit.new_zeros(stable_sf_logit.shape)
        alpha_ratio = stable_sf_logit.new_zeros(stable_sf_logit.shape)
        noise_delta_logit = stable_sf_logit.new_zeros(stable_sf_logit.shape)
        hybrid_logit = stable_sf_logit
        if self.enable_noise_expert:
            noise_delta_feat = self.noise_delta_norm(self.noise_proj(noise_feat))
            noise_delta_logit = self.noise_delta_head(noise_delta_feat)
            alpha, alpha_raw = self.noise_controller(
                semantic_feat=semantic_feat,
                frequency_feat=frequency_feat,
                semantic_logit=semantic_logit.view(-1),
                frequency_logit=frequency_logit.view(-1),
                base_logit=stable_sf_logit.view(-1),
            )
            alpha_ratio = alpha / max(float(self.noise_controller.alpha_max), 1e-6)
            hybrid_logit = stable_sf_logit + alpha * noise_delta_logit
        active_alpha = torch.zeros_like(alpha)
        active_alpha_ratio = torch.zeros_like(alpha_ratio)
        tri_fusion_enabled = bool(self.use_trained_tri_fusion and self.inference_mode == "trained_tri_fusion")
        final_logit = tri_fusion_logit if tri_fusion_enabled else stable_sf_logit
        decision_weights = tri_weights if tri_fusion_enabled else self._compose_decision_weights(sf_weights, noise_weight=0.0)
        evidence_weights = self._compose_evidence_weights(semantic_logit, frequency_logit, noise_logit)
        fusion_weights = evidence_weights
        active_mode = "trained_tri_fusion" if tri_fusion_enabled else "deploy_safe_tri_branch"
        return {
            "semantic_feat": semantic_feat,
            "frequency_feat": frequency_feat,
            "noise_feat": noise_feat,
            "fused_feat": fused_feat,
            "tri_fusion_feat": fused_feat,
            "base_feat": fused_feat,
            "legacy_base_feat": base_feat,
            "sf_weights": sf_weights,
            "sf_weights_raw": sf_weights,
            "fusion_weights": fusion_weights,
            "fusion_weights_raw": fusion_weights,
            "decision_weights": decision_weights,
            "evidence_weights": evidence_weights,
            "tri_fusion_weights": tri_weights,
            "semantic_logit": semantic_logit,
            "freq_logit": frequency_logit,
            "frequency_logit": frequency_logit,
            "noise_logit": noise_logit,
            "base_logit": final_logit,
            "decision_logit": final_logit,
            "stable_sf_logit": stable_sf_logit,
            "legacy_base_logit": stable_sf_logit,
            "noise_delta_logit": noise_delta_logit,
            "hybrid_logit": hybrid_logit,
            "logit": final_logit,
            "tri_fusion_logit": tri_fusion_logit,
            "fused_logit": tri_fusion_logit,
            "alpha": alpha,
            "alpha_raw": alpha_raw,
            "alpha_used": active_alpha,
            "alpha_ratio": alpha_ratio,
            "alpha_ratio_used": active_alpha_ratio,
            "noise_enabled_for_decision": bool(tri_fusion_enabled),
            "tri_fusion_enabled_for_decision": bool(tri_fusion_enabled),
            "active_inference_mode": active_mode,
        }


__all__ = [
    "NoiseControllerV10",
    "PrimaryFusionSFV10",
    "TriBranchGatedFusion",
    "V10CompetitionResetModel",
]
