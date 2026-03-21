from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ai_image_detector.training.ema import ModelEma

from .calibration import TemperatureScaler
from .losses import HybridDetectionLoss
from .metrics import compute_metrics


class NTIRETrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        save_dir: Path,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        warmup_epochs: int = 2,
        grad_clip: float = 1.0,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        bce_weight: float = 1.0,
        focal_weight: float = 0.0,
        aux_weight: float = 0.15,
        consistency_weight: float = 0.1,
        mixup_alpha: float = 1.0,
        mixup_prob: float = 0.0,
        max_checkpoint_keep: int = 999999,
        aux_schedule: str = "default",
        collapse_min_separation: float = 0.2,
        collapse_min_logit_std: float = 0.4,
        hybrid_score_sep_weight: float = 0.1,
        v8_enable_base_residual_fusion: bool = False,
        v8_stage: str = "residual_finetune",
        v8_base_loss_weight: float = 0.75,
        v8_semantic_head_weight: float = 0.25,
        v8_frequency_head_weight: float = 0.20,
        v8_noise_head_weight: float = 0.05,
        v8_real_alpha_loss_weight: float = 0.02,
        v8_real_noise_push_weight: float = 0.02,
        v8_real_counterfactual_weight: float = 0.02,
        v8_hard_aigc_alpha_weight: float = 0.02,
        v8_real_counterfactual_margin: float = 0.05,
        v8_hard_aigc_alpha_target: float = 0.15,
        v81_controller_only: bool = False,
        v81_alpha_real_weight: float = 0.05,
        v81_alpha_easy_aigc_weight: float = 0.03,
        v81_alpha_hard_aigc_weight: float = 0.02,
        v81_alpha_easy_aigc_threshold: float = 0.70,
        v81_alpha_hard_target: float = 0.50,
        v81_real_residual_push_weight: float = 0.05,
        v81_real_counterfactual_weight: float = 0.03,
        v81_real_counterfactual_margin: float = 0.05,
        v81_hard_aigc_residual_min_weight: float = 0.02,
        v81_easy_aigc_residual_suppress_weight: float = 0.02,
        v81_residual_push_target: float = 0.05,
        v9_phase: str = "none",
        v9_semantic_head_weight: float = 0.25,
        v9_frequency_head_weight: float = 0.20,
        v9_hard_real_margin_weight: float = 0.10,
        v9_hard_real_margin: float = 0.25,
        v9_prototype_weight: float = 0.05,
        v9_prototype_margin: float = 0.15,
    ) -> None:
        self.consistency_weight = consistency_weight
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.max_checkpoint_keep = max_checkpoint_keep
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = HybridDetectionLoss(bce_weight=bce_weight, focal_weight=focal_weight, aux_weight=aux_weight)
        self.base_aux_weight = aux_weight
        self.aux_schedule = aux_schedule
        self.mixup_alpha = mixup_alpha
        self.v81_controller_only = bool(v81_controller_only)
        self.mixup_prob = 0.0 if self.v81_controller_only else mixup_prob
        self.v8_enable_base_residual_fusion = bool(v8_enable_base_residual_fusion)
        self.v8_stage = str(v8_stage)
        self.v8_base_loss_weight = float(v8_base_loss_weight)
        self.v8_semantic_head_weight = float(v8_semantic_head_weight)
        self.v8_frequency_head_weight = float(v8_frequency_head_weight)
        self.v8_noise_head_weight = float(v8_noise_head_weight)
        self.v8_real_alpha_loss_weight = float(v8_real_alpha_loss_weight)
        self.v8_real_noise_push_weight = float(v8_real_noise_push_weight)
        self.v8_real_counterfactual_weight = float(v8_real_counterfactual_weight)
        self.v8_hard_aigc_alpha_weight = float(v8_hard_aigc_alpha_weight)
        self.v8_real_counterfactual_margin = float(v8_real_counterfactual_margin)
        self.v8_hard_aigc_alpha_target = float(v8_hard_aigc_alpha_target)
        self.v81_alpha_real_weight = float(v81_alpha_real_weight)
        self.v81_alpha_easy_aigc_weight = float(v81_alpha_easy_aigc_weight)
        self.v81_alpha_hard_aigc_weight = float(v81_alpha_hard_aigc_weight)
        self.v81_alpha_easy_aigc_threshold = float(v81_alpha_easy_aigc_threshold)
        self.v81_alpha_hard_target = float(v81_alpha_hard_target)
        self.v81_real_residual_push_weight = float(v81_real_residual_push_weight)
        self.v81_real_counterfactual_weight = float(v81_real_counterfactual_weight)
        self.v81_real_counterfactual_margin = float(v81_real_counterfactual_margin)
        self.v81_hard_aigc_residual_min_weight = float(v81_hard_aigc_residual_min_weight)
        self.v81_easy_aigc_residual_suppress_weight = float(v81_easy_aigc_residual_suppress_weight)
        self.v81_residual_push_target = float(v81_residual_push_target)
        self.v9_phase = str(v9_phase)
        self.v9_semantic_head_weight = float(v9_semantic_head_weight)
        self.v9_frequency_head_weight = float(v9_frequency_head_weight)
        self.v9_hard_real_margin_weight = float(v9_hard_real_margin_weight)
        self.v9_hard_real_margin = float(v9_hard_real_margin)
        self.v9_prototype_weight = float(v9_prototype_weight)
        self.v9_prototype_margin = float(v9_prototype_margin)
        self.semantic_floor_target = 0.0
        self.semantic_floor_loss_weight = 0.0
        self.real_noise_target = 0.45
        self.real_semantic_floor_target = 0.15
        self.aigc_semantic_floor_target = 0.20
        self.real_noise_penalty_loss_weight = 0.02
        self.real_semantic_floor_loss_weight = 0.02
        self.aigc_semantic_floor_loss_weight = 0.015
        self.semantic_consistency_loss_weight = 0.02
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        self.use_ema = use_ema
        self.model_ema = ModelEma(self.model, decay=ema_decay) if use_ema else None
        self.temperature_scaler = TemperatureScaler(init_temperature=1.0).to(device)
        self.best_val_auroc = -1.0
        self.best_hybrid_score = -1e9
        self.start_epoch = 1
        self.collapse_min_separation = float(collapse_min_separation)
        self.collapse_min_logit_std = float(collapse_min_logit_std)
        self.hybrid_score_sep_weight = float(hybrid_score_sep_weight)
        self.set_v8_stage(self.v8_stage)
        if self.v81_controller_only:
            self.configure_v81_controller_only(True)
        if self.v9_phase == "base_debias":
            self.mixup_prob = 0.0

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def set_v8_stage(self, stage: str) -> None:
        self.v8_stage = str(stage)
        base_model = self._unwrap_model()
        if hasattr(base_model, "set_v8_stage"):
            base_model.set_v8_stage(stage)
        if self.v81_controller_only and hasattr(base_model, "apply_v81_training_mode"):
            base_model.apply_v81_training_mode()

    def configure_v81_controller_only(self, enabled: bool = True) -> Dict[str, object]:
        self.v81_controller_only = bool(enabled)
        if self.v81_controller_only:
            self.mixup_prob = 0.0
        base_model = self._unwrap_model()
        summary: Dict[str, object] = {
            "enabled": self.v81_controller_only,
            "frozen_modules": [],
            "trainable_modules": [],
        }
        if hasattr(base_model, "configure_v81_controller_only"):
            summary = base_model.configure_v81_controller_only(self.v81_controller_only)
        if self.v81_controller_only and hasattr(base_model, "apply_v81_training_mode"):
            base_model.apply_v81_training_mode()
        return summary

    def configure_v9_base_debias(self, semantic_trainable_layers: int = 2) -> Dict[str, object]:
        self.v9_phase = "base_debias"
        self.v81_controller_only = False
        self.mixup_prob = 0.0
        base_model = self._unwrap_model()
        summary: Dict[str, object] = {
            "enabled": False,
            "semantic_trainable_layers": int(semantic_trainable_layers),
            "frozen_modules": [],
            "trainable_modules": [],
        }
        if hasattr(base_model, "configure_v9_base_debias"):
            summary = base_model.configure_v9_base_debias(
                semantic_trainable_layers=int(semantic_trainable_layers)
            )
        if hasattr(base_model, "apply_v9_training_mode"):
            base_model.apply_v9_training_mode()
        return summary

    def _lr_lambda(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return max((epoch + 1) / max(self.warmup_epochs, 1), 1e-3)
        progress = (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(float(cosine), 1e-4)

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best_official: bool = False, is_best_hybrid: bool = False) -> None:
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "temperature": self.temperature_scaler.value(),
            "val_metrics": val_metrics,
        }
        if self.model_ema is not None:
            ckpt["ema_shadow"] = {k: v.cpu() for k, v in self.model_ema.shadow.items()}
        latest = self.save_dir / "latest.pth"
        torch.save(ckpt, latest)
        epoch_ckpt = self.save_dir / f"epoch_{epoch:03d}.pth"
        torch.save(ckpt, epoch_ckpt)
        if is_best_official:
            torch.save(ckpt, self.save_dir / "best_official.pth")
            torch.save(ckpt, self.save_dir / "best.pth")
        if is_best_hybrid:
            torch.save(ckpt, self.save_dir / "best_hybrid.pth")
        all_ckpts = sorted(self.save_dir.glob("epoch_*.pth"))
        while len(all_ckpts) > self.max_checkpoint_keep:
            all_ckpts[0].unlink(missing_ok=True)
            all_ckpts.pop(0)

    def resume(self, checkpoint_path: Path) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_state = ckpt["model"]
        current_state = self.model.state_dict()
        filtered_state = {
            key: value
            for key, value in checkpoint_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        load_result = self.model.load_state_dict(filtered_state, strict=False)
        missing_keys = list(getattr(load_result, "missing_keys", []))
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
        skipped_keys = [
            key for key, value in checkpoint_state.items()
            if key in current_state and current_state[key].shape != value.shape
        ]
        if missing_keys or unexpected_keys or skipped_keys:
            print(
                "Resume state dict note | "
                f"loaded={len(filtered_state)} missing={len(missing_keys)} "
                f"unexpected={len(unexpected_keys)} skipped_shape={len(skipped_keys)}"
            )
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            print(f"Skipping optimizer state restore due to mismatch: {exc}")
        try:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as exc:
            print(f"Skipping scheduler state restore due to mismatch: {exc}")
        try:
            self.scaler.load_state_dict(ckpt["scaler"])
        except Exception as exc:
            print(f"Skipping scaler state restore due to mismatch: {exc}")
        temp = float(ckpt.get("temperature", 1.0))
        with torch.no_grad():
            self.temperature_scaler.log_temperature.copy_(torch.log(torch.tensor([temp], device=self.device)))
        if self.model_ema is not None and "ema_shadow" in ckpt:
            self.model_ema.shadow = {k: v.to(self.device) for k, v in ckpt["ema_shadow"].items()}
        self.start_epoch = int(ckpt["epoch"]) + 1
        metrics = ckpt.get("val_metrics", {})
        self.best_val_auroc = float(metrics.get("auroc", self.best_val_auroc))

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.view(-1).float()
        values = values.view(-1).float()
        weight_sum = weights.sum()
        if float(weight_sum.detach().item()) <= 0.0:
            return values.new_zeros(())
        return (values * weights).sum() / weight_sum.clamp_min(1e-6)

    def _masked_binary_cross_entropy_probabilities(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = probabilities.view(-1).float().clamp(1e-6, 1.0 - 1e-6)
        targets = targets.view(-1).float()
        mask = mask.view(-1).float()
        with torch.autocast(device_type=self.device.type, enabled=False):
            losses = F.binary_cross_entropy(probabilities, targets, reduction="none")
        return self._weighted_mean(losses, mask)

    def _masked_mse(
        self,
        values: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        values = values.view(-1).float()
        targets = targets.view(-1).float()
        mask = mask.view(-1).float()
        losses = F.mse_loss(values, targets, reduction="none")
        return self._weighted_mean(losses, mask)

    def _metadata_boolean_mask(
        self,
        metadata,
        key: str,
        batch_size: int,
    ) -> torch.Tensor:
        if metadata is None or not isinstance(metadata, dict):
            return torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        values = metadata.get(key)
        if values is None:
            return torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        if torch.is_tensor(values):
            return values.to(self.device).view(-1).float()
        if isinstance(values, np.ndarray):
            return torch.as_tensor(values, device=self.device).view(-1).float()
        if isinstance(values, (list, tuple)):
            normalized = []
            for value in values:
                if isinstance(value, str):
                    normalized.append(1.0 if value.lower() in {"1", "true", "yes"} else 0.0)
                else:
                    normalized.append(float(bool(value)))
            return torch.tensor(normalized, device=self.device, dtype=torch.float32).view(-1)
        return torch.zeros(batch_size, device=self.device, dtype=torch.float32)

    def _compute_v9_prototype_margin_loss(
        self,
        base_feat: torch.Tensor,
        labels: torch.Tensor,
        hard_real_mask: torch.Tensor,
    ) -> torch.Tensor:
        labels_flat = labels.view(-1).float()
        hard_real_mask = hard_real_mask.view(-1).float()
        real_mask = labels_flat < 0.5
        aigc_mask = labels_flat >= 0.5
        active_hard_real_mask = hard_real_mask > 0.5
        if (not torch.any(real_mask)) or (not torch.any(aigc_mask)) or (not torch.any(active_hard_real_mask)):
            return base_feat.new_zeros(())
        feat = F.normalize(base_feat.float(), dim=1)
        real_proto = F.normalize(feat[real_mask].mean(dim=0, keepdim=True), dim=1)
        aigc_proto = F.normalize(feat[aigc_mask].mean(dim=0, keepdim=True), dim=1)
        hard_feats = feat[active_hard_real_mask]
        sim_real = torch.matmul(hard_feats, real_proto.t()).view(-1)
        sim_aigc = torch.matmul(hard_feats, aigc_proto.t()).view(-1)
        return torch.relu(base_feat.new_tensor(self.v9_prototype_margin) - (sim_real - sim_aigc)).mean()

    def _compute_branch_supervision_losses(
        self,
        raw_weights: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        labels = labels.view(-1).float()
        semantic = raw_weights[:, 0]
        frequency = raw_weights[:, 1]
        noise = raw_weights[:, 2]

        real_factor = (1.0 - labels).clamp(0.0, 1.0)
        aigc_factor = labels.clamp(0.0, 1.0)

        real_noise_penalty = torch.relu(noise - noise.new_tensor(self.real_noise_target))
        real_semantic_floor = torch.relu(semantic.new_tensor(self.real_semantic_floor_target) - semantic)
        aigc_semantic_floor = torch.relu(semantic.new_tensor(self.aigc_semantic_floor_target) - semantic)

        return {
            "semantic_floor_loss": semantic.new_zeros(()),  # Deprecated in V7.
            "semantic_vs_noise_loss": semantic.new_zeros(()),  # Deprecated in V7.
            "noise_penalty_loss": semantic.new_zeros(()),  # Deprecated in V7.
            "raw_semantic_mean": semantic.mean(),
            "raw_frequency_mean": frequency.mean(),
            "raw_noise_mean": noise.mean(),
            "real_noise_mean_raw": self._weighted_mean(noise, real_factor),
            "real_semantic_mean_raw": self._weighted_mean(semantic, real_factor),
            "aigc_semantic_mean_raw": self._weighted_mean(semantic, aigc_factor),
            "real_noise_penalty_loss": self.real_noise_penalty_loss_weight
            * self._weighted_mean(real_noise_penalty, real_factor),
            "real_semantic_floor_loss": self.real_semantic_floor_loss_weight
            * self._weighted_mean(real_semantic_floor, real_factor),
            "aigc_semantic_floor_loss": self.aigc_semantic_floor_loss_weight
            * self._weighted_mean(aigc_semantic_floor, aigc_factor),
        }

    def _build_v81_alpha_masks(
        self,
        labels: torch.Tensor,
        base_probability: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        labels = labels.view(-1).float()
        base_probability = base_probability.view(-1).float()
        real_mask = labels < 0.5
        easy_aigc_mask = (labels >= 0.5) & (base_probability >= self.v81_alpha_easy_aigc_threshold)
        hard_aigc_mask = (labels >= 0.5) & (~easy_aigc_mask)
        return {
            "real_mask": real_mask,
            "easy_aigc_mask": easy_aigc_mask,
            "hard_aigc_mask": hard_aigc_mask,
            "aigc_mask": labels >= 0.5,
        }

    def _compute_v81_controller_only_terms(
        self,
        out: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        labels = labels.view(-1).float()
        final_logit = out["logit"].view(-1)
        base_logit = out["base_logit"].view(-1)
        noise_delta_logit = out["noise_delta_logit"].view(-1)
        alpha_used = out.get("alpha_used", out["alpha"]).view(-1)
        alpha_ratio = out.get("alpha_ratio_used", out.get("alpha_ratio"))
        if alpha_ratio is None:
            alpha_norm = alpha_used.new_zeros(alpha_used.shape)
        else:
            alpha_norm = alpha_ratio.view(-1).float().clamp(0.0, 1.0)
        base_probability = torch.sigmoid(base_logit.detach())
        masks = self._build_v81_alpha_masks(labels=labels, base_probability=base_probability)
        real_mask = masks["real_mask"].float()
        easy_aigc_mask = masks["easy_aigc_mask"].float()
        hard_aigc_mask = masks["hard_aigc_mask"].float()
        aigc_mask = masks["aigc_mask"].float()

        zeros = torch.zeros_like(alpha_norm)
        hard_target = torch.full_like(alpha_norm, self.v81_alpha_hard_target)

        residual_push = alpha_used * noise_delta_logit
        base_vs_final_gap = final_logit - base_logit

        alpha_real_loss = self.v81_alpha_real_weight * self._masked_binary_cross_entropy_probabilities(
            probabilities=alpha_norm,
            targets=zeros,
            mask=real_mask,
        )
        alpha_easy_aigc_loss = self.v81_alpha_easy_aigc_weight * self._masked_binary_cross_entropy_probabilities(
            probabilities=alpha_norm,
            targets=zeros,
            mask=easy_aigc_mask,
        )
        alpha_hard_aigc_loss = self.v81_alpha_hard_aigc_weight * self._masked_mse(
            values=alpha_norm,
            targets=hard_target,
            mask=hard_aigc_mask,
        )
        real_residual_push_loss = self.v81_real_residual_push_weight * self._weighted_mean(
            torch.relu(residual_push),
            real_mask,
        )
        real_counterfactual_loss = self.v81_real_counterfactual_weight * self._weighted_mean(
            torch.relu(base_vs_final_gap - self.v81_real_counterfactual_margin),
            real_mask,
        )
        hard_aigc_residual_min_loss = self.v81_hard_aigc_residual_min_weight * self._weighted_mean(
            torch.relu(residual_push.new_tensor(self.v81_residual_push_target) - residual_push),
            hard_aigc_mask,
        )
        easy_aigc_residual_suppress_loss = self.v81_easy_aigc_residual_suppress_weight * self._weighted_mean(
            torch.relu(residual_push - residual_push.new_tensor(self.v81_residual_push_target)),
            easy_aigc_mask,
        )

        return {
            "alpha_real_loss": alpha_real_loss,
            "alpha_easy_aigc_loss": alpha_easy_aigc_loss,
            "alpha_hard_aigc_loss": alpha_hard_aigc_loss,
            "real_residual_push_loss": real_residual_push_loss,
            "real_counterfactual_loss": real_counterfactual_loss,
            "hard_aigc_residual_min_loss": hard_aigc_residual_min_loss,
            "easy_aigc_residual_suppress_loss": easy_aigc_residual_suppress_loss,
            "alpha_mean": alpha_used.mean(),
            "alpha_mean_real": self._weighted_mean(alpha_used, real_mask),
            "alpha_mean_easy_aigc": self._weighted_mean(alpha_used, easy_aigc_mask),
            "alpha_mean_hard_aigc": self._weighted_mean(alpha_used, hard_aigc_mask),
            "alpha_mean_aigc": self._weighted_mean(alpha_used, aigc_mask),
            "residual_push_real_mean": self._weighted_mean(residual_push, real_mask),
            "residual_push_hard_aigc_mean": self._weighted_mean(residual_push, hard_aigc_mask),
            "residual_push_easy_aigc_mean": self._weighted_mean(residual_push, easy_aigc_mask),
            "base_vs_final_gap_real": self._weighted_mean(base_vs_final_gap, real_mask),
            "base_vs_final_gap_aigc": self._weighted_mean(base_vs_final_gap, aigc_mask),
            "real_mask_count": real_mask.sum(),
            "easy_aigc_mask_count": easy_aigc_mask.sum(),
            "hard_aigc_mask_count": hard_aigc_mask.sum(),
        }

    def _compute_v9_base_debias_loss(
        self,
        out: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        hard_real_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        labels = labels.view(-1, 1).float()
        labels_flat = labels.view(-1)
        base_logit = out["base_logit"]
        semantic_logit = out.get("semantic_logit")
        frequency_logit = out.get("freq_logit")
        base_feat = out.get("base_feat", out.get("fused_feat"))
        if hard_real_mask is None:
            hard_real_mask = base_logit.new_zeros(labels_flat.shape)
        hard_real_mask = hard_real_mask.view(-1).float()
        hard_real_real_mask = hard_real_mask * (labels_flat < 0.5).float()

        main_bce = self.loss_fn.bce_weight * F.binary_cross_entropy_with_logits(base_logit, labels)
        focal_loss = (
            self.loss_fn._focal(base_logit, labels)  # type: ignore[attr-defined]
            if self.loss_fn.focal_weight > 0
            else labels.new_zeros(())
        )
        total = main_bce + self.loss_fn.focal_weight * focal_loss

        semantic_head_loss = labels.new_zeros(())
        if semantic_logit is not None and self.v9_semantic_head_weight > 0:
            semantic_head_loss = self.v9_semantic_head_weight * F.binary_cross_entropy_with_logits(
                semantic_logit,
                labels,
            )
            total = total + semantic_head_loss

        frequency_head_loss = labels.new_zeros(())
        if frequency_logit is not None and self.v9_frequency_head_weight > 0:
            frequency_head_loss = self.v9_frequency_head_weight * F.binary_cross_entropy_with_logits(
                frequency_logit,
                labels,
            )
            total = total + frequency_head_loss

        hard_real_margin_loss = self.v9_hard_real_margin_weight * self._weighted_mean(
            torch.relu(base_logit.view(-1) + self.v9_hard_real_margin),
            hard_real_real_mask,
        )
        total = total + hard_real_margin_loss

        prototype_margin_loss = labels.new_zeros(())
        if base_feat is not None and self.v9_prototype_weight > 0:
            prototype_margin_loss = self.v9_prototype_weight * self._compute_v9_prototype_margin_loss(
                base_feat=base_feat,
                labels=labels_flat,
                hard_real_mask=hard_real_real_mask,
            )
            total = total + prototype_margin_loss

        sf_weights_raw = out.get("sf_weights_raw")
        if sf_weights_raw is None:
            raw_semantic_mean = labels.new_zeros(())
            raw_frequency_mean = labels.new_zeros(())
            raw_noise_mean = labels.new_zeros(())
        else:
            raw_semantic_mean = sf_weights_raw[:, 0].mean()
            raw_frequency_mean = sf_weights_raw[:, 1].mean()
            raw_noise_mean = out.get("alpha_ratio_used", out.get("alpha_ratio", labels.new_zeros(labels_flat.shape))).view(-1).mean()

        return {
            "total_loss": total,
            "main_bce_loss": main_bce,
            "focal_loss": focal_loss,
            "base_bce_loss": main_bce,
            "semantic_head_loss": semantic_head_loss,
            "frequency_head_loss": frequency_head_loss,
            "noise_head_loss": labels.new_zeros(()),
            "hard_real_margin_loss": hard_real_margin_loss,
            "prototype_margin_loss": prototype_margin_loss,
            "hard_real_count": hard_real_real_mask.sum(),
            "hard_real_hit_rate": hard_real_real_mask.mean(),
            "fusion_reg_loss": labels.new_zeros(()),
            "semantic_floor_loss": labels.new_zeros(()),
            "semantic_vs_noise_loss": labels.new_zeros(()),
            "noise_penalty_loss": labels.new_zeros(()),
            "real_noise_penalty_loss": labels.new_zeros(()),
            "real_semantic_floor_loss": labels.new_zeros(()),
            "aigc_semantic_floor_loss": labels.new_zeros(()),
            "real_alpha_loss": labels.new_zeros(()),
            "alpha_easy_aigc_loss": labels.new_zeros(()),
            "real_noise_push_loss": labels.new_zeros(()),
            "real_counterfactual_loss": labels.new_zeros(()),
            "hard_aigc_alpha_loss": labels.new_zeros(()),
            "hard_aigc_residual_min_loss": labels.new_zeros(()),
            "easy_aigc_residual_suppress_loss": labels.new_zeros(()),
            "semantic_mean": out["fusion_weights"][:, 0].mean() if "fusion_weights" in out else raw_semantic_mean,
            "semantic_mean_raw": raw_semantic_mean,
            "raw_semantic_mean": raw_semantic_mean,
            "raw_frequency_mean": raw_frequency_mean,
            "raw_noise_mean": raw_noise_mean,
            "noise_mean_raw": raw_noise_mean,
            "real_noise_mean_raw": labels.new_zeros(()),
            "real_semantic_mean_raw": labels.new_zeros(()),
            "aigc_semantic_mean_raw": labels.new_zeros(()),
            "base_logit_mean": base_logit.mean(),
            "final_logit_mean": out["logit"].mean(),
            "noise_delta_mean": out.get("noise_delta_logit", labels.new_zeros(labels.shape)).mean(),
            "alpha_mean": out.get("alpha_active", labels.new_zeros(labels.shape)).mean(),
            "alpha_mean_real": labels.new_zeros(()),
            "alpha_mean_easy_aigc": labels.new_zeros(()),
            "alpha_mean_hard_aigc": labels.new_zeros(()),
            "alpha_mean_aigc": labels.new_zeros(()),
            "residual_push_real_mean": labels.new_zeros(()),
            "residual_push_hard_aigc_mean": labels.new_zeros(()),
            "base_vs_final_gap_real": labels.new_zeros(()),
            "base_vs_final_gap_aigc": labels.new_zeros(()),
            "counterfactual_hits": labels.new_zeros(()),
        }

    def _compute_v8_loss(
        self,
        out: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        labels = labels.view(-1, 1).float()
        final_logit = out["logit"]
        base_logit = out["base_logit"]
        alpha = out["alpha"]
        alpha_used = out.get("alpha_used", alpha)
        noise_delta_logit = out["noise_delta_logit"]

        final_bce = F.binary_cross_entropy_with_logits(final_logit, labels)
        focal_loss = (
            self.loss_fn._focal(final_logit, labels)  # type: ignore[attr-defined]
            if self.loss_fn.focal_weight > 0
            else labels.new_zeros(())
        )
        total = self.loss_fn.bce_weight * final_bce + self.loss_fn.focal_weight * focal_loss

        base_bce_loss = self.v8_base_loss_weight * F.binary_cross_entropy_with_logits(base_logit, labels)
        total = total + base_bce_loss

        semantic_head_loss = labels.new_zeros(())
        if "semantic_logit" in out and self.v8_semantic_head_weight > 0:
            semantic_head_loss = self.v8_semantic_head_weight * F.binary_cross_entropy_with_logits(
                out["semantic_logit"],
                labels,
            )
            total = total + semantic_head_loss

        frequency_head_loss = labels.new_zeros(())
        if "freq_logit" in out and self.v8_frequency_head_weight > 0:
            frequency_head_loss = self.v8_frequency_head_weight * F.binary_cross_entropy_with_logits(
                out["freq_logit"],
                labels,
            )
            total = total + frequency_head_loss

        noise_head_loss = labels.new_zeros(())
        noise_head_weight = self.v8_noise_head_weight if self.v8_stage != "debias_base" else 0.0
        if noise_head_weight > 0:
            noise_head_loss = noise_head_weight * F.binary_cross_entropy_with_logits(noise_delta_logit, labels)
            total = total + noise_head_loss

        labels_flat = labels.view(-1)
        real_mask = (labels_flat < 0.5).float()
        aigc_mask = (labels_flat >= 0.5).float()
        hard_aigc_mask = (
            (labels_flat >= 0.5)
            & (torch.sigmoid(base_logit.detach().view(-1)) < 0.5)
        ).float()
        easy_aigc_mask = (
            (labels_flat >= 0.5)
            & (torch.sigmoid(base_logit.detach().view(-1)) >= self.v81_alpha_easy_aigc_threshold)
        ).float()

        real_alpha_loss = labels.new_zeros(())
        real_noise_push_loss = labels.new_zeros(())
        real_counterfactual_loss = labels.new_zeros(())
        hard_aigc_alpha_loss = labels.new_zeros(())
        alpha_easy_aigc_loss = labels.new_zeros(())
        hard_aigc_residual_min_loss = labels.new_zeros(())
        easy_aigc_residual_suppress_loss = labels.new_zeros(())
        alpha_mean_easy_aigc = labels.new_zeros(())
        alpha_mean_hard_aigc = labels.new_zeros(())
        residual_push_real_mean = labels.new_zeros(())
        residual_push_hard_aigc_mean = labels.new_zeros(())
        residual_push_easy_aigc_mean = labels.new_zeros(())
        base_vs_final_gap_real = labels.new_zeros(())
        base_vs_final_gap_aigc = labels.new_zeros(())
        if self.v8_stage != "debias_base":
            if self.v81_controller_only:
                v81_terms = self._compute_v81_controller_only_terms(out, labels)
                real_alpha_loss = v81_terms["alpha_real_loss"]
                real_noise_push_loss = v81_terms["real_residual_push_loss"]
                real_counterfactual_loss = v81_terms["real_counterfactual_loss"]
                hard_aigc_alpha_loss = v81_terms["alpha_hard_aigc_loss"]
                alpha_easy_aigc_loss = v81_terms["alpha_easy_aigc_loss"]
                hard_aigc_residual_min_loss = v81_terms["hard_aigc_residual_min_loss"]
                easy_aigc_residual_suppress_loss = v81_terms["easy_aigc_residual_suppress_loss"]
                alpha_mean_easy_aigc = v81_terms["alpha_mean_easy_aigc"]
                alpha_mean_hard_aigc = v81_terms["alpha_mean_hard_aigc"]
                residual_push_real_mean = v81_terms["residual_push_real_mean"]
                residual_push_hard_aigc_mean = v81_terms["residual_push_hard_aigc_mean"]
                residual_push_easy_aigc_mean = v81_terms["residual_push_easy_aigc_mean"]
                base_vs_final_gap_real = v81_terms["base_vs_final_gap_real"]
                base_vs_final_gap_aigc = v81_terms["base_vs_final_gap_aigc"]
                total = (
                    total
                    + real_alpha_loss
                    + alpha_easy_aigc_loss
                    + hard_aigc_alpha_loss
                    + real_noise_push_loss
                    + real_counterfactual_loss
                    + hard_aigc_residual_min_loss
                    + easy_aigc_residual_suppress_loss
                )
            else:
                real_alpha_loss = self.v8_real_alpha_loss_weight * self._weighted_mean(alpha_used.view(-1), real_mask)
                real_noise_push_loss = self.v8_real_noise_push_weight * self._weighted_mean(
                    alpha_used.view(-1) * torch.relu(noise_delta_logit.view(-1)),
                    real_mask,
                )
                real_counterfactual_loss = self.v8_real_counterfactual_weight * self._weighted_mean(
                    torch.relu((final_logit - base_logit).view(-1) - self.v8_real_counterfactual_margin),
                    real_mask,
                )
                hard_aigc_alpha_loss = self.v8_hard_aigc_alpha_weight * self._weighted_mean(
                    torch.relu(alpha_used.new_tensor(self.v8_hard_aigc_alpha_target) - alpha_used.view(-1)),
                    hard_aigc_mask,
                )
                total = (
                    total
                    + real_alpha_loss
                    + real_noise_push_loss
                    + real_counterfactual_loss
                    + hard_aigc_alpha_loss
                )

        sf_weights_raw = out.get("sf_weights_raw")
        if sf_weights_raw is None:
            raw_semantic_mean = labels.new_zeros(())
            raw_frequency_mean = labels.new_zeros(())
            raw_noise_mean = labels.new_zeros(())
        else:
            raw_semantic_mean = sf_weights_raw[:, 0].mean()
            raw_frequency_mean = sf_weights_raw[:, 1].mean()
            raw_noise_mean = out.get("alpha_ratio", alpha_used.new_zeros(alpha_used.shape)).mean()

        return {
            "total_loss": total,
            "main_bce_loss": final_bce,
            "focal_loss": focal_loss,
            "base_bce_loss": base_bce_loss,
            "semantic_head_loss": semantic_head_loss,
            "frequency_head_loss": frequency_head_loss,
            "noise_head_loss": noise_head_loss,
            "real_alpha_loss": real_alpha_loss,
            "alpha_easy_aigc_loss": alpha_easy_aigc_loss,
            "real_noise_push_loss": real_noise_push_loss,
            "real_counterfactual_loss": real_counterfactual_loss,
            "hard_aigc_alpha_loss": hard_aigc_alpha_loss,
            "hard_aigc_residual_min_loss": hard_aigc_residual_min_loss,
            "easy_aigc_residual_suppress_loss": easy_aigc_residual_suppress_loss,
            "fusion_reg_loss": total.new_zeros(()),
            "semantic_floor_loss": total.new_zeros(()),  # deprecated in V8
            "semantic_vs_noise_loss": total.new_zeros(()),  # deprecated in V8
            "noise_penalty_loss": total.new_zeros(()),  # deprecated in V8
            "real_noise_penalty_loss": total.new_zeros(()),  # deprecated in V8
            "real_semantic_floor_loss": total.new_zeros(()),  # deprecated in V8
            "aigc_semantic_floor_loss": total.new_zeros(()),  # deprecated in V8
            "semantic_mean": out["fusion_weights"][:, 0].mean() if "fusion_weights" in out else raw_semantic_mean,
            "semantic_mean_raw": raw_semantic_mean,
            "raw_semantic_mean": raw_semantic_mean,
            "raw_frequency_mean": raw_frequency_mean,
            "raw_noise_mean": raw_noise_mean,
            "noise_mean_raw": raw_noise_mean,
            "real_noise_mean_raw": self._weighted_mean(out.get("alpha_ratio", alpha_used.view(-1)), real_mask),
            "real_semantic_mean_raw": self._weighted_mean(sf_weights_raw[:, 0] if sf_weights_raw is not None else alpha_used.view(-1) * 0.0, real_mask),
            "aigc_semantic_mean_raw": self._weighted_mean(sf_weights_raw[:, 0] if sf_weights_raw is not None else alpha_used.view(-1) * 0.0, aigc_mask),
            "base_logit_mean": base_logit.mean(),
            "final_logit_mean": final_logit.mean(),
            "noise_delta_mean": noise_delta_logit.mean(),
            "alpha_mean": alpha_used.mean(),
            "alpha_mean_real": self._weighted_mean(alpha_used.view(-1), real_mask),
            "alpha_mean_easy_aigc": alpha_mean_easy_aigc,
            "alpha_mean_hard_aigc": alpha_mean_hard_aigc,
            "alpha_mean_aigc": self._weighted_mean(alpha_used.view(-1), aigc_mask),
            "residual_push_real_mean": residual_push_real_mean,
            "residual_push_hard_aigc_mean": residual_push_hard_aigc_mean,
            "residual_push_easy_aigc_mean": residual_push_easy_aigc_mean,
            "base_vs_final_gap_real": base_vs_final_gap_real,
            "base_vs_final_gap_aigc": base_vs_final_gap_aigc,
            "easy_aigc_count": easy_aigc_mask.sum(),
            "hard_aigc_count": hard_aigc_mask.sum(),
            "counterfactual_hits": hard_aigc_mask.sum(),
        }

    def _forward_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        metadata=None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        out = self.model(images)
        if self.v9_phase == "base_debias" and self.v8_enable_base_residual_fusion and ("base_logit" in out):
            hard_real_mask = self._metadata_boolean_mask(
                metadata=metadata,
                key="hard_real_buffer_hit",
                batch_size=int(labels.shape[0]),
            )
            loss_dict = self._compute_v9_base_debias_loss(out, labels, hard_real_mask=hard_real_mask)
            return out, loss_dict
        if self.v8_enable_base_residual_fusion and ("base_logit" in out):
            loss_dict = self._compute_v8_loss(out, labels)
            return out, loss_dict
        loss_dict = self.loss_fn(out, labels)

        # fusion_reg_loss is already scaled in the model; add it directly.
        if "fusion_reg_loss" in out:
            loss_dict["fusion_reg_loss"] = out["fusion_reg_loss"]
            loss_dict["total_loss"] = loss_dict["total_loss"] + out["fusion_reg_loss"]
        else:
            loss_dict["fusion_reg_loss"] = loss_dict["total_loss"].new_zeros(())

        if "fusion_weights" in out:
            raw_weights = out.get("fusion_weights_raw", out["fusion_weights"])
            semantic_mean = out["fusion_weights"][:, 0].mean()
            branch_losses = self._compute_branch_supervision_losses(raw_weights, labels)
            loss_dict["semantic_floor_loss"] = branch_losses["semantic_floor_loss"]
            loss_dict["semantic_mean"] = semantic_mean
            loss_dict["semantic_mean_raw"] = branch_losses["raw_semantic_mean"]
            loss_dict["raw_semantic_mean"] = branch_losses["raw_semantic_mean"]
            loss_dict["raw_frequency_mean"] = branch_losses["raw_frequency_mean"]
            loss_dict["raw_noise_mean"] = branch_losses["raw_noise_mean"]
            loss_dict["noise_mean_raw"] = branch_losses["raw_noise_mean"]
            loss_dict["real_noise_mean_raw"] = branch_losses["real_noise_mean_raw"]
            loss_dict["real_semantic_mean_raw"] = branch_losses["real_semantic_mean_raw"]
            loss_dict["aigc_semantic_mean_raw"] = branch_losses["aigc_semantic_mean_raw"]
            loss_dict["semantic_vs_noise_loss"] = branch_losses["semantic_vs_noise_loss"]
            loss_dict["noise_penalty_loss"] = branch_losses["noise_penalty_loss"]
            loss_dict["real_noise_penalty_loss"] = branch_losses["real_noise_penalty_loss"]
            loss_dict["real_semantic_floor_loss"] = branch_losses["real_semantic_floor_loss"]
            loss_dict["aigc_semantic_floor_loss"] = branch_losses["aigc_semantic_floor_loss"]
            loss_dict["total_loss"] = (
                loss_dict["total_loss"]
                + branch_losses["real_noise_penalty_loss"]
                + branch_losses["real_semantic_floor_loss"]
                + branch_losses["aigc_semantic_floor_loss"]
            )
        else:
            loss_dict["semantic_floor_loss"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["semantic_mean"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["semantic_mean_raw"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["raw_semantic_mean"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["raw_frequency_mean"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["raw_noise_mean"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["noise_mean_raw"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["real_noise_mean_raw"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["real_semantic_mean_raw"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["aigc_semantic_mean_raw"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["semantic_vs_noise_loss"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["noise_penalty_loss"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["real_noise_penalty_loss"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["real_semantic_floor_loss"] = loss_dict["total_loss"].new_zeros(())
            loss_dict["aigc_semantic_floor_loss"] = loss_dict["total_loss"].new_zeros(())

        return out, loss_dict

    def _compute_consistency_loss(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        temperature = max(self.temperature_scaler.value(), 1e-6)
        prob1 = torch.sigmoid(logits1 / temperature)
        prob2 = torch.sigmoid(logits2 / temperature)
        return F.mse_loss(prob1, prob2)

    def _compute_semantic_consistency_loss(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        feat1 = F.normalize(feat1.float(), dim=1)
        feat2 = F.normalize(feat2.float(), dim=1)
        return F.mse_loss(feat1, feat2)

    def _set_aux_weight_for_epoch(self, epoch: int) -> None:
        if self.aux_schedule == "default":
            self.loss_fn.aux_weight = 0.5 if epoch <= 4 else self.base_aux_weight
        elif self.aux_schedule == "staged":
            if epoch <= 4:
                self.loss_fn.aux_weight = 0.5
            elif epoch <= 10:
                self.loss_fn.aux_weight = 0.3
            else:
                self.loss_fn.aux_weight = self.base_aux_weight
        else:
            self.loss_fn.aux_weight = self.base_aux_weight

    def _mixup(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        paired_images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.v81_controller_only:
            return images, labels, paired_images
        if self.mixup_prob <= 0.0 or self.mixup_alpha <= 0.0:
            return images, labels, paired_images
        if torch.rand(1, device=images.device).item() > self.mixup_prob:
            return images, labels, paired_images
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        idx = torch.randperm(images.size(0), device=images.device)
        mixed_images = lam * images + (1.0 - lam) * images[idx]
        mixed_labels = lam * labels + (1.0 - lam) * labels[idx]
        mixed_paired = None
        if paired_images is not None:
            mixed_paired = lam * paired_images + (1.0 - lam) * paired_images[idx]
        return mixed_images, mixed_labels, mixed_paired

    def _compute_logit_separation_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        logits = logits.view(-1).float()
        labels = labels.view(-1).float()
        real_mask = labels < 0.5
        aigc_mask = labels >= 0.5
        logit_std = float(logits.std(unbiased=False).item()) if logits.numel() > 0 else 0.0
        real_mean = float(logits[real_mask].mean().item()) if real_mask.any() else 0.0
        aigc_mean = float(logits[aigc_mask].mean().item()) if aigc_mask.any() else 0.0
        separation = float(aigc_mean - real_mean)
        collapse = int((separation < self.collapse_min_separation) or (logit_std < self.collapse_min_logit_std))
        return {
            "logit_mean_aigc": aigc_mean,
            "logit_mean_real": real_mean,
            "logit_separation": separation,
            "logit_std": logit_std,
            "collapse_flag": collapse,
        }

    def _compute_hybrid_score(self, val_metrics: Dict[str, float]) -> float:
        auroc = float(val_metrics.get("auroc", 0.0))
        separation = max(float(val_metrics.get("logit_separation", 0.0)), 0.0)
        collapse_penalty = 0.2 if int(val_metrics.get("collapse_flag", 0)) > 0 else 0.0
        return auroc + self.hybrid_score_sep_weight * separation - collapse_penalty

    def _summarize_branch_weights(self, fusion_batches, prefix: str = "fusion") -> Dict[str, float]:
        if not fusion_batches:
            return {}
        weights = torch.cat(fusion_batches, dim=0).float()
        branch_names = ("semantic", "frequency", "noise")
        stats: Dict[str, float] = {
            f"{prefix}_weight_samples": float(weights.shape[0]),
        }
        for idx, branch in enumerate(branch_names):
            column = weights[:, idx]
            stats[f"{prefix}_{branch}_mean"] = float(column.mean().item())
            stats[f"{prefix}_{branch}_std"] = float(column.std(unbiased=False).item())
            stats[f"{prefix}_{branch}_min"] = float(column.min().item())
            stats[f"{prefix}_{branch}_max"] = float(column.max().item())
        return stats

    def train_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        base_model = self._unwrap_model()
        if self.v81_controller_only and hasattr(base_model, "apply_v81_training_mode"):
            base_model.apply_v81_training_mode()
        if self.v9_phase == "base_debias" and hasattr(base_model, "apply_v9_training_mode"):
            base_model.apply_v9_training_mode()
        self._set_aux_weight_for_epoch(epoch)
        epoch_loss = []
        epoch_bce = []
        epoch_consistency = []
        epoch_fusion_reg = []
        epoch_semantic_floor = []
        epoch_real_noise_penalty = []
        epoch_real_semantic_floor = []
        epoch_aigc_semantic_floor = []
        epoch_semantic_consistency = []
        epoch_base_bce = []
        epoch_semantic_head = []
        epoch_frequency_head = []
        epoch_noise_head = []
        epoch_real_alpha = []
        epoch_real_noise_push = []
        epoch_real_counterfactual = []
        epoch_hard_aigc_alpha = []
        epoch_alpha_easy_aigc_loss = []
        epoch_hard_aigc_residual_min = []
        epoch_easy_aigc_residual_suppress = []
        epoch_alpha_mean = []
        epoch_alpha_mean_real = []
        epoch_alpha_mean_easy_aigc = []
        epoch_alpha_mean_hard_aigc = []
        epoch_alpha_mean_aigc = []
        epoch_base_logit_mean = []
        epoch_noise_delta_mean = []
        epoch_residual_push_real = []
        epoch_residual_push_hard_aigc = []
        epoch_base_gap_real = []
        epoch_base_gap_aigc = []
        epoch_hard_real_margin = []
        epoch_prototype_margin = []
        epoch_hard_real_hits = []
        fusion_batches = []
        raw_fusion_batches = []
        y_true = []
        y_prob = []
        y_prob_base = []
        start = time.time()
        pbar = tqdm(loader, desc=f"Train {epoch}", leave=False)
        for batch in pbar:
            images_view2 = None
            metadata = None
            if len(batch) == 4:
                images, labels, metadata, images_view2 = batch
            else:
                images, labels, metadata = batch
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if images_view2 is not None:
                images_view2 = images_view2.to(self.device, non_blocking=True)
            images, labels_mix, images_view2 = self._mixup(images, labels, images_view2)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                out, loss_dict = self._forward_loss(images, labels_mix, metadata=metadata)
                consistency_loss = loss_dict["total_loss"].new_zeros(())
                total_loss = loss_dict["total_loss"]
                bce_loss = loss_dict["main_bce_loss"]
                fusion_reg_loss = loss_dict["fusion_reg_loss"]
                semantic_floor_loss = loss_dict["semantic_floor_loss"]
                real_noise_penalty_loss = loss_dict["real_noise_penalty_loss"]
                real_semantic_floor_loss = loss_dict["real_semantic_floor_loss"]
                aigc_semantic_floor_loss = loss_dict["aigc_semantic_floor_loss"]
                semantic_consistency_loss = loss_dict["total_loss"].new_zeros(())
                if images_view2 is not None and self.consistency_weight > 0:
                    out_view2, loss_dict_view2 = self._forward_loss(images_view2, labels_mix, metadata=metadata)
                    consistency_loss = self._compute_consistency_loss(out["logit"], out_view2["logit"])
                    semantic_consistency_loss = self.semantic_consistency_loss_weight * self._compute_semantic_consistency_loss(
                        out["semantic_feat"],
                        out_view2["semantic_feat"],
                    )
                    total_loss = 0.5 * (loss_dict["total_loss"] + loss_dict_view2["total_loss"])
                    total_loss = total_loss + self.consistency_weight * consistency_loss + semantic_consistency_loss
                    bce_loss = 0.5 * (loss_dict["main_bce_loss"] + loss_dict_view2["main_bce_loss"])
                    fusion_reg_loss = 0.5 * (loss_dict["fusion_reg_loss"] + loss_dict_view2["fusion_reg_loss"])
                    semantic_floor_loss = 0.5 * (
                        loss_dict["semantic_floor_loss"] + loss_dict_view2["semantic_floor_loss"]
                    )
                    real_noise_penalty_loss = 0.5 * (
                        loss_dict["real_noise_penalty_loss"] + loss_dict_view2["real_noise_penalty_loss"]
                    )
                    real_semantic_floor_loss = 0.5 * (
                        loss_dict["real_semantic_floor_loss"] + loss_dict_view2["real_semantic_floor_loss"]
                    )
                    aigc_semantic_floor_loss = 0.5 * (
                        loss_dict["aigc_semantic_floor_loss"] + loss_dict_view2["aigc_semantic_floor_loss"]
                    )
                loss = total_loss
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.model_ema is not None:
                self.model_ema.update(self.model)
            prob = torch.sigmoid(out["logit"]).detach().view(-1).cpu().numpy().tolist()
            y_prob.extend(prob)
            if "base_logit" in out:
                base_prob = torch.sigmoid(out["base_logit"]).detach().view(-1).cpu().numpy().tolist()
                y_prob_base.extend(base_prob)
            y_true.extend((labels.detach().view(-1) >= 0.5).float().cpu().numpy().tolist())
            if "fusion_weights" in out:
                fusion_batches.append(out["fusion_weights"].detach().cpu())
            if "fusion_weights_raw" in out:
                raw_fusion_batches.append(out["fusion_weights_raw"].detach().cpu())
            epoch_loss.append(float(loss.detach().cpu().item()))
            epoch_bce.append(float(bce_loss.detach().cpu().item()))
            epoch_consistency.append(float(consistency_loss.detach().cpu().item()))
            epoch_fusion_reg.append(float(fusion_reg_loss.detach().cpu().item()))
            epoch_semantic_floor.append(float(semantic_floor_loss.detach().cpu().item()))
            epoch_real_noise_penalty.append(float(real_noise_penalty_loss.detach().cpu().item()))
            epoch_real_semantic_floor.append(float(real_semantic_floor_loss.detach().cpu().item()))
            epoch_aigc_semantic_floor.append(float(aigc_semantic_floor_loss.detach().cpu().item()))
            epoch_semantic_consistency.append(float(semantic_consistency_loss.detach().cpu().item()))
            epoch_base_bce.append(float(loss_dict.get("base_bce_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_semantic_head.append(float(loss_dict.get("semantic_head_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_frequency_head.append(float(loss_dict.get("frequency_head_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_noise_head.append(float(loss_dict.get("noise_head_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_real_alpha.append(float(loss_dict.get("real_alpha_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_real_noise_push.append(float(loss_dict.get("real_noise_push_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_real_counterfactual.append(float(loss_dict.get("real_counterfactual_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_hard_aigc_alpha.append(float(loss_dict.get("hard_aigc_alpha_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_alpha_easy_aigc_loss.append(float(loss_dict.get("alpha_easy_aigc_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_hard_aigc_residual_min.append(float(loss_dict.get("hard_aigc_residual_min_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_easy_aigc_residual_suppress.append(float(loss_dict.get("easy_aigc_residual_suppress_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_alpha_mean.append(float(loss_dict.get("alpha_mean", loss.detach() * 0.0).detach().cpu().item()))
            epoch_alpha_mean_real.append(float(loss_dict.get("alpha_mean_real", loss.detach() * 0.0).detach().cpu().item()))
            epoch_alpha_mean_easy_aigc.append(float(loss_dict.get("alpha_mean_easy_aigc", loss.detach() * 0.0).detach().cpu().item()))
            epoch_alpha_mean_hard_aigc.append(float(loss_dict.get("alpha_mean_hard_aigc", loss.detach() * 0.0).detach().cpu().item()))
            epoch_alpha_mean_aigc.append(float(loss_dict.get("alpha_mean_aigc", loss.detach() * 0.0).detach().cpu().item()))
            epoch_base_logit_mean.append(float(loss_dict.get("base_logit_mean", loss.detach() * 0.0).detach().cpu().item()))
            epoch_noise_delta_mean.append(float(loss_dict.get("noise_delta_mean", loss.detach() * 0.0).detach().cpu().item()))
            epoch_residual_push_real.append(float(loss_dict.get("residual_push_real_mean", loss.detach() * 0.0).detach().cpu().item()))
            epoch_residual_push_hard_aigc.append(float(loss_dict.get("residual_push_hard_aigc_mean", loss.detach() * 0.0).detach().cpu().item()))
            epoch_base_gap_real.append(float(loss_dict.get("base_vs_final_gap_real", loss.detach() * 0.0).detach().cpu().item()))
            epoch_base_gap_aigc.append(float(loss_dict.get("base_vs_final_gap_aigc", loss.detach() * 0.0).detach().cpu().item()))
            epoch_hard_real_margin.append(float(loss_dict.get("hard_real_margin_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_prototype_margin.append(float(loss_dict.get("prototype_margin_loss", loss.detach() * 0.0).detach().cpu().item()))
            epoch_hard_real_hits.append(float(loss_dict.get("hard_real_count", loss.detach() * 0.0).detach().cpu().item()))
            pbar.set_postfix({"loss": f"{epoch_loss[-1]:.4f}"})
        elapsed = max(time.time() - start, 1e-6)
        metrics = compute_metrics(y_true=y_true, y_prob=y_prob)
        if y_prob_base:
            base_metrics = compute_metrics(y_true=y_true, y_prob=y_prob_base)
            metrics["base_auroc"] = float(base_metrics.get("auroc", 0.0))
            metrics["base_f1"] = float(base_metrics.get("f1", 0.0))
            metrics["base_precision"] = float(base_metrics.get("precision", 0.0))
            metrics["base_recall"] = float(base_metrics.get("recall", 0.0))
        metrics["loss"] = float(np.mean(epoch_loss)) if epoch_loss else 0.0
        metrics["main_bce_loss"] = float(np.mean(epoch_bce)) if epoch_bce else 0.0
        metrics["consistency_loss"] = float(np.mean(epoch_consistency)) if epoch_consistency else 0.0
        metrics["fusion_reg_loss"] = float(np.mean(epoch_fusion_reg)) if epoch_fusion_reg else 0.0
        metrics["semantic_floor_loss"] = float(np.mean(epoch_semantic_floor)) if epoch_semantic_floor else 0.0
        metrics["semantic_vs_noise_loss"] = 0.0
        metrics["noise_penalty_loss"] = 0.0
        metrics["real_noise_penalty_loss"] = float(np.mean(epoch_real_noise_penalty)) if epoch_real_noise_penalty else 0.0
        metrics["real_semantic_floor_loss"] = float(np.mean(epoch_real_semantic_floor)) if epoch_real_semantic_floor else 0.0
        metrics["aigc_semantic_floor_loss"] = float(np.mean(epoch_aigc_semantic_floor)) if epoch_aigc_semantic_floor else 0.0
        metrics["semantic_consistency_loss"] = float(np.mean(epoch_semantic_consistency)) if epoch_semantic_consistency else 0.0
        metrics["base_bce_loss"] = float(np.mean(epoch_base_bce)) if epoch_base_bce else 0.0
        metrics["semantic_head_loss"] = float(np.mean(epoch_semantic_head)) if epoch_semantic_head else 0.0
        metrics["frequency_head_loss"] = float(np.mean(epoch_frequency_head)) if epoch_frequency_head else 0.0
        metrics["noise_head_loss"] = float(np.mean(epoch_noise_head)) if epoch_noise_head else 0.0
        metrics["real_alpha_loss"] = float(np.mean(epoch_real_alpha)) if epoch_real_alpha else 0.0
        metrics["real_noise_push_loss"] = float(np.mean(epoch_real_noise_push)) if epoch_real_noise_push else 0.0
        metrics["real_counterfactual_loss"] = float(np.mean(epoch_real_counterfactual)) if epoch_real_counterfactual else 0.0
        metrics["hard_aigc_alpha_loss"] = float(np.mean(epoch_hard_aigc_alpha)) if epoch_hard_aigc_alpha else 0.0
        metrics["alpha_easy_aigc_loss"] = float(np.mean(epoch_alpha_easy_aigc_loss)) if epoch_alpha_easy_aigc_loss else 0.0
        metrics["hard_aigc_residual_min_loss"] = float(np.mean(epoch_hard_aigc_residual_min)) if epoch_hard_aigc_residual_min else 0.0
        metrics["easy_aigc_residual_suppress_loss"] = float(np.mean(epoch_easy_aigc_residual_suppress)) if epoch_easy_aigc_residual_suppress else 0.0
        metrics["alpha_mean"] = float(np.mean(epoch_alpha_mean)) if epoch_alpha_mean else 0.0
        metrics["alpha_mean_real"] = float(np.mean(epoch_alpha_mean_real)) if epoch_alpha_mean_real else 0.0
        metrics["alpha_mean_easy_aigc"] = float(np.mean(epoch_alpha_mean_easy_aigc)) if epoch_alpha_mean_easy_aigc else 0.0
        metrics["alpha_mean_hard_aigc"] = float(np.mean(epoch_alpha_mean_hard_aigc)) if epoch_alpha_mean_hard_aigc else 0.0
        metrics["alpha_mean_aigc"] = float(np.mean(epoch_alpha_mean_aigc)) if epoch_alpha_mean_aigc else 0.0
        metrics["base_logit_mean"] = float(np.mean(epoch_base_logit_mean)) if epoch_base_logit_mean else 0.0
        metrics["noise_delta_mean"] = float(np.mean(epoch_noise_delta_mean)) if epoch_noise_delta_mean else 0.0
        metrics["residual_push_real_mean"] = float(np.mean(epoch_residual_push_real)) if epoch_residual_push_real else 0.0
        metrics["residual_push_hard_aigc_mean"] = float(np.mean(epoch_residual_push_hard_aigc)) if epoch_residual_push_hard_aigc else 0.0
        metrics["base_vs_final_gap_real"] = float(np.mean(epoch_base_gap_real)) if epoch_base_gap_real else 0.0
        metrics["base_vs_final_gap_aigc"] = float(np.mean(epoch_base_gap_aigc)) if epoch_base_gap_aigc else 0.0
        metrics["hard_real_margin_loss"] = float(np.mean(epoch_hard_real_margin)) if epoch_hard_real_margin else 0.0
        metrics["prototype_margin_loss"] = float(np.mean(epoch_prototype_margin)) if epoch_prototype_margin else 0.0
        metrics["hard_real_count"] = float(np.mean(epoch_hard_real_hits)) if epoch_hard_real_hits else 0.0
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        metrics["img_s"] = float(len(y_true) / elapsed)
        metrics.update(self._summarize_branch_weights(fusion_batches, prefix="fusion"))
        metrics.update(self._summarize_branch_weights(raw_fusion_batches, prefix="fusion_raw"))
        metrics["raw_semantic_mean"] = float(metrics.get("fusion_raw_semantic_mean", 0.0))
        metrics["raw_frequency_mean"] = float(metrics.get("fusion_raw_frequency_mean", 0.0))
        metrics["raw_noise_mean"] = float(metrics.get("fusion_raw_noise_mean", 0.0))
        return metrics

    @torch.no_grad()
    def validate_epoch(self, loader, epoch: int, use_ema: bool = True) -> Dict[str, float]:
        self.loss_fn.aux_weight = self.base_aux_weight
        if use_ema and self.model_ema is not None:
            self.model_ema.apply_shadow(self.model)
        self.model.eval()
        epoch_loss = []
        epoch_bce = []
        epoch_fusion_reg = []
        epoch_semantic_floor = []
        epoch_real_noise_penalty = []
        epoch_real_semantic_floor = []
        epoch_aigc_semantic_floor = []
        epoch_base_bce = []
        epoch_semantic_head = []
        epoch_frequency_head = []
        epoch_noise_head = []
        epoch_real_alpha = []
        epoch_real_noise_push = []
        epoch_real_counterfactual = []
        epoch_hard_aigc_alpha = []
        epoch_alpha_easy_aigc_loss = []
        epoch_hard_aigc_residual_min = []
        epoch_easy_aigc_residual_suppress = []
        epoch_alpha_mean = []
        epoch_alpha_mean_real = []
        epoch_alpha_mean_easy_aigc = []
        epoch_alpha_mean_hard_aigc = []
        epoch_alpha_mean_aigc = []
        epoch_base_logit_mean = []
        epoch_noise_delta_mean = []
        epoch_residual_push_real = []
        epoch_residual_push_hard_aigc = []
        epoch_base_gap_real = []
        epoch_base_gap_aigc = []
        epoch_hard_real_margin = []
        epoch_prototype_margin = []
        epoch_hard_real_hits = []
        fusion_batches = []
        raw_fusion_batches = []
        y_true = []
        y_prob = []
        y_prob_base = []
        logits_all = []
        base_logits_all = []
        labels_all = []
        for images, labels, metadata in tqdm(loader, desc=f"Val {epoch}", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            out, loss_dict = self._forward_loss(images, labels, metadata=metadata)
            logits = out["logit"].detach()
            scaled_logits = self.temperature_scaler(logits)
            probs = torch.sigmoid(scaled_logits).view(-1)
            y_prob.extend(probs.cpu().numpy().tolist())
            if "base_logit" in out:
                base_logits = out["base_logit"].detach()
                base_probs = torch.sigmoid(self.temperature_scaler(base_logits)).view(-1)
                y_prob_base.extend(base_probs.cpu().numpy().tolist())
                base_logits_all.append(base_logits.cpu())
            y_true.extend(labels.detach().view(-1).cpu().numpy().tolist())
            epoch_loss.append(float(loss_dict["total_loss"].detach().cpu().item()))
            epoch_bce.append(float(loss_dict["main_bce_loss"].detach().cpu().item()))
            epoch_fusion_reg.append(float(loss_dict["fusion_reg_loss"].detach().cpu().item()))
            epoch_semantic_floor.append(float(loss_dict["semantic_floor_loss"].detach().cpu().item()))
            epoch_real_noise_penalty.append(float(loss_dict["real_noise_penalty_loss"].detach().cpu().item()))
            epoch_real_semantic_floor.append(float(loss_dict["real_semantic_floor_loss"].detach().cpu().item()))
            epoch_aigc_semantic_floor.append(float(loss_dict["aigc_semantic_floor_loss"].detach().cpu().item()))
            epoch_base_bce.append(float(loss_dict.get("base_bce_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_semantic_head.append(float(loss_dict.get("semantic_head_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_frequency_head.append(float(loss_dict.get("frequency_head_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_noise_head.append(float(loss_dict.get("noise_head_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_real_alpha.append(float(loss_dict.get("real_alpha_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_real_noise_push.append(float(loss_dict.get("real_noise_push_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_real_counterfactual.append(float(loss_dict.get("real_counterfactual_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_hard_aigc_alpha.append(float(loss_dict.get("hard_aigc_alpha_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_alpha_easy_aigc_loss.append(float(loss_dict.get("alpha_easy_aigc_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_hard_aigc_residual_min.append(float(loss_dict.get("hard_aigc_residual_min_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_easy_aigc_residual_suppress.append(float(loss_dict.get("easy_aigc_residual_suppress_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_alpha_mean.append(float(loss_dict.get("alpha_mean", logits.sum() * 0.0).detach().cpu().item()))
            epoch_alpha_mean_real.append(float(loss_dict.get("alpha_mean_real", logits.sum() * 0.0).detach().cpu().item()))
            epoch_alpha_mean_easy_aigc.append(float(loss_dict.get("alpha_mean_easy_aigc", logits.sum() * 0.0).detach().cpu().item()))
            epoch_alpha_mean_hard_aigc.append(float(loss_dict.get("alpha_mean_hard_aigc", logits.sum() * 0.0).detach().cpu().item()))
            epoch_alpha_mean_aigc.append(float(loss_dict.get("alpha_mean_aigc", logits.sum() * 0.0).detach().cpu().item()))
            epoch_base_logit_mean.append(float(loss_dict.get("base_logit_mean", logits.sum() * 0.0).detach().cpu().item()))
            epoch_noise_delta_mean.append(float(loss_dict.get("noise_delta_mean", logits.sum() * 0.0).detach().cpu().item()))
            epoch_residual_push_real.append(float(loss_dict.get("residual_push_real_mean", logits.sum() * 0.0).detach().cpu().item()))
            epoch_residual_push_hard_aigc.append(float(loss_dict.get("residual_push_hard_aigc_mean", logits.sum() * 0.0).detach().cpu().item()))
            epoch_base_gap_real.append(float(loss_dict.get("base_vs_final_gap_real", logits.sum() * 0.0).detach().cpu().item()))
            epoch_base_gap_aigc.append(float(loss_dict.get("base_vs_final_gap_aigc", logits.sum() * 0.0).detach().cpu().item()))
            epoch_hard_real_margin.append(float(loss_dict.get("hard_real_margin_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_prototype_margin.append(float(loss_dict.get("prototype_margin_loss", logits.sum() * 0.0).detach().cpu().item()))
            epoch_hard_real_hits.append(float(loss_dict.get("hard_real_count", logits.sum() * 0.0).detach().cpu().item()))
            if "fusion_weights" in out:
                fusion_batches.append(out["fusion_weights"].detach().cpu())
            if "fusion_weights_raw" in out:
                raw_fusion_batches.append(out["fusion_weights_raw"].detach().cpu())
            logits_all.append(logits.cpu())
            labels_all.append(labels.detach().view(-1, 1).float().cpu())
        if logits_all and labels_all:
            fit_logits = torch.cat(logits_all, dim=0).to(self.device)
            fit_labels = torch.cat(labels_all, dim=0).to(self.device)
            self.temperature_scaler.fit(fit_logits, fit_labels)
        metrics = compute_metrics(y_true=y_true, y_prob=y_prob)
        if y_prob_base:
            base_metrics = compute_metrics(y_true=y_true, y_prob=y_prob_base)
            metrics["base_auroc"] = float(base_metrics.get("auroc", 0.0))
            metrics["base_f1"] = float(base_metrics.get("f1", 0.0))
            metrics["base_precision"] = float(base_metrics.get("precision", 0.0))
            metrics["base_recall"] = float(base_metrics.get("recall", 0.0))
        metrics["loss"] = float(np.mean(epoch_loss)) if epoch_loss else 0.0
        metrics["main_bce_loss"] = float(np.mean(epoch_bce)) if epoch_bce else 0.0
        metrics["fusion_reg_loss"] = float(np.mean(epoch_fusion_reg)) if epoch_fusion_reg else 0.0
        metrics["semantic_floor_loss"] = float(np.mean(epoch_semantic_floor)) if epoch_semantic_floor else 0.0
        metrics["semantic_vs_noise_loss"] = 0.0
        metrics["noise_penalty_loss"] = 0.0
        metrics["real_noise_penalty_loss"] = float(np.mean(epoch_real_noise_penalty)) if epoch_real_noise_penalty else 0.0
        metrics["real_semantic_floor_loss"] = float(np.mean(epoch_real_semantic_floor)) if epoch_real_semantic_floor else 0.0
        metrics["aigc_semantic_floor_loss"] = float(np.mean(epoch_aigc_semantic_floor)) if epoch_aigc_semantic_floor else 0.0
        metrics["base_bce_loss"] = float(np.mean(epoch_base_bce)) if epoch_base_bce else 0.0
        metrics["semantic_head_loss"] = float(np.mean(epoch_semantic_head)) if epoch_semantic_head else 0.0
        metrics["frequency_head_loss"] = float(np.mean(epoch_frequency_head)) if epoch_frequency_head else 0.0
        metrics["noise_head_loss"] = float(np.mean(epoch_noise_head)) if epoch_noise_head else 0.0
        metrics["real_alpha_loss"] = float(np.mean(epoch_real_alpha)) if epoch_real_alpha else 0.0
        metrics["real_noise_push_loss"] = float(np.mean(epoch_real_noise_push)) if epoch_real_noise_push else 0.0
        metrics["real_counterfactual_loss"] = float(np.mean(epoch_real_counterfactual)) if epoch_real_counterfactual else 0.0
        metrics["hard_aigc_alpha_loss"] = float(np.mean(epoch_hard_aigc_alpha)) if epoch_hard_aigc_alpha else 0.0
        metrics["alpha_easy_aigc_loss"] = float(np.mean(epoch_alpha_easy_aigc_loss)) if epoch_alpha_easy_aigc_loss else 0.0
        metrics["hard_aigc_residual_min_loss"] = float(np.mean(epoch_hard_aigc_residual_min)) if epoch_hard_aigc_residual_min else 0.0
        metrics["easy_aigc_residual_suppress_loss"] = float(np.mean(epoch_easy_aigc_residual_suppress)) if epoch_easy_aigc_residual_suppress else 0.0
        metrics["alpha_mean"] = float(np.mean(epoch_alpha_mean)) if epoch_alpha_mean else 0.0
        metrics["alpha_mean_real"] = float(np.mean(epoch_alpha_mean_real)) if epoch_alpha_mean_real else 0.0
        metrics["alpha_mean_easy_aigc"] = float(np.mean(epoch_alpha_mean_easy_aigc)) if epoch_alpha_mean_easy_aigc else 0.0
        metrics["alpha_mean_hard_aigc"] = float(np.mean(epoch_alpha_mean_hard_aigc)) if epoch_alpha_mean_hard_aigc else 0.0
        metrics["alpha_mean_aigc"] = float(np.mean(epoch_alpha_mean_aigc)) if epoch_alpha_mean_aigc else 0.0
        metrics["base_logit_mean"] = float(np.mean(epoch_base_logit_mean)) if epoch_base_logit_mean else 0.0
        metrics["noise_delta_mean"] = float(np.mean(epoch_noise_delta_mean)) if epoch_noise_delta_mean else 0.0
        metrics["residual_push_real_mean"] = float(np.mean(epoch_residual_push_real)) if epoch_residual_push_real else 0.0
        metrics["residual_push_hard_aigc_mean"] = float(np.mean(epoch_residual_push_hard_aigc)) if epoch_residual_push_hard_aigc else 0.0
        metrics["base_vs_final_gap_real"] = float(np.mean(epoch_base_gap_real)) if epoch_base_gap_real else 0.0
        metrics["base_vs_final_gap_aigc"] = float(np.mean(epoch_base_gap_aigc)) if epoch_base_gap_aigc else 0.0
        metrics["hard_real_margin_loss"] = float(np.mean(epoch_hard_real_margin)) if epoch_hard_real_margin else 0.0
        metrics["prototype_margin_loss"] = float(np.mean(epoch_prototype_margin)) if epoch_prototype_margin else 0.0
        metrics["hard_real_count"] = float(np.mean(epoch_hard_real_hits)) if epoch_hard_real_hits else 0.0
        metrics["temperature"] = self.temperature_scaler.value()
        metrics.update(self._summarize_branch_weights(fusion_batches, prefix="fusion"))
        metrics.update(self._summarize_branch_weights(raw_fusion_batches, prefix="fusion_raw"))
        metrics["raw_semantic_mean"] = float(metrics.get("fusion_raw_semantic_mean", 0.0))
        metrics["raw_frequency_mean"] = float(metrics.get("fusion_raw_frequency_mean", 0.0))
        metrics["raw_noise_mean"] = float(metrics.get("fusion_raw_noise_mean", 0.0))
        if logits_all and labels_all:
            raw_logits = torch.cat(logits_all, dim=0)
            raw_labels = torch.cat(labels_all, dim=0)
            metrics.update(self._compute_logit_separation_metrics(raw_logits, raw_labels))
        if base_logits_all and labels_all:
            base_logits = torch.cat(base_logits_all, dim=0)
            base_labels = torch.cat(labels_all, dim=0)
            base_sep = self._compute_logit_separation_metrics(base_logits, base_labels)
            metrics["base_logit_mean_aigc"] = float(base_sep["logit_mean_aigc"])
            metrics["base_logit_mean_real"] = float(base_sep["logit_mean_real"])
            metrics["base_logit_separation"] = float(base_sep["logit_separation"])
            metrics["base_logit_std"] = float(base_sep["logit_std"])
        if use_ema and self.model_ema is not None:
            self.model_ema.restore(self.model)
        return metrics

    def fit(
        self,
        train_loader=None,
        val_loader=None,
        log_path: Optional[Path] = None,
        train_loader_factory: Optional[Callable[[int], object]] = None,
    ) -> None:
        if val_loader is None:
            raise ValueError("val_loader must be provided")
        history = []
        if log_path is not None and log_path.exists():
            try:
                loaded = json.loads(log_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    history = loaded
            except Exception:
                history = []
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_train_loader = train_loader_factory(epoch) if train_loader_factory is not None else train_loader
            if epoch_train_loader is None:
                raise ValueError("train_loader or train_loader_factory must provide a loader")
            train_metrics = self.train_epoch(epoch_train_loader, epoch)
            val_metrics = self.validate_epoch(val_loader, epoch, use_ema=True)
            self.scheduler.step()
            is_best_official = float(val_metrics.get("auroc", -1.0)) > self.best_val_auroc
            if is_best_official:
                self.best_val_auroc = float(val_metrics["auroc"])
            hybrid_score = self._compute_hybrid_score(val_metrics)
            is_best_hybrid = hybrid_score > self.best_hybrid_score
            if is_best_hybrid:
                self.best_hybrid_score = hybrid_score
            val_metrics["hybrid_score"] = float(hybrid_score)
            self.save_checkpoint(epoch, val_metrics, is_best_official=is_best_official, is_best_hybrid=is_best_hybrid)
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(row)
            print(
                f"Epoch {epoch} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_bce={train_metrics.get('main_bce_loss', 0.0):.4f} "
                f"train_basebce={train_metrics.get('base_bce_loss', 0.0):.4f} "
                f"train_hreal={train_metrics.get('hard_real_margin_loss', 0.0):.4f} "
                f"train_proto={train_metrics.get('prototype_margin_loss', 0.0):.4f} "
                f"train_cons={train_metrics.get('consistency_loss', 0.0):.4f} "
                f"train_semcons={train_metrics.get('semantic_consistency_loss', 0.0):.4f} "
                f"train_freg={train_metrics.get('fusion_reg_loss', 0.0):.4f} "
                f"train_semfloor={train_metrics.get('semantic_floor_loss', 0.0):.4f} "
                f"train_rnpen={train_metrics.get('real_noise_penalty_loss', 0.0):.4f} "
                f"train_rsfloor={train_metrics.get('real_semantic_floor_loss', 0.0):.4f} "
                f"train_asfloor={train_metrics.get('aigc_semantic_floor_loss', 0.0):.4f} "
                f"train_alpha_split=({train_metrics.get('alpha_mean_real', 0.0):.3f}/"
                f"{train_metrics.get('alpha_mean_easy_aigc', 0.0):.3f}/"
                f"{train_metrics.get('alpha_mean_hard_aigc', 0.0):.3f}) "
                f"train_rpush=({train_metrics.get('residual_push_real_mean', 0.0):.3f}/"
                f"{train_metrics.get('residual_push_hard_aigc_mean', 0.0):.3f}) "
                f"train_gate=({train_metrics.get('raw_semantic_mean', 0.0):.3f}/"
                f"{train_metrics.get('raw_frequency_mean', 0.0):.3f}/"
                f"{train_metrics.get('raw_noise_mean', 0.0):.3f}) "
                f"train_fusion=({train_metrics.get('fusion_semantic_mean', 0.0):.3f}/"
                f"{train_metrics.get('fusion_frequency_mean', 0.0):.3f}/"
                f"{train_metrics.get('fusion_noise_mean', 0.0):.3f}) "
                f"train_auroc={train_metrics['auroc']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_bce={val_metrics.get('main_bce_loss', 0.0):.4f} "
                f"val_basebce={val_metrics.get('base_bce_loss', 0.0):.4f} "
                f"val_hreal={val_metrics.get('hard_real_margin_loss', 0.0):.4f} "
                f"val_proto={val_metrics.get('prototype_margin_loss', 0.0):.4f} "
                f"val_freg={val_metrics.get('fusion_reg_loss', 0.0):.4f} "
                f"val_semfloor={val_metrics.get('semantic_floor_loss', 0.0):.4f} "
                f"val_rnpen={val_metrics.get('real_noise_penalty_loss', 0.0):.4f} "
                f"val_rsfloor={val_metrics.get('real_semantic_floor_loss', 0.0):.4f} "
                f"val_asfloor={val_metrics.get('aigc_semantic_floor_loss', 0.0):.4f} "
                f"val_alpha_split=({val_metrics.get('alpha_mean_real', 0.0):.3f}/"
                f"{val_metrics.get('alpha_mean_easy_aigc', 0.0):.3f}/"
                f"{val_metrics.get('alpha_mean_hard_aigc', 0.0):.3f}) "
                f"val_rpush=({val_metrics.get('residual_push_real_mean', 0.0):.3f}/"
                f"{val_metrics.get('residual_push_hard_aigc_mean', 0.0):.3f}) "
                f"val_gate=({val_metrics.get('raw_semantic_mean', 0.0):.3f}/"
                f"{val_metrics.get('raw_frequency_mean', 0.0):.3f}/"
                f"{val_metrics.get('raw_noise_mean', 0.0):.3f}) "
                f"val_fusion=({val_metrics.get('fusion_semantic_mean', 0.0):.3f}/"
                f"{val_metrics.get('fusion_frequency_mean', 0.0):.3f}/"
                f"{val_metrics.get('fusion_noise_mean', 0.0):.3f}) "
                f"val_auroc={val_metrics['auroc']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"val_basef1={val_metrics.get('base_f1', 0.0):.4f} "
                f"val_gap=({val_metrics.get('base_vs_final_gap_real', 0.0):.3f}/"
                f"{val_metrics.get('base_vs_final_gap_aigc', 0.0):.3f}) "
                f"logit_sep={val_metrics.get('logit_separation', 0.0):.4f} "
                f"collapse={int(val_metrics.get('collapse_flag', 0))} "
                f"T={val_metrics['temperature']:.3f}"
            )
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
