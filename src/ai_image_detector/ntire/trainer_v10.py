from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from .metrics import compute_metrics


class V10Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: str | Path,
        epochs: int,
        lr: float = 5e-5,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        base_bce_weight: float = 1.0,
        semantic_aux_weight: float = 0.20,
        frequency_aux_weight: float = 0.20,
        noise_aux_weight: float = 0.05,
        hard_real_margin_weight: float = 0.10,
        hard_real_margin: float = 0.25,
        anchor_real_margin_weight: float = 0.14,
        anchor_real_margin: float = 0.30,
        prototype_margin_weight: float = 0.05,
        prototype_margin: float = 0.15,
        fragile_aigc_weight: float = 0.05,
        fragile_aigc_target_prob: float = 0.20,
        hybrid_main_weight: float = 1.0,
        base_support_weight: float = 0.50,
        checkpoint_interval: int = 1,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.base_bce_weight = float(base_bce_weight)
        self.semantic_aux_weight = float(semantic_aux_weight)
        self.frequency_aux_weight = float(frequency_aux_weight)
        self.noise_aux_weight = float(noise_aux_weight)
        self.hard_real_margin_weight = float(hard_real_margin_weight)
        self.hard_real_margin = float(hard_real_margin)
        self.anchor_real_margin_weight = float(anchor_real_margin_weight)
        self.anchor_real_margin = float(anchor_real_margin)
        self.prototype_margin_weight = float(prototype_margin_weight)
        self.prototype_margin = float(prototype_margin)
        self.fragile_aigc_weight = float(fragile_aigc_weight)
        self.fragile_aigc_target_prob = float(fragile_aigc_target_prob)
        self.hybrid_main_weight = float(hybrid_main_weight)
        self.base_support_weight = float(base_support_weight)
        self.checkpoint_interval = max(int(checkpoint_interval), 1)
        self.phase = "phase1_warmup"
        self.current_epoch = 0
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.best_selection_key: Optional[Tuple[float, ...]] = None
        self.best_selection_summary: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        self.optimizer = self._build_optimizer()

    def _unwrap_model(self) -> nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def _build_optimizer(self) -> AdamW:
        params = [param for param in self.model.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters were found for V10 trainer.")
        return AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def refresh_optimizer(self) -> None:
        self.optimizer = self._build_optimizer()

    def set_phase(self, phase: str, semantic_trainable_layers: int = 0) -> Dict[str, Any]:
        base_model = self._unwrap_model()
        if not hasattr(base_model, "configure_phase"):
            raise RuntimeError("V10 model must expose configure_phase().")
        summary = base_model.configure_phase(
            phase=phase,
            semantic_trainable_layers=semantic_trainable_layers,
        )
        self.phase = str(phase)
        self.refresh_optimizer()
        return summary

    def reset_tracking(self, clear_history: bool = False) -> None:
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.best_selection_key = None
        self.best_selection_summary = None
        if clear_history:
            self.history = []

    def _metadata_boolean_mask(
        self,
        metadata: Optional[Dict[str, Any]],
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
            normalized: List[float] = []
            for value in values:
                if isinstance(value, str):
                    normalized.append(1.0 if value.lower() in {"1", "true", "yes"} else 0.0)
                else:
                    normalized.append(float(bool(value)))
            return torch.tensor(normalized, device=self.device, dtype=torch.float32).view(-1)
        return torch.zeros(batch_size, device=self.device, dtype=torch.float32)

    @staticmethod
    def _weighted_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.view(-1)
        mask = mask.view(-1).float()
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom

    def _prototype_margin_loss(
        self,
        base_feat: torch.Tensor,
        labels: torch.Tensor,
        hard_real_mask: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.view(-1).float()
        hard_real_mask = hard_real_mask.view(-1).float()
        real_mask = labels < 0.5
        aigc_mask = labels >= 0.5
        active_hard_real = hard_real_mask > 0.5
        if (not torch.any(real_mask)) or (not torch.any(aigc_mask)) or (not torch.any(active_hard_real)):
            return base_feat.new_zeros(())
        feat = F.normalize(base_feat.float(), dim=1)
        real_proto = F.normalize(feat[real_mask].mean(dim=0, keepdim=True), dim=1)
        aigc_proto = F.normalize(feat[aigc_mask].mean(dim=0, keepdim=True), dim=1)
        hard_feat = feat[active_hard_real]
        sim_real = torch.matmul(hard_feat, real_proto.t()).view(-1)
        sim_aigc = torch.matmul(hard_feat, aigc_proto.t()).view(-1)
        return torch.relu(base_feat.new_tensor(self.prototype_margin) - (sim_real - sim_aigc)).mean()

    def _set_runtime_train_mode(self) -> None:
        self.model.train()
        base_model = self._unwrap_model()
        module_names = (
            "semantic_branch",
            "frequency_branch",
            "noise_branch",
            "primary_fusion",
            "base_classifier",
            "semantic_head",
            "frequency_head",
            "noise_proj",
            "noise_delta_norm",
            "noise_delta_head",
            "noise_controller",
            "noise_head",
        )
        for module_name in module_names:
            module = getattr(base_model, module_name, None)
            if module is None:
                continue
            params = list(module.parameters())
            if params and any(param.requires_grad for param in params):
                module.train()
            else:
                module.eval()

    def _forward_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        out = self.model(images)
        labels_2d = labels.view(-1, 1).float()
        labels_flat = labels_2d.view(-1)
        base_logit = out["base_logit"]
        semantic_logit = out["semantic_logit"]
        frequency_logit = out["freq_logit"]
        main_logit = out["logit"]
        hard_real_mask = self._metadata_boolean_mask(metadata, key="hard_real_buffer_hit", batch_size=labels.shape[0])
        hard_real_real_mask = hard_real_mask * (labels_flat < 0.5).float()
        anchor_hard_real_mask = self._metadata_boolean_mask(
            metadata,
            key="anchor_hard_real_buffer_hit",
            batch_size=labels.shape[0],
        ) * (labels_flat < 0.5).float()
        fragile_aigc_mask = self._metadata_boolean_mask(
            metadata,
            key="fragile_aigc_buffer_hit",
            batch_size=labels.shape[0],
        ) * (labels_flat >= 0.5).float()

        base_bce_loss = self.base_bce_weight * F.binary_cross_entropy_with_logits(base_logit, labels_2d)
        semantic_aux_loss = self.semantic_aux_weight * F.binary_cross_entropy_with_logits(semantic_logit, labels_2d)
        frequency_aux_loss = self.frequency_aux_weight * F.binary_cross_entropy_with_logits(frequency_logit, labels_2d)
        hard_real_margin_loss = self.hard_real_margin_weight * self._weighted_mean(
            torch.relu(base_logit.view(-1) + self.hard_real_margin),
            hard_real_real_mask,
        )
        anchor_real_margin_loss = self.anchor_real_margin_weight * self._weighted_mean(
            torch.relu(base_logit.view(-1) + self.anchor_real_margin),
            anchor_hard_real_mask,
        )
        prototype_margin_loss = self.prototype_margin_weight * self._prototype_margin_loss(
            base_feat=out["base_feat"],
            labels=labels_flat,
            hard_real_mask=hard_real_real_mask,
        )
        fragile_aigc_support_loss = self.fragile_aigc_weight * self._weighted_mean(
            torch.relu(
                base_logit.new_tensor(self.fragile_aigc_target_prob) - torch.sigmoid(base_logit.view(-1))
            ),
            fragile_aigc_mask,
        )

        if self.phase == "phase3_competition" and getattr(self._unwrap_model(), "inference_mode", "base_only") == "hybrid_optional":
            hybrid_bce_loss = self.hybrid_main_weight * F.binary_cross_entropy_with_logits(main_logit, labels_2d)
            base_support_loss = self.base_support_weight * F.binary_cross_entropy_with_logits(base_logit, labels_2d)
            noise_head_loss = self.noise_aux_weight * F.binary_cross_entropy_with_logits(
                out["noise_delta_logit"],
                labels_2d,
            )
            total_loss = (
                hybrid_bce_loss
                + base_support_loss
                + semantic_aux_loss
                + frequency_aux_loss
                + noise_head_loss
                + hard_real_margin_loss
                + anchor_real_margin_loss
                + prototype_margin_loss
                + fragile_aigc_support_loss
            )
            main_bce_loss = hybrid_bce_loss
        else:
            noise_head_loss = labels_2d.new_zeros(())
            total_loss = (
                base_bce_loss
                + semantic_aux_loss
                + frequency_aux_loss
                + hard_real_margin_loss
                + anchor_real_margin_loss
                + prototype_margin_loss
                + fragile_aigc_support_loss
            )
            main_bce_loss = base_bce_loss

        sf_weights = out["sf_weights"]
        alpha_ratio = out.get("alpha_ratio_used", out.get("alpha_ratio", labels_2d.new_zeros(labels_2d.shape)))
        loss_dict = {
            "total_loss": total_loss,
            "main_bce_loss": main_bce_loss,
            "base_bce_loss": base_bce_loss,
            "semantic_aux_loss": semantic_aux_loss,
            "frequency_aux_loss": frequency_aux_loss,
            "noise_aux_loss": noise_head_loss,
            "hard_real_margin_loss": hard_real_margin_loss,
            "anchor_real_margin_loss": anchor_real_margin_loss,
            "prototype_margin_loss": prototype_margin_loss,
            "fragile_aigc_support_loss": fragile_aigc_support_loss,
            "hard_real_count": hard_real_real_mask.sum(),
            "anchor_hard_real_count": anchor_hard_real_mask.sum(),
            "fragile_aigc_count": fragile_aigc_mask.sum(),
            "base_logit_mean": base_logit.mean(),
            "final_logit_mean": main_logit.mean(),
            "sf_semantic_mean": sf_weights[:, 0].mean(),
            "sf_frequency_mean": sf_weights[:, 1].mean(),
            "alpha_mean": alpha_ratio.view(-1).mean(),
        }
        return out, loss_dict

    @staticmethod
    def _to_float_metrics(loss_accumulator: Dict[str, List[float]]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for key, values in loss_accumulator.items():
            metrics[key] = float(np.mean(values)) if values else 0.0
        return metrics

    def _run_epoch(
        self,
        loader: Iterable,
        train: bool,
        epoch: int,
    ) -> Dict[str, float]:
        if train:
            self._set_runtime_train_mode()
        else:
            self.model.eval()

        loss_accumulator: Dict[str, List[float]] = {
            "loss": [],
            "main_bce_loss": [],
            "base_bce_loss": [],
            "semantic_aux_loss": [],
            "frequency_aux_loss": [],
            "noise_aux_loss": [],
            "hard_real_margin_loss": [],
            "anchor_real_margin_loss": [],
            "prototype_margin_loss": [],
            "fragile_aigc_support_loss": [],
            "hard_real_count": [],
            "anchor_hard_real_count": [],
            "fragile_aigc_count": [],
            "base_logit_mean": [],
            "final_logit_mean": [],
            "sf_semantic_mean": [],
            "sf_frequency_mean": [],
            "alpha_mean": [],
        }
        y_true: List[float] = []
        y_prob: List[float] = []
        y_prob_base: List[float] = []
        start = time.time()
        iterator = tqdm(loader, desc=f"{'Train' if train else 'Val'} {epoch}", leave=False)
        for batch in iterator:
            images, labels, metadata = batch[:3]
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                out, loss_dict = self._forward_loss(images=images, labels=labels, metadata=metadata)
                loss = loss_dict["total_loss"]
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    out, loss_dict = self._forward_loss(images=images, labels=labels, metadata=metadata)
                    loss = loss_dict["total_loss"]

            probs = torch.sigmoid(out["logit"].detach()).view(-1)
            base_probs = torch.sigmoid(out["base_logit"].detach()).view(-1)
            y_true.extend(labels.detach().view(-1).cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            y_prob_base.extend(base_probs.cpu().numpy().tolist())

            loss_accumulator["loss"].append(float(loss.detach().cpu().item()))
            for key in tuple(loss_accumulator.keys()):
                if key == "loss":
                    continue
                loss_accumulator[key].append(float(loss_dict[key].detach().cpu().item()))
            iterator.set_postfix({"loss": f"{loss_accumulator['loss'][-1]:.4f}"})

        elapsed = max(time.time() - start, 1e-6)
        metrics = compute_metrics(y_true=y_true, y_prob=y_prob)
        base_metrics = compute_metrics(y_true=y_true, y_prob=y_prob_base)
        metrics.update(self._to_float_metrics(loss_accumulator))
        metrics["base_precision"] = float(base_metrics.get("precision", 0.0))
        metrics["base_recall"] = float(base_metrics.get("recall", 0.0))
        metrics["base_f1"] = float(base_metrics.get("f1", 0.0))
        metrics["base_auroc"] = float(base_metrics.get("auroc", float("nan")))
        metrics["base_auprc"] = float(base_metrics.get("auprc", float("nan")))
        metrics["base_ece"] = float(base_metrics.get("ece", float("nan")))
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        metrics["img_s"] = float(len(y_true) / elapsed) if y_true else 0.0
        return metrics

    def train_epoch(self, loader: Iterable, epoch: int) -> Dict[str, float]:
        return self._run_epoch(loader=loader, train=True, epoch=epoch)

    @torch.no_grad()
    def validate_epoch(self, loader: Iterable, epoch: int) -> Dict[str, float]:
        return self._run_epoch(loader=loader, train=False, epoch=epoch)

    def _checkpoint_payload(
        self,
        epoch: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "epoch": int(epoch),
            "phase": self.phase,
            "model": self._unwrap_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
            "best_metric": float(self.best_metric),
            "best_epoch": int(self.best_epoch),
            "best_selection_key": None if self.best_selection_key is None else list(self.best_selection_key),
            "best_selection_summary": self.best_selection_summary,
        }
        if extra:
            payload.update(extra)
        return payload

    def save_checkpoint(
        self,
        epoch: int,
        name: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        path = self.save_dir / name
        torch.save(self._checkpoint_payload(epoch=epoch, extra=extra), path)
        return path

    def resume(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._unwrap_model().load_state_dict(checkpoint["model"], strict=False)
        self.refresh_optimizer()
        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except Exception:
                pass
        self.history = list(checkpoint.get("history", []))
        self.current_epoch = int(checkpoint.get("epoch", 0))
        self.best_metric = float(checkpoint.get("best_metric", float("-inf")))
        self.best_epoch = int(checkpoint.get("best_epoch", self.current_epoch))
        key = checkpoint.get("best_selection_key")
        self.best_selection_key = tuple(float(x) for x in key) if key is not None else None
        self.best_selection_summary = checkpoint.get("best_selection_summary")
        self.phase = str(checkpoint.get("phase", self.phase))
        return {
            "epoch": self.current_epoch,
            "phase": self.phase,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }

    def fit(
        self,
        train_loader: Iterable,
        val_loader: Iterable,
        start_epoch: int = 0,
        eval_callback: Optional[Any] = None,
        best_selector: Optional[Any] = None,
        early_stop_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        run_summary: Dict[str, Any] = {
            "phase": self.phase,
            "epochs": [],
        }
        no_improve_epochs = 0
        for epoch in range(start_epoch + 1, self.epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(loader=train_loader, epoch=epoch)
            val_metrics = self.validate_epoch(loader=val_loader, epoch=epoch)
            epoch_record = {
                "epoch": int(epoch),
                "phase": self.phase,
                "train": train_metrics,
                "val": val_metrics,
            }
            if eval_callback is not None:
                epoch_record["evaluation"] = eval_callback(epoch=epoch, trainer=self)
            self.history.append(epoch_record)
            run_summary["epochs"].append(epoch_record)

            selection_summary: Optional[Dict[str, Any]] = None
            if best_selector is not None:
                selection_summary = best_selector(epoch_record)
                if selection_summary is not None:
                    epoch_record["selection"] = selection_summary
            metric_key = "base_f1" if self.phase in {"phase1_warmup", "phase2_curriculum", "phase4_final_polish"} else "f1"
            score = float(val_metrics.get(metric_key, val_metrics.get("f1", 0.0)))
            selection_key: Tuple[float, ...]
            if selection_summary is not None and selection_summary.get("key") is not None:
                selection_key = tuple(float(x) for x in selection_summary["key"])
                metric_key = str(selection_summary.get("metric_key", "custom_selection"))
                score = float(selection_summary.get("score", score))
            else:
                selection_key = (score,)
            is_best = self.best_selection_key is None or selection_key > self.best_selection_key
            if is_best:
                self.best_selection_key = selection_key
                self.best_selection_summary = selection_summary
                self.best_metric = score
                self.best_epoch = int(epoch)
                no_improve_epochs = 0
                self.save_checkpoint(
                    epoch=epoch,
                    name="best.pth",
                    extra={
                        "metric_key": metric_key,
                        "metric_value": score,
                        "selection_key": list(selection_key),
                        "selection_summary": selection_summary,
                    },
                )
            else:
                no_improve_epochs += 1
            self.save_checkpoint(
                epoch=epoch,
                name="latest.pth",
                extra={
                    "metric_key": metric_key,
                    "metric_value": score,
                    "selection_key": list(selection_key),
                    "selection_summary": selection_summary,
                },
            )
            if epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    name=f"epoch_{epoch:03d}.pth",
                    extra={
                        "metric_key": metric_key,
                        "metric_value": score,
                        "selection_key": list(selection_key),
                        "selection_summary": selection_summary,
                    },
                )
            history_path = self.save_dir / "history.json"
            history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            print(
                f"[V10][{self.phase}] epoch={epoch} "
                f"train_loss={train_metrics.get('loss', 0.0):.4f} "
                f"val_base_f1={val_metrics.get('base_f1', 0.0):.4f} "
                f"val_base_auroc={val_metrics.get('base_auroc', 0.0):.4f} "
                f"hard_real={val_metrics.get('hard_real_count', 0.0):.1f} "
                f"anchor_real={val_metrics.get('anchor_hard_real_count', 0.0):.1f} "
                f"fragile_aigc={val_metrics.get('fragile_aigc_count', 0.0):.1f}"
            )
            if early_stop_fn is not None:
                stop_info = early_stop_fn(
                    epoch_record=epoch_record,
                    is_best=is_best,
                    no_improve_epochs=no_improve_epochs,
                    best_selection=self.best_selection_summary,
                )
                if stop_info:
                    run_summary["early_stop"] = stop_info
                    break
        run_summary["best_metric"] = float(self.best_metric)
        run_summary["best_epoch"] = int(self.best_epoch)
        run_summary["best_selection_key"] = None if self.best_selection_key is None else list(self.best_selection_key)
        run_summary["best_selection_summary"] = self.best_selection_summary
        run_summary["history_path"] = str(self.save_dir / "history.json")
        return run_summary


__all__ = ["V10Trainer"]
