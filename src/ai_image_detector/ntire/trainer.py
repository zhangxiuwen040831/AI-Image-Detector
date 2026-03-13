from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
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
        mixup_alpha: float = 1.0,
        mixup_prob: float = 0.0,
        max_checkpoint_keep: int = 3,
    ) -> None:
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
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        self.use_ema = use_ema
        self.model_ema = ModelEma(self.model, decay=ema_decay) if use_ema else None
        self.temperature_scaler = TemperatureScaler(init_temperature=1.0).to(device)
        self.best_val_auroc = -1.0
        self.start_epoch = 1

    def _lr_lambda(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return max((epoch + 1) / max(self.warmup_epochs, 1), 1e-3)
        progress = (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(float(cosine), 1e-4)

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False) -> None:
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
        if is_best:
            torch.save(ckpt, self.save_dir / "best.pth")
        all_ckpts = sorted(self.save_dir.glob("epoch_*.pth"))
        if len(all_ckpts) > self.max_checkpoint_keep:
            for p in all_ckpts[: len(all_ckpts) - self.max_checkpoint_keep]:
                p.unlink(missing_ok=True)

    def resume(self, checkpoint_path: Path) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        temp = float(ckpt.get("temperature", 1.0))
        with torch.no_grad():
            self.temperature_scaler.log_temperature.copy_(torch.log(torch.tensor([temp], device=self.device)))
        if self.model_ema is not None and "ema_shadow" in ckpt:
            self.model_ema.shadow = {k: v.to(self.device) for k, v in ckpt["ema_shadow"].items()}
        self.start_epoch = int(ckpt["epoch"]) + 1
        metrics = ckpt.get("val_metrics", {})
        self.best_val_auroc = float(metrics.get("auroc", self.best_val_auroc))

    def _forward_loss(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        out = self.model(images)
        loss_dict = self.loss_fn(out, labels)
        return out, loss_dict

    def _set_aux_weight_for_epoch(self, epoch: int) -> None:
        self.loss_fn.aux_weight = 0.5 if epoch <= 4 else self.base_aux_weight

    def _mixup(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mixup_prob <= 0.0 or self.mixup_alpha <= 0.0:
            return images, labels
        if torch.rand(1, device=images.device).item() > self.mixup_prob:
            return images, labels
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        idx = torch.randperm(images.size(0), device=images.device)
        mixed_images = lam * images + (1.0 - lam) * images[idx]
        mixed_labels = lam * labels + (1.0 - lam) * labels[idx]
        return mixed_images, mixed_labels

    def train_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        self._set_aux_weight_for_epoch(epoch)
        epoch_loss = []
        y_true = []
        y_prob = []
        start = time.time()
        pbar = tqdm(loader, desc=f"Train {epoch}", leave=False)
        for images, labels, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            images, labels_mix = self._mixup(images, labels)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                out, loss_dict = self._forward_loss(images, labels_mix)
                loss = loss_dict["total_loss"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.model_ema is not None:
                self.model_ema.update(self.model)
            prob = torch.sigmoid(out["logit"]).detach().view(-1).cpu().numpy().tolist()
            y_prob.extend(prob)
            y_true.extend((labels.detach().view(-1) >= 0.5).float().cpu().numpy().tolist())
            epoch_loss.append(float(loss.detach().cpu().item()))
            pbar.set_postfix({"loss": f"{epoch_loss[-1]:.4f}"})
        elapsed = max(time.time() - start, 1e-6)
        metrics = compute_metrics(y_true=y_true, y_prob=y_prob)
        metrics["loss"] = float(np.mean(epoch_loss)) if epoch_loss else 0.0
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        metrics["img_s"] = float(len(y_true) / elapsed)
        return metrics

    @torch.no_grad()
    def validate_epoch(self, loader, epoch: int, use_ema: bool = True) -> Dict[str, float]:
        self.loss_fn.aux_weight = self.base_aux_weight
        if use_ema and self.model_ema is not None:
            self.model_ema.apply_shadow(self.model)
        self.model.eval()
        epoch_loss = []
        y_true = []
        y_prob = []
        logits_all = []
        labels_all = []
        for images, labels, _ in tqdm(loader, desc=f"Val {epoch}", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            out, loss_dict = self._forward_loss(images, labels)
            logits = out["logit"].detach()
            scaled_logits = self.temperature_scaler(logits)
            probs = torch.sigmoid(scaled_logits).view(-1)
            y_prob.extend(probs.cpu().numpy().tolist())
            y_true.extend(labels.detach().view(-1).cpu().numpy().tolist())
            epoch_loss.append(float(loss_dict["total_loss"].detach().cpu().item()))
            logits_all.append(logits.cpu())
            labels_all.append(labels.detach().view(-1, 1).float().cpu())
        if logits_all and labels_all:
            fit_logits = torch.cat(logits_all, dim=0).to(self.device)
            fit_labels = torch.cat(labels_all, dim=0).to(self.device)
            self.temperature_scaler.fit(fit_logits, fit_labels)
        metrics = compute_metrics(y_true=y_true, y_prob=y_prob)
        metrics["loss"] = float(np.mean(epoch_loss)) if epoch_loss else 0.0
        metrics["temperature"] = self.temperature_scaler.value()
        if use_ema and self.model_ema is not None:
            self.model_ema.restore(self.model)
        return metrics

    def fit(self, train_loader, val_loader, log_path: Optional[Path] = None) -> None:
        history = []
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate_epoch(val_loader, epoch, use_ema=True)
            self.scheduler.step()
            is_best = float(val_metrics.get("auroc", -1.0)) > self.best_val_auroc
            if is_best:
                self.best_val_auroc = float(val_metrics["auroc"])
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(row)
            print(
                f"Epoch {epoch} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_auroc={train_metrics['auroc']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_auroc={val_metrics['auroc']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"T={val_metrics['temperature']:.3f}"
            )
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
