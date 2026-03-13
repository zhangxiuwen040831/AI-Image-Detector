import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .calibration import TemperatureScaler
from .ema import ModelEma
from .losses import DetectionLoss
from ai_image_detector.evaluation.metrics import compute_binary_metrics as compute_metrics


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict,
        device: torch.device,
        logger,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger
        self.loss_fn = DetectionLoss(
            bce_weight=float(config["train"].get("bce_weight", 0.7)),
            focal_weight=float(config["train"].get("focal_weight", 0.3)),
            focal_alpha=float(config["train"].get("focal_alpha", 0.25)),
            focal_gamma=float(config["train"].get("focal_gamma", 2.0)),
            lambda_rgb=float(config["train"].get("lambda_rgb", 0.2)),
            lambda_freq=float(config["train"].get("lambda_freq", 0.2)),
            lambda_spatial=float(config["train"].get("lambda_spatial", 0.2)),
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(config["train"]["lr"]),
            weight_decay=float(config["train"]["weight_decay"]),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=int(config["train"]["epochs"]),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        self.save_dir = Path(config["train"]["save_dir"]).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.label_smoothing = float(config["train"].get("label_smoothing", 0.1))
        self.grad_clip = float(config["train"].get("grad_clip", 1.0))
        self.use_ema = bool(config["train"].get("use_ema", True))
        self.model_ema = ModelEma(self.model, decay=float(config["train"].get("ema_decay", 0.999)))
        self.temperature_scaler = TemperatureScaler(
            init_temperature=float(config["train"].get("temperature_init", 1.0))
        ).to(device)
        self.best_val_acc = 0.0

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1], {}
        raise RuntimeError("unexpected batch format")

    def _smooth_labels(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float().view(-1, 1)
        return labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

    def train_epoch(self, loader, epoch: int) -> Dict[str, float]:
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        self.model.train()
        losses = []
        main_losses = []
        rgb_losses = []
        freq_losses = []
        spatial_losses = []
        y_true, y_prob = [], []
        start = time.time()
        for batch in tqdm(loader, desc=f"Train {epoch}", leave=False):
            images, labels, _ = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            smooth_labels = self._smooth_labels(labels)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                out = self.model(images)
                loss_out = self.loss_fn(out, smooth_labels)
                loss = loss_out["total_loss"] if isinstance(loss_out, dict) else loss_out
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.use_ema:
                self.model_ema.update(self.model)

            probs = out["probability"].detach().view(-1).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.detach().view(-1).cpu().numpy().tolist())
            losses.append(float(loss.detach().cpu().item()))
            if isinstance(loss_out, dict):
                main_losses.append(float(loss_out["main_loss"].detach().cpu().item()))
                rgb_losses.append(float(loss_out["rgb_loss"].detach().cpu().item()))
                freq_losses.append(float(loss_out["freq_loss"].detach().cpu().item()))
                spatial_losses.append(float(loss_out["spatial_loss"].detach().cpu().item()))

        elapsed = max(time.time() - start, 1e-6)
        throughput = float(len(y_true) / elapsed)
        metrics = compute_metrics(np.array(y_true), np.array(y_prob))
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        metrics["main_loss"] = float(np.mean(main_losses)) if main_losses else 0.0
        metrics["rgb_loss"] = float(np.mean(rgb_losses)) if rgb_losses else 0.0
        metrics["freq_loss"] = float(np.mean(freq_losses)) if freq_losses else 0.0
        metrics["spatial_loss"] = float(np.mean(spatial_losses)) if spatial_losses else 0.0
        metrics["throughput_img_s"] = throughput
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics

    @torch.no_grad()
    def validate_epoch(self, loader, epoch: int) -> Dict[str, float]:
        if self.use_ema:
            self.model_ema.apply_shadow(self.model)
        self.model.eval()
        losses = []
        y_true, y_prob = [], []
        logits_buffer = []
        labels_buffer = []
        for batch in tqdm(loader, desc=f"Val {epoch}", leave=False):
            images, labels, _ = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            smooth_labels = self._smooth_labels(labels)
            out = self.model(images)
            loss_out = self.loss_fn(out, smooth_labels)
            loss = loss_out["total_loss"] if isinstance(loss_out, dict) else loss_out
            logits = out["logit"].detach()
            scaled_logits = self.temperature_scaler(logits)
            probs = torch.sigmoid(scaled_logits).view(-1).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.detach().view(-1).cpu().numpy().tolist())
            losses.append(float(loss.detach().cpu().item()))
            logits_buffer.append(logits.cpu())
            labels_buffer.append(labels.detach().view(-1, 1).float().cpu())

        metrics = compute_metrics(np.array(y_true), np.array(y_prob))
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        if len(logits_buffer) > 0 and len(labels_buffer) > 0:
            all_logits = torch.cat(logits_buffer, dim=0).to(self.device)
            all_labels = torch.cat(labels_buffer, dim=0).to(self.device)
            temperature = self.temperature_scaler.fit(all_logits, all_labels)
            metrics["temperature"] = float(temperature)
        else:
            metrics["temperature"] = float(self.temperature_scaler.value())
        if self.use_ema:
            self.model_ema.restore(self.model)
        return metrics

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_metrics": val_metrics,
            "temperature": float(self.temperature_scaler.value()),
        }
        if self.use_ema:
            checkpoint["ema_shadow"] = {k: v.cpu() for k, v in self.model_ema.shadow.items()}
        latest_path = self.save_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        if float(val_metrics.get("accuracy", 0.0)) >= self.best_val_acc:
            self.best_val_acc = float(val_metrics.get("accuracy", 0.0))
            torch.save(checkpoint, self.save_dir / "best.pth")

    def fit(self, train_loader, val_loader=None) -> None:
        epochs = int(self.config["train"]["epochs"])
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader, epoch)
            else:
                val_metrics = {"accuracy": 0.0, "auc": 0.0, "loss": 0.0}
            self.save_checkpoint(epoch, val_metrics)
            self.scheduler.step()
            self.logger.info(
                f"Epoch {epoch} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"train_auc={train_metrics['auc']:.4f} "
                f"img_s={train_metrics['throughput_img_s']:.2f} "
                f"lr={train_metrics['lr']:.8f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_auc={val_metrics['auc']:.4f}"
            )
