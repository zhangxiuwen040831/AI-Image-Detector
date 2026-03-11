import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .losses import DetectionLoss
from .metrics import compute_metrics


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
        self.loss_fn = DetectionLoss()
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
        self.model.train()
        losses = []
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
                loss = self.loss_fn(out["logit"], smooth_labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            probs = out["probability"].detach().view(-1).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.detach().view(-1).cpu().numpy().tolist())
            losses.append(float(loss.detach().cpu().item()))

        elapsed = max(time.time() - start, 1e-6)
        throughput = float(len(y_true) / elapsed)
        metrics = compute_metrics(np.array(y_true), np.array(y_prob))
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        metrics["throughput_img_s"] = throughput
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics

    @torch.no_grad()
    def validate_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        losses = []
        y_true, y_prob = [], []
        for batch in tqdm(loader, desc=f"Val {epoch}", leave=False):
            images, labels, _ = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            smooth_labels = self._smooth_labels(labels)
            out = self.model(images)
            loss = self.loss_fn(out["logit"], smooth_labels)
            probs = out["probability"].detach().view(-1).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.detach().view(-1).cpu().numpy().tolist())
            losses.append(float(loss.detach().cpu().item()))

        metrics = compute_metrics(np.array(y_true), np.array(y_prob))
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        return metrics

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_metrics": val_metrics,
        }
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
