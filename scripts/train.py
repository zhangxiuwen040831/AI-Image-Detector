import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ImageDataset, build_train_transforms, build_val_transforms
from src.models import MultiBranchDetector
from src.training import Trainer
from src.utils import load_config, setup_logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    base_config_path = ROOT / "configs" / "base_config.yaml"
    cfg = load_config(
        str(config_path),
        str(base_config_path) if config_path.name != "base_config.yaml" else None,
    )

    logger = setup_logger(
        "train",
        str(Path(cfg["logging"]["log_dir"]) / "train.log"),
    )
    set_seed(int(cfg["system"]["seed"]))
    device = build_device(cfg["system"]["device"])
    logger.info(f"Using device: {device}")

    train_set = ImageDataset(
        root_dir=cfg["data"]["train_dir"],
        transform=build_train_transforms(int(cfg["data"]["image_size"])),
    )
    val_set = ImageDataset(
        root_dir=cfg["data"]["val_dir"],
        transform=build_val_transforms(int(cfg["data"]["image_size"])),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["system"]["num_workers"]),
        pin_memory=bool(cfg["system"]["pin_memory"]),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["system"]["num_workers"]),
        pin_memory=bool(cfg["system"]["pin_memory"]),
    )

    model = MultiBranchDetector(
        rgb_backbone=cfg["model"]["backbone"],
        rgb_pretrained=bool(cfg["model"]["rgb_pretrained"]),
        noise_pretrained=bool(cfg["model"]["noise_pretrained"]),
        freq_pretrained=bool(cfg["model"]["freq_pretrained"]),
        fused_dim=int(cfg["model"]["fused_dim"]),
        classifier_hidden_dim=int(cfg["model"]["classifier_hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )

    trainer = Trainer(model=model, config=cfg, device=device)
    epochs = int(cfg["training"]["epochs"])
    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate_epoch(val_loader, epoch)
        trainer.scheduler.step()
        trainer.save_checkpoint(epoch, val_metrics)
        logger.info(
            f"Epoch {epoch} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
