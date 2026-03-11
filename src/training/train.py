import argparse
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader_builder import build_train_loader, build_val_loader
from src.models.model import AIGCImageDetector
from src.training.trainer import Trainer
from src.utils.logger import setup_logger


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def patch_runtime_dataset_root(cfg: dict) -> dict:
    data_cfg = cfg["data"]
    if "runtime_dataset_root" in data_cfg and data_cfg["runtime_dataset_root"]:
        return cfg
    if platform.system().lower().startswith("win"):
        data_cfg["runtime_dataset_root"] = data_cfg["local_dataset_root"]
    else:
        data_cfg["runtime_dataset_root"] = data_cfg["server_dataset_root"]
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "dataset_config.yaml"),
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    cfg = patch_runtime_dataset_root(cfg)
    set_seed(int(cfg["train"].get("seed", 42)))
    device = resolve_device(cfg["train"].get("device", "cuda"))
    logger = setup_logger("train", str(ROOT / "logs" / "pipeline_train.log"))
    logger.info(f"Using device: {device}")
    logger.info(f"Dataset root: {cfg['data']['runtime_dataset_root']}")

    model = AIGCImageDetector(cfg["model"]).to(device)
    # Check model device
    model_device = next(model.parameters()).device
    logger.info(f"Model is on device: {model_device}")
    if device.type == "cuda" and model_device.type != "cuda":
        logger.error("FAILED to move model to CUDA!")
    
    trainer = Trainer(model=model, config=cfg, device=device, logger=logger)
    train_loader = build_train_loader(args.config, batch_size=int(cfg["loader"]["batch_size"]))
    val_loader = build_val_loader(args.config, batch_size=int(cfg["loader"]["batch_size"]))
    trainer.fit(train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()
