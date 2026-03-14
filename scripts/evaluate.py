import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_image_detector.data import ImageDataset, build_val_transforms
from ai_image_detector.models import MultiBranchDetector
from ai_image_detector.evaluation.metrics import compute_binary_metrics as compute_metrics
from ai_image_detector.utils import load_config, setup_logger


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    base_config_path = ROOT / "configs" / "base_config.yaml"
    cfg = load_config(
        str(config_path),
        str(base_config_path) if config_path.name != "base_config.yaml" else None,
    )
    logger = setup_logger("evaluate")
    device = build_device(cfg["system"]["device"])

    dataset = ImageDataset(
        root_dir=cfg["data"]["val_dir"],
        transform=build_val_transforms(int(cfg["data"]["image_size"])),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["system"]["num_workers"]),
        pin_memory=bool(cfg["system"]["pin_memory"]),
    )

    model = MultiBranchDetector(
        rgb_backbone=cfg["model"]["backbone"],
        rgb_pretrained=False,
        noise_pretrained=False,
        freq_pretrained=False,
        fused_dim=int(cfg["model"]["fused_dim"]),
        classifier_hidden_dim=int(cfg["model"]["classifier_hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, y_prob = [], []
    for images, labels in loader:
        images = images.to(device)
        out = model(images)
        probs = out["probability"].view(-1).detach().cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().tolist())

    metrics = compute_metrics(np.array(y_true), np.array(y_prob))
    logger.info(
        f"Evaluation | accuracy={metrics['accuracy']:.4f} auc={metrics['auc']:.4f}"
    )


if __name__ == "__main__":
    main()
