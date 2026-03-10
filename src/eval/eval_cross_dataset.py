import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from src.datasets.aigibench import AIGIBenchDataset
from src.datasets.chameleon import ChameleonDataset
from src.datasets.genimage import GenImageDataset
from src.eval.metrics import compute_binary_metrics, summarize_metrics
from src.models.detector_model import DetectorModel, build_detector_from_config


def load_model(ckpt_path: str, device: torch.device) -> Tuple[DetectorModel, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg: Dict[str, Any] = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {})

    model = build_detector_from_config(
        device=str(device),
        train_backbone=False,
        head_hidden_dim=model_cfg.get("head_hidden_dim"),
        head_dropout=model_cfg.get("head_dropout", 0.0),
        backbone_name="ViT-L-14",
        backbone_pretrained="openai",
        use_lora=model_cfg.get("use_lora", False),
        lora_rank=model_cfg.get("lora_rank", 8),
        lora_alpha=model_cfg.get("lora_alpha", 16.0),
        lora_dropout=model_cfg.get("lora_dropout", 0.0),
        use_osd=model_cfg.get("use_osd", False),
        osd_proj_dim=model_cfg.get("osd_proj_dim", 128),
        osd_lambda_orth=model_cfg.get("osd_lambda_orth", 1e-3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def eval_on_loader(
    model: DetectorModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    y_true: List[int] = []
    y_prob: List[float] = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = outputs["probs"]

        y_true.extend(labels.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

    return compute_binary_metrics(y_true, y_prob)


def build_loaders(image_size: int, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}

    loaders["genimage_val"] = DataLoader(
        GenImageDataset(split="val", image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    loaders["aigibench_val"] = DataLoader(
        AIGIBenchDataset(split="val", image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    loaders["chameleon_test"] = DataLoader(
        ChameleonDataset(split="test", image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loaders


def main(ckpt_path: str, output_path: str, batch_size: int, num_workers: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(ckpt_path, device)

    image_size = cfg.get("data", {}).get("image_size", 224)

    loaders = build_loaders(
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    per_dataset_results: Dict[str, Dict[str, float]] = {}
    for name, loader in loaders.items():
        print(f"[INFO] Evaluating on {name} ...")
        per_dataset_results[name] = eval_on_loader(model, loader, device)
        print(f"  {name} metrics: {per_dataset_results[name]}")

    mean_metrics, _ = summarize_metrics(per_dataset_results)
    print(f"[INFO] Mean cross-dataset metrics: {mean_metrics}")

    out = {
        "checkpoint": ckpt_path,
        "mean_metrics": mean_metrics,
        "per_dataset": per_dataset_results,
    }
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved evaluation report to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/cross_dataset_eval.json",
        help="Path to output JSON report",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args.ckpt, args.output, args.batch_size, args.num_workers)

