from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire.augmentations import build_eval_transform
from ai_image_detector.ntire.model import HybridAIGCDetector


def load_model(checkpoint: Path, device: torch.device, backbone_name: str) -> tuple[torch.nn.Module, float]:
    model = HybridAIGCDetector(backbone_name=backbone_name, pretrained_backbone=False)
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt["model"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    temperature = float(ckpt.get("temperature", 1.0))
    return model, temperature


def _predict_tensor(
    model: torch.nn.Module,
    image: Image.Image,
    scales: List[int],
    tta_flip: bool,
    temperature: float,
    device: torch.device,
) -> Dict[str, float]:
    logits = []
    fusion_weights = []
    base = image.convert("RGB")
    for sz in scales:
        transform = build_eval_transform(sz)
        arr = np.array(base)
        x = transform(image=arr)["image"].unsqueeze(0).to(device)
        out = model(x)
        logits.append(out["logit"].detach().cpu())
        fusion_weights.append(out["fusion_weights"].detach().cpu())
        if tta_flip:
            arr_flip = np.ascontiguousarray(np.fliplr(arr))
            xf = transform(image=arr_flip)["image"].unsqueeze(0).to(device)
            out_f = model(xf)
            logits.append(out_f["logit"].detach().cpu())
            fusion_weights.append(out_f["fusion_weights"].detach().cpu())
    mean_logit = torch.cat(logits, dim=0).mean(dim=0, keepdim=True)
    calibrated_logit = mean_logit / max(temperature, 1e-6)
    prob = torch.sigmoid(calibrated_logit).item()
    fw = torch.cat(fusion_weights, dim=0).mean(dim=0)
    return {
        "probability": float(prob),
        "raw_logit": float(mean_logit.item()),
        "calibrated_logit": float(calibrated_logit.item()),
        "fusion_semantic": float(fw[0].item()),
        "fusion_frequency": float(fw[1].item()),
        "fusion_noise": float(fw[2].item()),
    }


def infer_single(args: argparse.Namespace, model: torch.nn.Module, device: torch.device, temperature: float) -> None:
    image = Image.open(args.image).convert("RGB")
    pred = _predict_tensor(
        model=model,
        image=image,
        scales=args.scales,
        tta_flip=args.tta_flip,
        temperature=temperature,
        device=device,
    )
    label = int(pred["probability"] >= args.threshold)
    print(f"image={args.image}")
    print(f"probability={pred['probability']:.6f} predicted_label={label} threshold={args.threshold:.3f}")
    print(
        "fusion_weights="
        f"(semantic={pred['fusion_semantic']:.3f}, "
        f"frequency={pred['fusion_frequency']:.3f}, "
        f"noise={pred['fusion_noise']:.3f})"
    )


def infer_folder(args: argparse.Namespace, model: torch.nn.Module, device: torch.device, temperature: float) -> None:
    folder = Path(args.folder)
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        paths.extend(folder.rglob(ext))
    rows = []
    for p in sorted(paths):
        image = Image.open(p).convert("RGB")
        pred = _predict_tensor(
            model=model,
            image=image,
            scales=args.scales,
            tta_flip=args.tta_flip,
            temperature=temperature,
            device=device,
        )
        pred["path"] = str(p)
        pred["pred_label"] = int(pred["probability"] >= args.threshold)
        rows.append(pred)
    out_csv = Path(args.out_csv) if args.out_csv else folder / "inference_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved inference CSV: {out_csv}")
    print(f"Processed images: {len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--scales", type=int, nargs="+", default=[224, 336])
    parser.add_argument("--tta-flip", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image is None and args.folder is None:
        raise ValueError("Provide either --image or --folder")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, temperature = load_model(Path(args.checkpoint), device=device, backbone_name=args.backbone_name)
    if args.image is not None:
        infer_single(args, model, device, temperature)
    if args.folder is not None:
        infer_folder(args, model, device, temperature)


if __name__ == "__main__":
    main()
