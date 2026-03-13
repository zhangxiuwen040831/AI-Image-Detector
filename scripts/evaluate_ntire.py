from __future__ import annotations

import argparse
import io
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire.augmentations import build_eval_transform
from ai_image_detector.ntire.dataset import NTIRETrainDataset, build_train_val_indices
from ai_image_detector.ntire.metrics import compute_metrics
from ai_image_detector.ntire.model import HybridAIGCDetector


class EvalDataset(Dataset):
    def __init__(
        self,
        base_dataset: NTIRETrainDataset,
        image_size: int,
        corruption: Optional[str] = None,
        severity: Optional[float] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.corruption = corruption
        self.severity = severity
        self.transform = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _apply_corruption(self, image: Image.Image) -> Image.Image:
        if self.corruption is None:
            return image
        if self.corruption == "jpeg":
            quality = int(self.severity) if self.severity is not None else 45
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        if self.corruption == "blur":
            sigma = float(self.severity) if self.severity is not None else 1.2
            return image.filter(ImageFilter.GaussianBlur(radius=sigma))
        if self.corruption == "resize":
            w, h = image.size
            down = image.resize((max(w // 2, 32), max(h // 2, 32)), Image.BILINEAR)
            return down.resize((w, h), Image.BICUBIC)
        return image

    def __getitem__(self, index: int):
        _, label, metadata = self.base_dataset[index]
        image = Image.open(self.base_dataset.records[index].image_path).convert("RGB")
        image = self._apply_corruption(image)
        tensor = self.transform(image=np.array(image))["image"]
        return tensor, label, metadata


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_shards(shards: Optional[str]) -> Optional[List[int]]:
    if not shards:
        return None
    return [int(x.strip()) for x in shards.split(",") if x.strip()]


def load_model(ckpt_path: Path, backbone_name: str, device: torch.device):
    model = HybridAIGCDetector(backbone_name=backbone_name, pretrained_backbone=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    temperature = float(ckpt.get("temperature", 1.0))
    return model, temperature


@torch.no_grad()
def evaluate_loader(model, loader, device, temperature: float) -> Dict[str, float]:
    y_true = []
    y_prob = []
    for images, labels, _ in loader:
        logits = model(images.to(device))["logit"] / max(temperature, 1e-6)
        probs = torch.sigmoid(logits).view(-1).cpu().numpy().tolist()
        y_prob.extend(probs)
        y_true.extend(labels.numpy().tolist())
    return compute_metrics(y_true=y_true, y_prob=y_prob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"C:\Users\32902\Desktop\ai-image-detector\NTIRE-RobustAIGenDetection-train",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--shards", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=str, default=str(PROJECT_ROOT / "eval_ntire_report.csv"))
    parser.add_argument("--include-resize-check", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    dataset = NTIRETrainDataset(root_dir=args.data_root, shard_ids=parse_shards(args.shards), transform=None)
    _, val_indices, val_mode = build_train_val_indices(dataset, val_ratio=args.val_ratio, seed=args.seed)
    model, temperature = load_model(Path(args.checkpoint), args.backbone_name, device)

    results = []
    eval_conditions = [("clean", None, None)]
    eval_conditions.extend([(f"jpeg_q{q}", "jpeg", float(q)) for q in range(30, 91, 10)])
    eval_conditions.extend([(f"blur_s{sigma:.1f}", "blur", float(sigma)) for sigma in [0.5, 1.0, 1.5, 2.0]])
    if args.include_resize_check:
        eval_conditions.append(("resize_x0.5", "resize", 0.5))
    for condition_name, corruption, severity in eval_conditions:
        eval_ds = EvalDataset(dataset, image_size=args.image_size, corruption=corruption, severity=severity)
        loader = DataLoader(
            Subset(eval_ds, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        m = evaluate_loader(model, loader, device, temperature=temperature)
        m["condition"] = condition_name
        m["corruption_type"] = "clean" if corruption is None else corruption
        m["severity"] = "" if severity is None else severity
        m["val_mode"] = val_mode
        results.append(m)
        print(
            f"{condition_name}: "
            f"auroc={m['auroc']:.4f} auprc={m['auprc']:.4f} "
            f"f1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} ece={m['ece']:.4f}"
        )

    df_val = dataset.to_dataframe().iloc[val_indices].copy()
    group_cols = [c for c in df_val.columns if "distort" in str(c).lower() or "source" in str(c).lower()]
    if group_cols:
        col = group_cols[0]
        clean_ds = EvalDataset(dataset, image_size=args.image_size, corruption=None, severity=None)
        for g, gdf in df_val.groupby(col):
            loader = DataLoader(
                Subset(clean_ds, gdf.index.tolist()),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            m = evaluate_loader(model, loader, device, temperature=temperature)
            m["condition"] = f"group:{col}={g}"
            m["val_mode"] = val_mode
            results.append(m)

    out_csv = Path(args.out_csv)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved report: {out_csv}")


if __name__ == "__main__":
    main()
