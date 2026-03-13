from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire.dataset import NTIRETrainDataset


def make_tiny_subset(
    data_root: Path,
    out_csv: Path,
    subset_size: int = 500,
    seed: int = 42,
) -> None:
    ds = NTIRETrainDataset(root_dir=data_root, transform=None, strict=False)
    df = ds.to_dataframe()
    rng = random.Random(seed)
    real_idx = df.index[df["label"] == 0].tolist()
    fake_idx = df.index[df["label"] == 1].tolist()
    rng.shuffle(real_idx)
    rng.shuffle(fake_idx)
    half = subset_size // 2
    n_real = min(len(real_idx), half)
    n_fake = min(len(fake_idx), subset_size - n_real)
    if n_real + n_fake < subset_size:
        remaining = subset_size - (n_real + n_fake)
        pool = [i for i in df.index.tolist() if i not in set(real_idx[:n_real] + fake_idx[:n_fake])]
        rng.shuffle(pool)
        extra = pool[:remaining]
    else:
        extra = []
    chosen = real_idx[:n_real] + fake_idx[:n_fake] + extra
    rng.shuffle(chosen)
    subset = df.loc[chosen].copy()
    subset["subset_index"] = list(range(len(subset)))
    subset = subset[
        ["subset_index", "image_path", "label", "shard_name"]
        + [c for c in subset.columns if c not in {"subset_index", "image_path", "label", "shard_name"}]
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_csv, index=False)
    label_counts = subset["label"].value_counts().to_dict()
    print(f"Saved tiny subset to: {out_csv}")
    print(f"Total samples: {len(subset)}")
    print(f"Label counts: {label_counts}")
    print(subset.head(8).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\32902\Desktop\ai-image-detector\NTIRE-RobustAIGenDetection-train"),
    )
    parser.add_argument("--out-csv", type=Path, default=PROJECT_ROOT / "tiny_subset_500.csv")
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_tiny_subset(
        data_root=args.data_root,
        out_csv=args.out_csv,
        subset_size=args.subset_size,
        seed=args.seed,
    )
