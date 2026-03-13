from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire import NTIRETrainer
from ai_image_detector.ntire.augmentations import build_eval_transform, build_train_transform
from ai_image_detector.ntire.dataset import NTIRETrainDataset, build_train_val_indices, print_dataset_sanity
from ai_image_detector.ntire.metrics import compute_metrics
from ai_image_detector.ntire.model import HybridAIGCDetector


class TransformDataset(Dataset):
    def __init__(self, base_dataset: NTIRETrainDataset, transform) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        record = self.base_dataset.records[index]
        label = torch.tensor(record.label, dtype=torch.float32)
        metadata = dict(record.metadata)
        metadata["image_path"] = str(record.image_path)
        metadata["label"] = int(record.label)
        if self.transform is None:
            image = np.array(Image.open(record.image_path).convert("RGB"))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            return image, label, metadata
        arr = np.array(Image.open(record.image_path).convert("RGB"))
        image = self.transform(image=arr)["image"]
        return image, label, metadata


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_shards(shards: Optional[str]) -> Optional[List[int]]:
    if shards is None or shards.strip() == "":
        return None
    out = []
    for part in shards.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or None


def build_sampler_from_indices(dataset: NTIRETrainDataset, train_indices: List[int]) -> WeightedRandomSampler:
    df = dataset.to_dataframe().iloc[train_indices].reset_index(drop=True)
    label_counts = df["label"].value_counts().to_dict()
    label_weight = {k: 1.0 / max(v, 1) for k, v in label_counts.items()}
    candidate_cols = [c for c in df.columns if "distort" in str(c).lower() or "source" in str(c).lower()]
    group_col = candidate_cols[0] if candidate_cols else "shard_name"
    group_counts = df[group_col].value_counts().to_dict()
    group_weight = {k: 1.0 / max(v, 1) for k, v in group_counts.items()}
    weights = []
    for _, row in df.iterrows():
        weights.append(label_weight[row["label"]] * group_weight[row[group_col]])
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    pin_memory = torch.cuda.is_available()
    shard_ids = parse_shards(args.shards)
    base_ds = NTIRETrainDataset(root_dir=args.data_root, shard_ids=shard_ids, transform=None, strict=False)
    print_dataset_sanity(base_ds, max_rows=3)
    train_indices, val_indices, val_mode = build_train_val_indices(base_ds, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Validation mode: {val_mode}")
    train_ds = TransformDataset(
        base_ds,
        build_train_transform(
            image_size=args.image_size,
            jpeg_aligned_crop_p=args.jpeg_aligned_crop_p,
            enable_grayscale=not args.disable_to_gray,
            enable_defocus=not args.disable_defocus,
            enable_webp=not args.disable_webp_compression,
        ),
    )
    val_ds = TransformDataset(base_ds, build_eval_transform(args.image_size))
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)
    if args.use_balanced_sampler:
        sampler = build_sampler_from_indices(base_ds, train_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridAIGCDetector(
        backbone_name=args.backbone_name,
        pretrained_backbone=args.pretrained_backbone,
        backbone_trainable_layers=args.backbone_trainable_layers,
        image_size=args.image_size,
        fused_dim=args.fused_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        use_aux_heads=not args.disable_aux_heads,
        frequency_use_global_pool=not args.disable_freq_global_pool,
        fusion_gate_input_dropout=args.fusion_gate_input_dropout,
        fusion_feature_dropout=args.fusion_feature_dropout,
    )
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    trainer = NTIRETrainer(
        model=model,
        device=device,
        save_dir=Path(args.save_dir),
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        use_ema=not args.disable_ema,
        ema_decay=args.ema_decay,
        bce_weight=args.bce_weight,
        focal_weight=args.focal_weight,
        aux_weight=args.aux_weight,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
    )
    if args.resume is not None:
        trainer.resume(Path(args.resume))
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        log_path=Path(args.save_dir) / "history.json",
    )
    evaluate_distortion_groups(
        model=trainer.model,
        dataset=base_ds,
        val_indices=val_indices,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        temperature=trainer.temperature_scaler.value(),
    )


@torch.no_grad()
def evaluate_distortion_groups(
    model: torch.nn.Module,
    dataset: NTIRETrainDataset,
    val_indices: List[int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    temperature: float,
) -> None:
    pin_memory = torch.cuda.is_available()
    group_cols = [c for c in dataset.to_dataframe().columns if "distort" in str(c).lower() or "source" in str(c).lower()]
    if not group_cols:
        print("No distortion/source metadata found for grouped validation reporting.")
        return
    group_col = group_cols[0]
    df = dataset.to_dataframe().iloc[val_indices].copy()
    val_ds = TransformDataset(dataset, build_eval_transform(image_size))
    model.eval()
    for group_name, group_df in df.groupby(group_col):
        indices = group_df.index.tolist()
        loader = DataLoader(
            Subset(val_ds, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        y_true = []
        y_prob = []
        for images, labels, _ in loader:
            logits = model(images.to(device))["logit"] / max(temperature, 1e-6)
            prob = torch.sigmoid(logits).view(-1).cpu().numpy().tolist()
            y_prob.extend(prob)
            y_true.extend(labels.numpy().tolist())
        m = compute_metrics(y_true, y_prob)
        print(
            f"Group {group_col}={group_name} | "
            f"auroc={m['auroc']:.4f} auprc={m['auprc']:.4f} f1={m['f1']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"C:\Users\32902\Desktop\ai-image-detector\NTIRE-RobustAIGenDetection-train",
    )
    parser.add_argument("--save-dir", type=str, default=str(PROJECT_ROOT / "checkpoints_ntire"))
    parser.add_argument("--shards", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=0.2)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--use-balanced-sampler", action="store_true")
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--backbone-trainable-layers", type=int, default=1)
    parser.add_argument("--fused-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fusion-gate-input-dropout", type=float, default=0.1)
    parser.add_argument("--fusion-feature-dropout", type=float, default=0.1)
    parser.add_argument("--disable-freq-global-pool", action="store_true")
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=0.3)
    parser.add_argument("--jpeg-aligned-crop-p", type=float, default=0.3)
    parser.add_argument("--disable-to-gray", action="store_true")
    parser.add_argument("--disable-defocus", action="store_true")
    parser.add_argument("--disable-webp-compression", action="store_true")
    parser.add_argument("--disable-aux-heads", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
