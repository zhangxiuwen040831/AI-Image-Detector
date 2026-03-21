from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire import NTIRETrainer  # noqa: E402
from ai_image_detector.ntire.augmentations import (  # noqa: E402
    build_eval_transform,
    build_real_balanced_negative_focus_transform,
    build_real_clean_transform,
    build_real_hard_negative_transform,
    build_train_transform,
    get_real_profile_probabilities,
)
from ai_image_detector.ntire.dataset import (  # noqa: E402
    BufferedTransformDataset,
    HardRealBatchSampler,
    NTIRETrainDataset,
    build_train_val_indices,
    print_dataset_sanity,
)
from ai_image_detector.ntire.model import HybridAIGCDetector  # noqa: E402
from mine_base_hard_reals import mine_base_hard_real_indices, parse_shards  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_total_epochs(resume_path: Optional[Path], requested_epochs: Optional[int], phase1_epochs: int) -> int:
    if requested_epochs is not None:
        return int(requested_epochs)
    if resume_path is None or not resume_path.exists():
        return int(phase1_epochs)
    ckpt = torch.load(resume_path, map_location="cpu")
    resume_epoch = int(ckpt.get("epoch", 0))
    return int(resume_epoch + phase1_epochs)


def main() -> int:
    parser = argparse.ArgumentParser(description="V9 Phase 1 base-debias quick run.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "v9_phase1"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--shards", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--phase1-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None, help="Absolute final epoch index after resume.")
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--semantic-trainable-layers", type=int, default=2)
    parser.add_argument("--fused-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fusion-gate-input-dropout", type=float, default=0.1)
    parser.add_argument("--fusion-feature-dropout", type=float, default=0.1)
    parser.add_argument("--jpeg-aligned-crop-p", type=float, default=0.3)
    parser.add_argument("--disable-defocus", action="store_true")
    parser.add_argument("--disable-webp-compression", action="store_true")
    parser.add_argument("--disable-to-gray", action="store_true")
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--max-checkpoint-keep", type=int, default=20)
    parser.add_argument("--collapse-min-separation", type=float, default=0.2)
    parser.add_argument("--collapse-min-logit-std", type=float, default=0.4)
    parser.add_argument("--hybrid-score-sep-weight", type=float, default=0.1)
    parser.add_argument("--v9-semantic-head-weight", type=float, default=0.25)
    parser.add_argument("--v9-frequency-head-weight", type=float, default=0.20)
    parser.add_argument("--v9-hard-real-margin-weight", type=float, default=0.10)
    parser.add_argument("--v9-hard-real-margin", type=float, default=0.25)
    parser.add_argument("--v9-prototype-weight", type=float, default=0.05)
    parser.add_argument("--v9-prototype-margin", type=float, default=0.15)
    parser.add_argument("--hard-real-top-k", type=int, default=1200)
    parser.add_argument("--hard-real-max-samples", type=int, default=12000)
    parser.add_argument("--hard-real-min-probability", type=float, default=0.0)
    parser.add_argument("--hard-real-buffer-ratio", type=float, default=0.18)
    args = parser.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    resume_path = Path(args.resume) if args.resume else None
    resume_epoch = 0
    if resume_path is not None and resume_path.exists():
        ckpt_meta = torch.load(resume_path, map_location="cpu")
        resume_epoch = int(ckpt_meta.get("epoch", 0))
    total_epochs = resolve_total_epochs(
        resume_path=resume_path,
        requested_epochs=args.epochs,
        phase1_epochs=args.phase1_epochs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_ds = NTIRETrainDataset(
        root_dir=args.data_root,
        shard_ids=parse_shards(args.shards),
        transform=None,
        strict=False,
    )
    print_dataset_sanity(base_ds, max_rows=3)
    train_indices, val_indices, val_mode = build_train_val_indices(base_ds, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Validation mode: {val_mode}")
    print(f"Train samples: {len(train_indices)} | Val samples: {len(val_indices)}")

    train_transform = build_train_transform(
        image_size=args.image_size,
        jpeg_aligned_crop_p=args.jpeg_aligned_crop_p,
        enable_grayscale=not args.disable_to_gray,
        enable_defocus=not args.disable_defocus,
        enable_webp=not args.disable_webp_compression,
        chain_mix=True,
        chain_mix_strength="aigc_focus",
    )
    eval_transform = build_eval_transform(args.image_size)
    real_clean_transform = build_real_clean_transform(
        image_size=args.image_size,
        jpeg_aligned_crop_p=args.jpeg_aligned_crop_p,
        enable_webp=not args.disable_webp_compression,
    )
    real_mild_transform = build_real_balanced_negative_focus_transform(
        image_size=args.image_size,
        jpeg_aligned_crop_p=args.jpeg_aligned_crop_p,
        enable_defocus=not args.disable_defocus,
        enable_webp=not args.disable_webp_compression,
    )
    real_hard_transform = build_real_hard_negative_transform(
        image_size=args.image_size,
        jpeg_aligned_crop_p=args.jpeg_aligned_crop_p,
        enable_defocus=not args.disable_defocus,
        enable_webp=not args.disable_webp_compression,
    )
    standard_probs = get_real_profile_probabilities("standard_v9")
    hard_real_probs = get_real_profile_probabilities("hard_real_v9")
    train_ds = BufferedTransformDataset(
        base_dataset=base_ds,
        transform=train_transform,
        real_clean_transform=real_clean_transform,
        real_mild_transform=real_mild_transform,
        real_hard_transform=real_hard_transform,
        real_clean_prob=standard_probs[0],
        real_mild_prob=standard_probs[1],
        real_hard_prob=standard_probs[2],
        hard_real_indices=set(),
        hard_real_clean_prob=hard_real_probs[0],
        hard_real_mild_prob=hard_real_probs[1],
        hard_real_hard_prob=hard_real_probs[2],
    )
    val_ds = BufferedTransformDataset(base_dataset=base_ds, transform=eval_transform)
    val_loader = DataLoader(
        Subset(val_ds, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False,
    )

    model = HybridAIGCDetector(
        backbone_name=args.backbone_name,
        pretrained_backbone=args.pretrained_backbone,
        backbone_trainable_layers=0,
        image_size=args.image_size,
        fused_dim=args.fused_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        use_aux_heads=True,
        fusion_gate_input_dropout=args.fusion_gate_input_dropout,
        fusion_feature_dropout=args.fusion_feature_dropout,
        enable_base_residual_fusion=True,
        v8_stage="residual_finetune",
    )
    trainer = NTIRETrainer(
        model=model,
        device=device,
        save_dir=save_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=total_epochs,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        use_ema=not args.disable_ema,
        ema_decay=args.ema_decay,
        bce_weight=args.bce_weight,
        focal_weight=args.focal_weight,
        aux_weight=args.aux_weight,
        consistency_weight=0.0,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        max_checkpoint_keep=args.max_checkpoint_keep,
        collapse_min_separation=args.collapse_min_separation,
        collapse_min_logit_std=args.collapse_min_logit_std,
        hybrid_score_sep_weight=args.hybrid_score_sep_weight,
        v8_enable_base_residual_fusion=True,
        v8_stage="residual_finetune",
        v9_phase="base_debias",
        v9_semantic_head_weight=args.v9_semantic_head_weight,
        v9_frequency_head_weight=args.v9_frequency_head_weight,
        v9_hard_real_margin_weight=args.v9_hard_real_margin_weight,
        v9_hard_real_margin=args.v9_hard_real_margin,
        v9_prototype_weight=args.v9_prototype_weight,
        v9_prototype_margin=args.v9_prototype_margin,
    )
    if resume_path is not None:
        trainer.resume(resume_path)
        print(f"Resumed from: {resume_path}")
    freeze_summary = trainer.configure_v9_base_debias(semantic_trainable_layers=args.semantic_trainable_layers)
    print(f"V9 base-debias config: {json.dumps(freeze_summary, ensure_ascii=False)}")

    hard_real_output_path = save_dir / "hard_real_buffer.json"
    hard_real_indices, hard_real_summary = mine_base_hard_real_indices(
        model=trainer.model,
        device=trainer.device,
        dataset=base_ds,
        train_indices=train_indices,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.hard_real_max_samples,
        top_k=args.hard_real_top_k,
        seed=args.seed,
        min_probability=args.hard_real_min_probability,
        output_path=hard_real_output_path,
    )
    active_hard_real_indices = sorted(set(train_indices).intersection(hard_real_indices))
    train_ds.set_hard_real_indices(active_hard_real_indices)
    print(
        "Hard-real mining | "
        f"selected={len(active_hard_real_indices)} "
        f"buffer_ratio={args.hard_real_buffer_ratio:.2f} "
        f"real_mix={standard_probs[0]:.2f}/{standard_probs[1]:.2f}/{standard_probs[2]:.2f} "
        f"hard_real_mix={hard_real_probs[0]:.2f}/{hard_real_probs[1]:.2f}/{hard_real_probs[2]:.2f}"
    )

    def build_train_loader_for_epoch(epoch: int) -> DataLoader:
        if active_hard_real_indices:
            batch_sampler = HardRealBatchSampler(
                primary_indices=train_indices,
                buffer_indices=active_hard_real_indices,
                batch_size=args.batch_size,
                buffer_ratio=args.hard_real_buffer_ratio,
                seed=args.seed + epoch,
            )
            return DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=True if args.num_workers > 0 else False,
            )
        return DataLoader(
            Subset(train_ds, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if args.num_workers > 0 else False,
            drop_last=False,
        )

    history_path = save_dir / "history.json"
    trainer.fit(
        train_loader_factory=build_train_loader_for_epoch,
        val_loader=val_loader,
        log_path=history_path,
    )

    history: List[Dict[str, object]] = []
    if history_path.exists():
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = loaded
        except Exception:
            history = []
    last_row = history[-1] if history else {}
    summary = {
        "phase": "base_debias",
        "resume_from": str(resume_path) if resume_path is not None else None,
        "resume_epoch": int(resume_epoch),
        "epochs_run": int(max(total_epochs - resume_epoch, 0)),
        "requested_total_epochs": int(total_epochs),
        "train_indices": int(len(train_indices)),
        "val_indices": int(len(val_indices)),
        "hard_real_buffer_ratio": float(args.hard_real_buffer_ratio),
        "hard_real_selected_count": int(len(active_hard_real_indices)),
        "freeze_summary": freeze_summary,
        "hard_real_buffer_stats": hard_real_summary,
        "last_epoch": last_row,
    }
    phase1_summary_path = save_dir / "phase1_summary.json"
    phase1_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Training completed. Outputs saved to: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
