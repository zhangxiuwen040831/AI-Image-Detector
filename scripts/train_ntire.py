from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire import NTIRETrainer
from ai_image_detector.ntire.augmentations import (
    build_eval_transform,
    build_real_balanced_negative_focus_transform,
    build_real_clean_transform,
    build_real_hard_negative_transform,
    build_train_transform,
    get_real_profile_probabilities,
)
from ai_image_detector.ntire.dataset import (
    NTIRETrainDataset,
    build_balanced_sample_weights_from_dataframe,
    build_train_val_indices,
    load_hard_negative_names,
    print_dataset_sanity,
)
from ai_image_detector.ntire.metrics import compute_metrics
from ai_image_detector.ntire.model import HybridAIGCDetector


def normalize_real_transform_probs(
    clean_prob: float,
    mild_prob: float,
    hard_prob: float,
) -> Tuple[float, float, float]:
    probs = np.asarray([clean_prob, mild_prob, hard_prob], dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 1e-12:
        return 1.0, 0.0, 0.0
    probs = probs / total
    return float(probs[0]), float(probs[1]), float(probs[2])


def hard_negative_weighting_active(epoch: int, start_epoch: int) -> bool:
    return int(epoch) >= int(start_epoch)


def compute_counterfactual_mining_score(
    final_logit: torch.Tensor,
    base_logit: torch.Tensor,
    noise_delta_logit: torch.Tensor,
    alpha: torch.Tensor,
    min_probability: float = 0.35,
    alpha_max: float = 0.35,
) -> torch.Tensor:
    final_probability = torch.sigmoid(final_logit.view(-1))
    base_probability = torch.sigmoid(base_logit.view(-1))
    alpha_norm = (alpha.view(-1) / max(float(alpha_max), 1e-6)).clamp(0.0, 1.0)
    residual_push = alpha.view(-1) * noise_delta_logit.view(-1)
    controller_failure_mask = (
        (base_probability < 0.5)
        & (final_probability >= float(min_probability))
        & (alpha_norm >= 0.10)
        & (residual_push > 0.0)
    ).float()
    score = (
        final_probability
        * alpha_norm
        * torch.relu(noise_delta_logit.view(-1))
        * torch.sigmoid(-base_logit.view(-1))
    )
    return score * controller_failure_mask


class AntiShortcutBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        primary_indices: Sequence[int],
        anti_shortcut_indices: Sequence[int],
        batch_size: int,
        anti_shortcut_ratio: float,
        seed: int,
    ) -> None:
        self.primary_indices = list(primary_indices)
        self.anti_shortcut_indices = sorted(set(int(idx) for idx in anti_shortcut_indices))
        self.batch_size = int(batch_size)
        self.anti_shortcut_ratio = max(0.0, min(float(anti_shortcut_ratio), 0.5))
        self.seed = int(seed)

    def __len__(self) -> int:
        return max(len(self.primary_indices) // max(self.batch_size, 1), 1)

    def __iter__(self):
        rng = random.Random(self.seed)
        primary = self.primary_indices[:]
        rng.shuffle(primary)
        anti_count = 0
        if self.anti_shortcut_indices:
            anti_count = max(1, int(round(self.batch_size * self.anti_shortcut_ratio)))
            anti_count = min(anti_count, self.batch_size - 1)
        primary_count = max(self.batch_size - anti_count, 1)
        cursor = 0
        for _ in range(len(self)):
            batch = []
            for _ in range(primary_count):
                if cursor >= len(primary):
                    rng.shuffle(primary)
                    cursor = 0
                batch.append(primary[cursor])
                cursor += 1
            for _ in range(anti_count):
                batch.append(rng.choice(self.anti_shortcut_indices))
            rng.shuffle(batch)
            yield batch


class TransformDataset(Dataset):
    def __init__(
        self,
        base_dataset: NTIRETrainDataset,
        transform,
        secondary_transform=None,
        real_clean_transform=None,
        real_mild_transform=None,
        real_hard_transform=None,
        secondary_real_clean_transform=None,
        secondary_real_mild_transform=None,
        secondary_real_hard_transform=None,
        real_clean_prob: float = 0.2,
        real_mild_prob: float = 0.4,
        real_hard_prob: float = 0.4,
        anti_shortcut_indices: Optional[Sequence[int]] = None,
        anti_shortcut_clean_prob: float = 0.6,
        anti_shortcut_mild_prob: float = 0.3,
        anti_shortcut_hard_prob: float = 0.1,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform
        self.secondary_transform = secondary_transform
        self.real_clean_transform = real_clean_transform
        self.real_mild_transform = real_mild_transform
        self.real_hard_transform = real_hard_transform
        self.secondary_real_clean_transform = secondary_real_clean_transform
        self.secondary_real_mild_transform = secondary_real_mild_transform
        self.secondary_real_hard_transform = secondary_real_hard_transform
        self.real_clean_prob, self.real_mild_prob, self.real_hard_prob = normalize_real_transform_probs(
            clean_prob=real_clean_prob,
            mild_prob=real_mild_prob,
            hard_prob=real_hard_prob,
        )
        self.anti_shortcut_indices = {int(idx) for idx in (anti_shortcut_indices or [])}
        (
            self.anti_shortcut_clean_prob,
            self.anti_shortcut_mild_prob,
            self.anti_shortcut_hard_prob,
        ) = normalize_real_transform_probs(
            clean_prob=anti_shortcut_clean_prob,
            mild_prob=anti_shortcut_mild_prob,
            hard_prob=anti_shortcut_hard_prob,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _sample_real_mode(self, anti_shortcut: bool = False) -> str:
        if anti_shortcut:
            clean_prob = self.anti_shortcut_clean_prob
            mild_prob = self.anti_shortcut_mild_prob
        else:
            clean_prob = self.real_clean_prob
            mild_prob = self.real_mild_prob
        r = random.random()
        if r < clean_prob:
            return "clean"
        if r < clean_prob + mild_prob:
            return "mild"
        return "hard"

    @staticmethod
    def _resolve_real_transform(
        mode: str,
        fallback_transform,
        clean_transform,
        mild_transform,
        hard_transform,
    ):
        candidates = {
            "clean": clean_transform,
            "mild": mild_transform,
            "hard": hard_transform,
        }
        if candidates.get(mode) is not None:
            return candidates[mode]
        for candidate_mode in ("clean", "mild", "hard"):
            candidate = candidates.get(candidate_mode)
            if candidate is not None:
                return candidate
        return fallback_transform

    def __getitem__(self, index: int):
        record = self.base_dataset.records[index]
        label = torch.tensor(record.label, dtype=torch.float32)
        metadata = dict(record.metadata)
        metadata["dataset_index"] = int(index)
        metadata["image_path"] = str(record.image_path)
        metadata["label"] = int(record.label)
        metadata["real_transform_policy"] = "not_applicable"
        metadata["real_transform_profile"] = "not_applicable"
        metadata["anti_shortcut_buffer_hit"] = bool(index in self.anti_shortcut_indices)
        arr = np.array(Image.open(record.image_path).convert("RGB"))
        primary_transform = self.transform
        secondary_transform = self.secondary_transform
        if record.label == 0:
            anti_shortcut = bool(index in self.anti_shortcut_indices)
            metadata["anti_shortcut_buffer_hit"] = anti_shortcut
            metadata["real_transform_policy"] = "anti_shortcut" if anti_shortcut else "standard_real"
            real_mode = self._sample_real_mode(anti_shortcut=anti_shortcut)
            metadata["real_transform_profile"] = real_mode
            primary_transform = self._resolve_real_transform(
                real_mode,
                primary_transform,
                self.real_clean_transform,
                self.real_mild_transform,
                self.real_hard_transform,
            )
            secondary_transform = self._resolve_real_transform(
                real_mode,
                secondary_transform,
                self.secondary_real_clean_transform,
                self.secondary_real_mild_transform,
                self.secondary_real_hard_transform,
            )
        if primary_transform is None:
            image = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        else:
            image = primary_transform(image=arr)["image"]
        if secondary_transform is None:
            return image, label, metadata
        image2 = secondary_transform(image=arr)["image"]
        return image, label, metadata, image2


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


def build_sampler_from_indices(
    dataset: NTIRETrainDataset,
    train_indices: List[int],
    hard_negative_names: Optional[set[str]] = None,
    hard_negative_indices: Optional[Sequence[int]] = None,
    hard_negative_boost: float = 3.0,
) -> WeightedRandomSampler:
    df = dataset.to_dataframe().iloc[train_indices].copy()
    df["_dataset_index"] = train_indices
    df = df.reset_index(drop=True)
    weights = build_balanced_sample_weights_from_dataframe(
        df,
        hard_negative_names=hard_negative_names,
        hard_negative_indices=hard_negative_indices,
        hard_negative_boost=hard_negative_boost,
    )
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True)


def mine_hard_negative_indices(
    trainer: NTIRETrainer,
    dataset: NTIRETrainDataset,
    train_indices: Sequence[int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    top_k: int,
    min_probability: float,
    seed: int,
    mode: str = "probability",
    output_path: Optional[Path] = None,
) -> Tuple[Set[int], Dict[str, object]]:
    real_indices = [idx for idx in train_indices if int(dataset.records[idx].label) == 0]
    if not real_indices:
        return set(), {"selected_count": 0, "counterfactual_hits": 0, "mode": mode}
    if max_samples > 0 and len(real_indices) > max_samples:
        rng = random.Random(seed)
        candidate_indices = sorted(rng.sample(real_indices, max_samples))
    else:
        candidate_indices = sorted(real_indices)
    mining_ds = TransformDataset(dataset, build_eval_transform(image_size))
    loader = DataLoader(
        Subset(mining_ds, candidate_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False,
    )
    ranked = []
    model = trainer.model
    base_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    alpha_max = float(getattr(base_model, "v8_alpha_max", 0.35))
    model.eval()
    if trainer.model_ema is not None:
        trainer.model_ema.apply_shadow(model)
    try:
        with torch.no_grad():
            for images, _, metadata in loader:
                images = images.to(trainer.device, non_blocking=True)
                out = model(images)
                logits = out["logit"].detach().view(-1)
                base_logits = out.get("base_logit")
                noise_delta_logits = out.get("noise_delta_logit")
                alpha = out.get("alpha_used", out.get("alpha"))
                raw_weights = out.get("fusion_weights_raw", out.get("fusion_weights"))
                raw_semantic = raw_frequency = raw_noise = None
                if raw_weights is not None:
                    raw_semantic = raw_weights[:, 0].detach().cpu().tolist()
                    raw_frequency = raw_weights[:, 1].detach().cpu().tolist()
                    raw_noise = raw_weights[:, 2].detach().cpu().tolist()
                scaled_logits = logits / max(trainer.temperature_scaler.value(), 1e-6)
                probabilities = torch.sigmoid(scaled_logits).cpu().tolist()
                raw_logits = logits.cpu().tolist()
                if base_logits is not None:
                    base_logits = base_logits.detach().view(-1)
                    base_probabilities = torch.sigmoid(base_logits / max(trainer.temperature_scaler.value(), 1e-6)).cpu().tolist()
                    base_logits_list = base_logits.cpu().tolist()
                else:
                    base_probabilities = [None] * len(probabilities)
                    base_logits_list = [None] * len(probabilities)
                if noise_delta_logits is not None:
                    noise_delta_logits = noise_delta_logits.detach().view(-1)
                    noise_delta_list = noise_delta_logits.cpu().tolist()
                else:
                    noise_delta_list = [None] * len(probabilities)
                if alpha is not None:
                    alpha_values = alpha.detach().view(-1).cpu().tolist()
                else:
                    alpha_values = [None] * len(probabilities)
                residual_push_values = []
                for alpha_value, noise_delta in zip(alpha_values, noise_delta_list):
                    if alpha_value is None or noise_delta is None:
                        residual_push_values.append(None)
                    else:
                        residual_push_values.append(float(alpha_value) * float(noise_delta))
                if (
                    mode == "counterfactual"
                    and base_logits is not None
                    and noise_delta_logits is not None
                    and alpha is not None
                ):
                    scores = compute_counterfactual_mining_score(
                        final_logit=logits,
                        base_logit=base_logits,
                        noise_delta_logit=noise_delta_logits,
                        alpha=alpha.detach().view(-1),
                        min_probability=min_probability,
                        alpha_max=alpha_max,
                    ).cpu().tolist()
                    counterfactual_hits = [
                        bool(
                            (score_value > 0.0)
                            and (base_logit < 0.0)
                            and (noise_delta > 0.0)
                            and ((alpha_value / max(alpha_max, 1e-6)) >= 0.10)
                        )
                        for score_value, base_logit, noise_delta, alpha_value in zip(
                            scores,
                            base_logits_list,
                            noise_delta_list,
                            alpha_values,
                        )
                    ]
                else:
                    scores = probabilities
                    counterfactual_hits = [False] * len(probabilities)
                dataset_index_values = metadata.get("dataset_index")
                if isinstance(dataset_index_values, torch.Tensor):
                    dataset_index_values = dataset_index_values.cpu().tolist()
                image_names = list(metadata.get("image_name", []))
                image_paths = list(metadata.get("image_path", []))
                raw_semantic = raw_semantic or [None] * len(probabilities)
                raw_frequency = raw_frequency or [None] * len(probabilities)
                raw_noise = raw_noise or [None] * len(probabilities)
                for dataset_index, image_name, image_path, probability, raw_logit, base_probability, base_logit, alpha_value, noise_delta, residual_push, score, is_counterfactual_hit, fusion_semantic, fusion_frequency, fusion_noise in zip(
                    dataset_index_values,
                    image_names,
                    image_paths,
                    probabilities,
                    raw_logits,
                    base_probabilities,
                    base_logits_list,
                    alpha_values,
                    noise_delta_list,
                    residual_push_values,
                    scores,
                    counterfactual_hits,
                    raw_semantic,
                    raw_frequency,
                    raw_noise,
                ):
                    ranked.append(
                        {
                            "dataset_index": int(dataset_index),
                            "image_name": str(image_name),
                            "image_path": str(image_path),
                            "probability": float(probability),
                            "logit": float(raw_logit),
                            "base_probability": None if base_probability is None else float(base_probability),
                            "base_logit": None if base_logit is None else float(base_logit),
                            "alpha": None if alpha_value is None else float(alpha_value),
                            "alpha_norm": None if alpha_value is None else float(alpha_value / max(alpha_max, 1e-6)),
                            "noise_delta_logit": None if noise_delta is None else float(noise_delta),
                            "residual_push": None if residual_push is None else float(residual_push),
                            "score": float(score),
                            "counterfactual_hit": bool(is_counterfactual_hit),
                            "fusion_raw_semantic": None if fusion_semantic is None else float(fusion_semantic),
                            "fusion_raw_frequency": None if fusion_frequency is None else float(fusion_frequency),
                            "fusion_raw_noise": None if fusion_noise is None else float(fusion_noise),
                        }
                    )
    finally:
        if trainer.model_ema is not None:
            trainer.model_ema.restore(model)
    ranked.sort(key=lambda row: row["score"], reverse=True)
    if mode == "counterfactual":
        eligible = [
            row for row in ranked
            if float(row.get("score", 0.0)) > 0.0 and float(row.get("probability", 0.0)) >= float(min_probability)
        ]
    else:
        eligible = [row for row in ranked if row["probability"] >= min_probability]
    if top_k > 0:
        selected = eligible[:top_k]
        if mode != "counterfactual" and len(selected) < min(top_k, len(ranked)):
            used = {row["dataset_index"] for row in selected}
            for row in ranked:
                if row["dataset_index"] in used:
                    continue
                selected.append(row)
                used.add(row["dataset_index"])
                if len(selected) >= top_k:
                    break
    else:
        selected = eligible
    selected_indices = {int(row["dataset_index"]) for row in selected}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": mode,
            "total_real_candidates": len(real_indices),
            "sampled_real_candidates": len(candidate_indices),
            "top_k": int(top_k),
            "min_probability": float(min_probability),
            "selected_count": len(selected_indices),
            "counterfactual_hits": int(sum(1 for row in selected if bool(row.get("counterfactual_hit")))),
            "avg_score": float(np.mean([row["score"] for row in selected])) if selected else None,
            "avg_residual_push": float(np.mean([row["residual_push"] for row in selected if row.get("residual_push") is not None])) if selected else None,
            "selected_preview": selected[: min(len(selected), 200)],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        payload = {
            "mode": mode,
            "selected_count": len(selected_indices),
            "counterfactual_hits": int(sum(1 for row in selected if bool(row.get("counterfactual_hit")))),
            "avg_score": float(np.mean([row["score"] for row in selected])) if selected else None,
            "avg_residual_push": float(np.mean([row["residual_push"] for row in selected if row.get("residual_push") is not None])) if selected else None,
        }
    return selected_indices, payload


def run(args: argparse.Namespace) -> None:
    if args.v81_controller_only:
        args.v8_enable_base_residual_fusion = True
        args.v8_stage = "residual_finetune"
        args.mixup_prob = 0.0
        args.label_smoothing = 0.0
        (
            args.anti_shortcut_real_clean_prob,
            args.anti_shortcut_real_mild_prob,
            args.anti_shortcut_real_hard_prob,
        ) = get_real_profile_probabilities("anti_shortcut_v81")
        print(
            "V8.1 controller-only overrides applied | "
            "stage=residual_finetune mixup_prob=0.0 label_smoothing=0.0 "
            "anti_shortcut_real_mix=0.70/0.20/0.10"
        )
    set_seed(args.seed)
    pin_memory = torch.cuda.is_available()
    shard_ids = parse_shards(args.shards)
    base_ds = NTIRETrainDataset(root_dir=args.data_root, shard_ids=shard_ids, transform=None, strict=False)
    print_dataset_sanity(base_ds, max_rows=3)
    train_indices, val_indices, val_mode = build_train_val_indices(base_ds, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Validation mode: {val_mode}")
    # V3: Enable aigc_focus enhancement mode
    train_transform = build_train_transform(
        image_size=args.image_size,
        jpeg_aligned_crop_p=args.jpeg_aligned_crop_p,
        enable_grayscale=not args.disable_to_gray,
        enable_defocus=not args.disable_defocus,
        enable_webp=not args.disable_webp_compression,
        strong_compression_p=args.strong_compression_p,
        down_up_resize_p=args.down_up_resize_p,
        roundtrip_p=args.roundtrip_p,
        chain_mix=True,
        chain_mix_strength="aigc_focus",
    )
    real_hard_transform = None
    real_mild_transform = None
    real_clean_transform = build_real_clean_transform(
        image_size=args.image_size,
        jpeg_aligned_crop_p=args.jpeg_aligned_crop_p,
        enable_webp=not args.disable_webp_compression,
    )
    if args.real_hard_augment:
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
    consistency_transform = None
    consistency_real_clean_transform = None
    consistency_real_mild_transform = None
    consistency_real_hard_transform = None
    if args.consistency_weight > 0:
        consistency_transform = build_train_transform(
            image_size=args.image_size,
            jpeg_aligned_crop_p=args.jpeg_aligned_crop_p,
            enable_grayscale=not args.disable_to_gray,
            enable_defocus=not args.disable_defocus,
            enable_webp=not args.disable_webp_compression,
            strong_compression_p=args.strong_compression_p,
            down_up_resize_p=args.down_up_resize_p,
            roundtrip_p=args.roundtrip_p,
            chain_mix=True,
            chain_mix_strength="aigc_focus",
        )
        consistency_real_clean_transform = build_real_clean_transform(
            image_size=args.image_size,
            jpeg_aligned_crop_p=args.jpeg_aligned_crop_p,
            enable_webp=not args.disable_webp_compression,
        )
        if args.real_hard_augment:
            consistency_real_mild_transform = build_real_balanced_negative_focus_transform(
                image_size=args.image_size,
                jpeg_aligned_crop_p=args.jpeg_aligned_crop_p,
                enable_defocus=not args.disable_defocus,
                enable_webp=not args.disable_webp_compression,
            )
            consistency_real_hard_transform = build_real_hard_negative_transform(
                image_size=args.image_size,
                jpeg_aligned_crop_p=args.jpeg_aligned_crop_p,
                enable_defocus=not args.disable_defocus,
                enable_webp=not args.disable_webp_compression,
            )
    real_clean_prob, real_mild_prob, real_hard_prob = normalize_real_transform_probs(
        clean_prob=args.real_clean_negative_prob,
        mild_prob=args.real_mild_negative_prob,
        hard_prob=args.real_hard_negative_prob,
    )
    anti_shortcut_clean_prob, anti_shortcut_mild_prob, anti_shortcut_hard_prob = normalize_real_transform_probs(
        clean_prob=args.anti_shortcut_real_clean_prob,
        mild_prob=args.anti_shortcut_real_mild_prob,
        hard_prob=args.anti_shortcut_real_hard_prob,
    )
    train_ds = TransformDataset(
        base_ds,
        train_transform,
        secondary_transform=consistency_transform,
        real_clean_transform=real_clean_transform,
        real_mild_transform=real_mild_transform,
        real_hard_transform=real_hard_transform,
        secondary_real_clean_transform=consistency_real_clean_transform,
        secondary_real_mild_transform=consistency_real_mild_transform,
        secondary_real_hard_transform=consistency_real_hard_transform,
        real_clean_prob=real_clean_prob,
        real_mild_prob=real_mild_prob,
        real_hard_prob=real_hard_prob,
        anti_shortcut_indices=set(),
        anti_shortcut_clean_prob=anti_shortcut_clean_prob,
        anti_shortcut_mild_prob=anti_shortcut_mild_prob,
        anti_shortcut_hard_prob=anti_shortcut_hard_prob,
    )
    val_ds = TransformDataset(base_ds, build_eval_transform(args.image_size))
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
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
        max_semantic_weight=args.max_semantic_weight,
        min_semantic_weight=args.min_semantic_weight,
        min_noise_weight=args.min_noise_weight,
        noise_branch_dropout_prob=args.noise_branch_dropout_prob,
        noise_branch_dropout_scale=args.noise_branch_dropout_scale,
        use_weight_regularization=args.use_weight_regularization,
        weight_reg_lambda=args.weight_reg_lambda,
        enable_base_residual_fusion=args.v8_enable_base_residual_fusion,
        v8_alpha_max=args.v8_alpha_max,
        v8_stage=args.v8_stage,
    )
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
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
        consistency_weight=args.consistency_weight,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        max_checkpoint_keep=args.max_checkpoint_keep,
        aux_schedule=args.aux_schedule,
        collapse_min_separation=args.collapse_min_separation,
        collapse_min_logit_std=args.collapse_min_logit_std,
        hybrid_score_sep_weight=args.hybrid_score_sep_weight,
        v8_enable_base_residual_fusion=args.v8_enable_base_residual_fusion,
        v8_stage=args.v8_stage,
        v8_base_loss_weight=args.v8_base_loss_weight,
        v8_semantic_head_weight=args.v8_semantic_head_weight,
        v8_frequency_head_weight=args.v8_frequency_head_weight,
        v8_noise_head_weight=args.v8_noise_head_weight,
        v8_real_alpha_loss_weight=args.v8_real_alpha_loss_weight,
        v8_real_noise_push_weight=args.v8_real_noise_push_weight,
        v8_real_counterfactual_weight=args.v8_real_counterfactual_weight,
        v8_hard_aigc_alpha_weight=args.v8_hard_aigc_alpha_weight,
        v8_real_counterfactual_margin=args.v8_real_counterfactual_margin,
        v8_hard_aigc_alpha_target=args.v8_hard_aigc_alpha_target,
        v81_controller_only=args.v81_controller_only,
        v81_alpha_real_weight=args.v81_alpha_real_weight,
        v81_alpha_easy_aigc_weight=args.v81_alpha_easy_aigc_weight,
        v81_alpha_hard_aigc_weight=args.v81_alpha_hard_aigc_weight,
        v81_alpha_easy_aigc_threshold=args.v81_alpha_easy_aigc_threshold,
        v81_alpha_hard_target=args.v81_alpha_hard_target,
        v81_real_residual_push_weight=args.v81_real_residual_push_weight,
        v81_real_counterfactual_weight=args.v81_real_counterfactual_weight,
        v81_real_counterfactual_margin=args.v81_real_counterfactual_margin,
        v81_hard_aigc_residual_min_weight=args.v81_hard_aigc_residual_min_weight,
        v81_easy_aigc_residual_suppress_weight=args.v81_easy_aigc_residual_suppress_weight,
        v81_residual_push_target=args.v81_residual_push_target,
    )
    
    if args.resume is not None:
        print(f"Resuming from: {args.resume}")
        trainer.resume(Path(args.resume))
    trainer.set_v8_stage(args.v8_stage)
    if args.v81_controller_only:
        freeze_summary = trainer.configure_v81_controller_only(True)
        print(f"V8.1 controller-only freeze summary: {json.dumps(freeze_summary, ensure_ascii=False)}")
    if args.v8_enable_base_residual_fusion and args.v8_stage == "debias_base":
        base_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        for module_name in ("noise_controller",):
            module = getattr(base_model, module_name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False
        print("V8 stage=debias_base | noise controller frozen and alpha contribution disabled.")

    hard_negative_names: Set[str] = set()
    hard_negative_indices: Set[int] = set()
    anti_shortcut_indices: Set[int] = set()
    mining_summary: Dict[str, object] = {"mode": args.v8_hard_negative_mode, "selected_count": 0}
    mining_output_path = Path(args.save_dir) / "hard_negative_mining.json"
    auto_mining_enabled = (
        args.use_hard_negatives
        and (not args.disable_hard_negative_mining)
        and (args.resume is not None)
        and args.v8_stage == "residual_finetune"
    )
    if args.use_hard_negatives and auto_mining_enabled:
        mined_indices, mining_summary = mine_hard_negative_indices(
            trainer=trainer,
            dataset=base_ds,
            train_indices=train_indices,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.hard_negative_max_samples,
            top_k=args.hard_negative_top_k,
            min_probability=args.hard_negative_min_probability,
            seed=args.seed,
            mode=args.v8_hard_negative_mode,
            output_path=mining_output_path,
        )
        hard_negative_indices = mined_indices
        anti_shortcut_indices = mined_indices
        train_ds.anti_shortcut_indices = set(anti_shortcut_indices)
        print(
            f"Auto-mined {len(hard_negative_indices)} anti-shortcut indices "
            f"mode={args.v8_hard_negative_mode} -> {mining_output_path}"
        )
    elif args.use_hard_negatives and args.resume is None and not args.disable_hard_negative_mining:
        print("Skipping auto hard-negative mining because no resume checkpoint was provided.")
    if args.use_hard_negatives and (not hard_negative_indices):
        hard_negative_names = load_hard_negative_names(args.hard_negatives_file)
        print(f"Loaded {len(hard_negative_names)} fallback hard-negative names from {args.hard_negatives_file}")
    matched_hard_negative_indices = sorted(set(train_indices).intersection(hard_negative_indices))
    if matched_hard_negative_indices:
        print(f"Hard-negative training hits: {len(matched_hard_negative_indices)}")

    def build_train_loader_for_epoch(epoch: int) -> DataLoader:
        weighting_active = args.use_hard_negatives and hard_negative_weighting_active(
            epoch=epoch,
            start_epoch=args.hard_negative_start_epoch,
        )
        active_hard_negative_names = hard_negative_names if weighting_active else set()
        active_hard_negative_indices = hard_negative_indices if weighting_active else set()
        active_anti_shortcut_indices = anti_shortcut_indices if weighting_active else set()
        train_ds.anti_shortcut_indices = set(active_anti_shortcut_indices)
        use_anti_shortcut_buffer = bool(
            args.v8_enable_base_residual_fusion
            and args.v8_hard_negative_mode == "counterfactual"
            and active_anti_shortcut_indices
            and args.v8_stage == "residual_finetune"
        )
        use_sampler = args.use_balanced_sampler or (
            weighting_active and (not use_anti_shortcut_buffer) and (not args.v81_controller_only)
        )
        print(
            f"Hard-negative schedule | epoch={epoch} "
            f"mining_active={bool(args.use_hard_negatives and auto_mining_enabled)} "
            f"weighting_active={bool(weighting_active)} "
            f"start_epoch={args.hard_negative_start_epoch} "
            f"active_indices={len(active_hard_negative_indices)} "
            f"anti_shortcut_buffer={bool(use_anti_shortcut_buffer)}"
        )
        if use_anti_shortcut_buffer:
            batch_sampler = AntiShortcutBatchSampler(
                primary_indices=train_indices,
                anti_shortcut_indices=sorted(active_anti_shortcut_indices),
                batch_size=args.batch_size,
                anti_shortcut_ratio=args.v8_anti_shortcut_buffer_ratio,
                seed=args.seed + epoch,
            )
            return DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=True if args.num_workers > 0 else False,
            )
        if use_sampler:
            sampler = build_sampler_from_indices(
                base_ds,
                train_indices,
                hard_negative_names=active_hard_negative_names,
                hard_negative_indices=active_hard_negative_indices,
                hard_negative_boost=args.hard_negative_boost,
            )
            return DataLoader(
                train_subset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=True if args.num_workers > 0 else False,
                drop_last=True,
            )
        return DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if args.num_workers > 0 else False,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False,
    )
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Workers: {args.num_workers}")
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    print(
        "Real transform mix (clean/mild/hard): "
        f"{real_clean_prob:.2f}/"
        f"{real_mild_prob:.2f}/"
        f"{real_hard_prob:.2f}"
    )
    print(
        "Anti-shortcut real mix (clean/mild/hard): "
        f"{anti_shortcut_clean_prob:.2f}/"
        f"{anti_shortcut_mild_prob:.2f}/"
        f"{anti_shortcut_hard_prob:.2f}"
    )
    print(f"Real hard augment enabled: {args.real_hard_augment}")
    print(
        f"Hard-negative weighting enabled: {args.use_hard_negatives} "
        f"(start_epoch={args.hard_negative_start_epoch})"
    )
    if args.use_hard_negatives:
        mining_summary_brief = {
            key: value for key, value in mining_summary.items() if key != "selected_preview"
        }
        print(
            f"Hard-negative mode: {args.v8_hard_negative_mode} | "
            f"anti_shortcut_buffer_ratio={args.v8_anti_shortcut_buffer_ratio:.2f}"
        )
        print(f"Mining summary: {json.dumps(mining_summary_brief, ensure_ascii=False)}")
    print(
        f"Noise branch dropout: p={args.noise_branch_dropout_prob:.2f}, "
        f"scale={args.noise_branch_dropout_scale:.2f}"
    )
    
    trainer.fit(
        train_loader_factory=build_train_loader_for_epoch,
        val_loader=val_loader,
        log_path=Path(args.save_dir) / "history.json",
    )
    
    print("\nTraining completed!")
    print(f"Checkpoints saved to: {args.save_dir}")
    
    # 可选：评估 distortion groups
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
        # 修复：不 return，继续执行
    else:
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
        default="/root/lanyun-tmp/NTIRE-RobustAIGenDetection-train",
    )
    parser.add_argument("--save-dir", type=str, default=str(PROJECT_ROOT / "checkpoints_ntire"))
    parser.add_argument("--shards", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate (V4: 5e-7)")
    parser.add_argument("--consistency-weight", type=float, default=0.1, help="Consistency loss weight")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--use-balanced-sampler", action="store_true")
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--pretrained-backbone", action="store_true", default=True)
    parser.add_argument("--backbone-trainable-layers", type=int, default=0)
    parser.add_argument("--fused-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fusion-gate-input-dropout", type=float, default=0.1)
    parser.add_argument("--fusion-feature-dropout", type=float, default=0.1)
    parser.add_argument("--disable-freq-global-pool", action="store_true")
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=0.3)
    parser.add_argument("--jpeg-aligned-crop-p", type=float, default=0.3)
    parser.add_argument("--strong-compression-p", type=float, default=0.65)
    parser.add_argument("--down-up-resize-p", type=float, default=0.35)
    parser.add_argument("--roundtrip-p", type=float, default=0.35)
    parser.add_argument("--disable-to-gray", action="store_true")
    parser.add_argument("--disable-defocus", action="store_true")
    parser.add_argument("--disable-webp-compression", action="store_true")
    parser.add_argument("--disable-aux-heads", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-schedule", type=str, default="default", choices=["default", "staged", "none"])
    parser.add_argument("--max-checkpoint-keep", type=int, default=100)
    parser.add_argument("--collapse-min-separation", type=float, default=0.2)
    parser.add_argument("--collapse-min-logit-std", type=float, default=0.4)
    parser.add_argument("--hybrid-score-sep-weight", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--cosine", action="store_true")
    # V3: Fusion weight constraint parameters
    parser.add_argument("--max-semantic-weight", type=float, default=0.45, help="Maximum semantic branch weight")
    parser.add_argument("--min-semantic-weight", type=float, default=0.0, help="Deprecated in V7; kept for compatibility")
    parser.add_argument("--min-noise-weight", type=float, default=0.20, help="Minimum noise branch weight")
    parser.add_argument("--noise-branch-dropout-prob", type=float, default=0.20)
    parser.add_argument("--noise-branch-dropout-scale", type=float, default=0.50)
    parser.add_argument("--use-weight-regularization", action="store_true", default=True, help="Use weight regularization")
    parser.add_argument("--weight-reg-lambda", type=float, default=0.01, help="Weight regularization lambda")
    parser.add_argument("--use-hard-negatives", action="store_true")
    parser.add_argument("--disable-hard-negative-mining", action="store_true")
    parser.add_argument("--hard-negatives-file", type=str, default=str(PROJECT_ROOT / "configs" / "hard_negatives.txt"))
    parser.add_argument("--hard-negative-boost", type=float, default=3.0)
    parser.add_argument("--hard-negative-top-k", type=int, default=1000)
    parser.add_argument("--hard-negative-max-samples", type=int, default=12000)
    parser.add_argument("--hard-negative-min-probability", type=float, default=0.35)
    parser.add_argument("--hard-negative-start-epoch", type=int, default=3)
    parser.add_argument("--real-hard-augment", action="store_true")
    parser.add_argument("--real-clean-negative-prob", type=float, default=0.4)
    parser.add_argument("--real-mild-negative-prob", type=float, default=0.4)
    parser.add_argument("--real-hard-negative-prob", type=float, default=0.2)
    parser.add_argument("--anti-shortcut-real-clean-prob", type=float, default=0.6)
    parser.add_argument("--anti-shortcut-real-mild-prob", type=float, default=0.3)
    parser.add_argument("--anti-shortcut-real-hard-prob", type=float, default=0.1)
    parser.add_argument("--v8-enable-base-residual-fusion", action="store_true")
    parser.add_argument("--v8-alpha-max", type=float, default=0.35)
    parser.add_argument("--v8-stage", type=str, default="residual_finetune", choices=["debias_base", "residual_finetune"])
    parser.add_argument("--v8-base-loss-weight", type=float, default=0.75)
    parser.add_argument("--v8-semantic-head-weight", type=float, default=0.25)
    parser.add_argument("--v8-frequency-head-weight", type=float, default=0.20)
    parser.add_argument("--v8-noise-head-weight", type=float, default=0.08)
    parser.add_argument("--v8-real-alpha-loss-weight", type=float, default=0.02)
    parser.add_argument("--v8-real-noise-push-weight", type=float, default=0.02)
    parser.add_argument("--v8-real-counterfactual-weight", type=float, default=0.02)
    parser.add_argument("--v8-hard-aigc-alpha-weight", type=float, default=0.02)
    parser.add_argument("--v8-real-counterfactual-margin", type=float, default=0.05)
    parser.add_argument("--v8-hard-aigc-alpha-target", type=float, default=0.15)
    parser.add_argument("--v8-hard-negative-mode", type=str, default="counterfactual", choices=["counterfactual", "probability"])
    parser.add_argument("--v8-anti-shortcut-buffer-ratio", type=float, default=0.15)
    parser.add_argument("--v81-controller-only", action="store_true")
    parser.add_argument("--v81-alpha-real-weight", type=float, default=0.05)
    parser.add_argument("--v81-alpha-easy-aigc-weight", type=float, default=0.03)
    parser.add_argument("--v81-alpha-hard-aigc-weight", type=float, default=0.02)
    parser.add_argument("--v81-alpha-easy-aigc-threshold", type=float, default=0.70)
    parser.add_argument("--v81-alpha-hard-target", type=float, default=0.50)
    parser.add_argument("--v81-real-residual-push-weight", type=float, default=0.05)
    parser.add_argument("--v81-real-counterfactual-weight", type=float, default=0.03)
    parser.add_argument("--v81-real-counterfactual-margin", type=float, default=0.05)
    parser.add_argument("--v81-hard-aigc-residual-min-weight", type=float, default=0.02)
    parser.add_argument("--v81-easy-aigc-residual-suppress-weight", type=float, default=0.02)
    parser.add_argument("--v81-residual-push-target", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
