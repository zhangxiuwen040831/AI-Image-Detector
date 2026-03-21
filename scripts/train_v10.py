from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire.augmentations import (  # noqa: E402
    build_aigc_clean_transform,
    build_aigc_hard_transform,
    build_aigc_mild_transform,
    build_eval_transform,
    build_real_balanced_negative_focus_transform,
    build_real_clean_transform,
    build_real_hard_negative_transform,
    build_train_transform,
    get_aigc_profile_probabilities,
    get_real_profile_probabilities,
)
from ai_image_detector.ntire.dataset import (  # noqa: E402
    BufferedTransformDataset,
    CurriculumBatchSampler,
    NTIRETrainDataset,
    build_train_val_indices,
    compute_fragile_aigc_score,
    print_dataset_sanity,
)
from ai_image_detector.ntire.model_v10 import V10CompetitionResetModel  # noqa: E402
from ai_image_detector.ntire.trainer_v10 import V10Trainer  # noqa: E402
from evaluate_v10 import evaluate_candidate  # noqa: E402
from mine_base_hard_reals import mine_base_hard_real_indices, parse_shards  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mine_fragile_aigc_indices(
    model: torch.nn.Module,
    device: torch.device,
    dataset: NTIRETrainDataset,
    train_indices: Sequence[int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    top_k: int,
    seed: int,
    max_probability: float = 0.80,
    output_path: Optional[Path] = None,
) -> Tuple[Set[int], Dict[str, object]]:
    aigc_indices = [idx for idx in train_indices if int(dataset.records[idx].label) == 1]
    if not aigc_indices:
        summary = {
            "candidate_count": 0,
            "selected_count": 0,
            "avg_score": 0.0,
            "avg_base_probability": 0.0,
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return set(), summary

    if max_samples > 0 and len(aigc_indices) > max_samples:
        rng = random.Random(seed)
        candidate_indices = sorted(rng.sample(aigc_indices, max_samples))
    else:
        candidate_indices = sorted(aigc_indices)

    eval_dataset = BufferedTransformDataset(
        base_dataset=dataset,
        transform=build_eval_transform(image_size=image_size),
    )
    loader = DataLoader(
        Subset(eval_dataset, candidate_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False,
    )

    base_model = model.module if hasattr(model, "module") else model
    original_mode = getattr(base_model, "inference_mode", "base_only")
    if hasattr(base_model, "set_inference_mode"):
        base_model.set_inference_mode("base_only")

    ranked: List[Dict[str, object]] = []
    model.eval()
    try:
        with torch.no_grad():
            for images, _, metadata in loader:
                images = images.to(device, non_blocking=True)
                out = model(images)
                base_logit = out["base_logit"].detach().view(-1)
                semantic_logit = out.get("semantic_logit")
                frequency_logit = out.get("freq_logit")
                scores = compute_fragile_aigc_score(
                    base_logit=base_logit,
                    semantic_logit=semantic_logit,
                    frequency_logit=frequency_logit,
                    max_probability=max_probability,
                ).cpu()
                base_probability = torch.sigmoid(base_logit).cpu()
                semantic_list = semantic_logit.detach().view(-1).cpu().tolist() if semantic_logit is not None else [None] * len(scores)
                frequency_list = frequency_logit.detach().view(-1).cpu().tolist() if frequency_logit is not None else [None] * len(scores)
                dataset_indices = metadata.get("dataset_index")
                if isinstance(dataset_indices, torch.Tensor):
                    dataset_indices = dataset_indices.cpu().tolist()
                image_names = list(metadata.get("image_name", []))
                image_paths = list(metadata.get("image_path", []))
                for dataset_index, image_name, image_path, score, base_prob, base_logit_i, semantic_i, frequency_i in zip(
                    dataset_indices,
                    image_names,
                    image_paths,
                    scores.tolist(),
                    base_probability.tolist(),
                    base_logit.cpu().tolist(),
                    semantic_list,
                    frequency_list,
                ):
                    ranked.append(
                        {
                            "dataset_index": int(dataset_index),
                            "image_name": str(image_name),
                            "image_path": str(image_path),
                            "score": float(score),
                            "base_probability": float(base_prob),
                            "base_logit": float(base_logit_i),
                            "semantic_logit": None if semantic_i is None else float(semantic_i),
                            "frequency_logit": None if frequency_i is None else float(frequency_i),
                        }
                    )
    finally:
        if hasattr(base_model, "set_inference_mode"):
            base_model.set_inference_mode(original_mode)

    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    selected_records = [item for item in ranked if float(item["score"]) > 0.0][: max(int(top_k), 0)]
    selected_indices = {int(item["dataset_index"]) for item in selected_records}
    summary = {
        "candidate_count": int(len(candidate_indices)),
        "selected_count": int(len(selected_records)),
        "top_k": int(top_k),
        "max_probability": float(max_probability),
        "avg_score": float(sum(float(item["score"]) for item in selected_records) / max(len(selected_records), 1)),
        "avg_base_probability": float(
            sum(float(item["base_probability"]) for item in selected_records) / max(len(selected_records), 1)
        ),
        "selected_preview": selected_records[:50],
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return selected_indices, summary


def _set_model_mode(model: torch.nn.Module, mode: str = "base_only") -> str:
    base_model = model.module if hasattr(model, "module") else model
    original_mode = getattr(base_model, "inference_mode", mode)
    if hasattr(base_model, "set_inference_mode"):
        base_model.set_inference_mode(mode)
    return str(original_mode)


def _collect_feature_records_for_indices(
    model: torch.nn.Module,
    device: torch.device,
    dataset: NTIRETrainDataset,
    candidate_indices: Sequence[int],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> List[Dict[str, object]]:
    if not candidate_indices:
        return []
    eval_dataset = BufferedTransformDataset(
        base_dataset=dataset,
        transform=build_eval_transform(image_size=image_size),
    )
    loader = DataLoader(
        Subset(eval_dataset, list(candidate_indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False,
    )
    original_mode = _set_model_mode(model, mode="base_only")
    rows: List[Dict[str, object]] = []
    model.eval()
    try:
        with torch.no_grad():
            for images, _, metadata in loader:
                images = images.to(device, non_blocking=True)
                out = model(images)
                base_feat = F.normalize(out["base_feat"].detach().float(), dim=1).cpu()
                base_logit = out["base_logit"].detach().view(-1).cpu()
                base_prob = torch.sigmoid(base_logit)
                semantic_logit = out["semantic_logit"].detach().view(-1).cpu()
                frequency_logit = out["freq_logit"].detach().view(-1).cpu()
                dataset_indices = metadata.get("dataset_index")
                if isinstance(dataset_indices, torch.Tensor):
                    dataset_indices = dataset_indices.cpu().tolist()
                image_names = list(metadata.get("image_name", []))
                image_paths = list(metadata.get("image_path", []))
                labels = list(metadata.get("label", []))
                for idx in range(base_feat.shape[0]):
                    rows.append(
                        {
                            "dataset_index": int(dataset_indices[idx]),
                            "image_name": str(image_names[idx]),
                            "image_path": str(image_paths[idx]),
                            "label": int(labels[idx]),
                            "base_feat": base_feat[idx].clone(),
                            "base_probability": float(base_prob[idx].item()),
                            "base_logit": float(base_logit[idx].item()),
                            "semantic_logit": float(semantic_logit[idx].item()),
                            "frequency_logit": float(frequency_logit[idx].item()),
                        }
                    )
    finally:
        _set_model_mode(model, mode=original_mode)
    return rows


def _collect_anchor_photo_features(
    model: torch.nn.Module,
    device: torch.device,
    photos_dir: Path,
    anchor_names: Sequence[str],
    image_size: int,
) -> List[Dict[str, object]]:
    transform = build_eval_transform(image_size=image_size)
    original_mode = _set_model_mode(model, mode="base_only")
    anchors: List[Dict[str, object]] = []
    model.eval()
    try:
        with torch.no_grad():
            for anchor_name in anchor_names:
                image_path = photos_dir / str(anchor_name)
                if not image_path.exists():
                    continue
                arr = np.array(Image.open(image_path).convert("RGB"))
                tensor = transform(image=arr)["image"].unsqueeze(0).to(device)
                out = model(tensor)
                anchors.append(
                    {
                        "anchor_name": str(anchor_name),
                        "image_path": str(image_path),
                        "base_feat": F.normalize(out["base_feat"].detach().float(), dim=1).cpu()[0].clone(),
                        "base_probability": float(torch.sigmoid(out["base_logit"].view(-1)).item()),
                        "base_logit": float(out["base_logit"].view(-1).item()),
                        "semantic_logit": float(out["semantic_logit"].view(-1).item()),
                        "frequency_logit": float(out["freq_logit"].view(-1).item()),
                    }
                )
    finally:
        _set_model_mode(model, mode=original_mode)
    return anchors


def mine_anchor_guided_real_indices(
    model: torch.nn.Module,
    device: torch.device,
    dataset: NTIRETrainDataset,
    train_indices: Sequence[int],
    photos_dir: Path,
    anchor_names: Sequence[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    top_k_per_anchor: int,
    seed: int,
    min_probability: float = 0.15,
    output_path: Optional[Path] = None,
) -> Tuple[Set[int], Dict[str, object]]:
    real_indices = [idx for idx in train_indices if int(dataset.records[idx].label) == 0]
    if max_samples > 0 and len(real_indices) > max_samples:
        rng = random.Random(seed)
        candidate_indices = sorted(rng.sample(real_indices, max_samples))
    else:
        candidate_indices = sorted(real_indices)
    anchor_rows = _collect_anchor_photo_features(
        model=model,
        device=device,
        photos_dir=photos_dir,
        anchor_names=anchor_names,
        image_size=image_size,
    )
    if not anchor_rows or not candidate_indices:
        summary = {
            "candidate_count": int(len(candidate_indices)),
            "selected_count": 0,
            "top_k_per_anchor": int(top_k_per_anchor),
            "min_probability": float(min_probability),
            "anchor_count": int(len(anchor_rows)),
            "anchors": [dict(anchor_name=str(item["anchor_name"]), image_path=str(item["image_path"])) for item in anchor_rows],
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return set(), summary

    candidate_rows = _collect_feature_records_for_indices(
        model=model,
        device=device,
        dataset=dataset,
        candidate_indices=candidate_indices,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    merged: Dict[int, Dict[str, object]] = {}
    per_anchor: List[Dict[str, object]] = []
    for anchor in anchor_rows:
        ranked: List[Dict[str, object]] = []
        anchor_feat = anchor["base_feat"]
        for item in candidate_rows:
            base_prob = float(item["base_probability"])
            if base_prob < float(min_probability):
                continue
            similarity = float(torch.dot(anchor_feat, item["base_feat"]).item())
            similarity = max(similarity, 0.0)
            disagreement = max(float(item["frequency_logit"]) - float(item["semantic_logit"]), 0.0)
            score = similarity * (0.5 + base_prob) * (1.0 + disagreement)
            if score <= 0.0:
                continue
            ranked.append(
                {
                    "anchor_name": str(anchor["anchor_name"]),
                    "dataset_index": int(item["dataset_index"]),
                    "image_name": str(item["image_name"]),
                    "image_path": str(item["image_path"]),
                    "score": float(score),
                    "similarity": float(similarity),
                    "base_probability": base_prob,
                    "base_logit": float(item["base_logit"]),
                    "semantic_logit": float(item["semantic_logit"]),
                    "frequency_logit": float(item["frequency_logit"]),
                    "disagreement": float(disagreement),
                }
            )
        ranked.sort(key=lambda row: float(row["score"]), reverse=True)
        selected = ranked[: max(int(top_k_per_anchor), 0)]
        per_anchor.append(
            {
                "anchor_name": str(anchor["anchor_name"]),
                "selected_count": int(len(selected)),
                "preview": selected[:25],
            }
        )
        for row in selected:
            existing = merged.get(int(row["dataset_index"]))
            if existing is None or float(row["score"]) > float(existing["score"]):
                merged[int(row["dataset_index"])] = row
    selected_records = sorted(merged.values(), key=lambda row: float(row["score"]), reverse=True)
    selected_indices = {int(row["dataset_index"]) for row in selected_records}
    summary = {
        "candidate_count": int(len(candidate_indices)),
        "selected_count": int(len(selected_records)),
        "top_k_per_anchor": int(top_k_per_anchor),
        "min_probability": float(min_probability),
        "anchor_count": int(len(anchor_rows)),
        "avg_score": float(sum(float(row["score"]) for row in selected_records) / max(len(selected_records), 1)),
        "avg_similarity": float(sum(float(row["similarity"]) for row in selected_records) / max(len(selected_records), 1)),
        "anchors": per_anchor,
        "selected_preview": selected_records[:50],
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return selected_indices, summary


def mine_anchor_guided_fragile_aigc_indices(
    model: torch.nn.Module,
    device: torch.device,
    dataset: NTIRETrainDataset,
    train_indices: Sequence[int],
    photos_dir: Path,
    anchor_names: Sequence[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    top_k_per_anchor: int,
    seed: int,
    max_probability: float = 0.80,
    output_path: Optional[Path] = None,
) -> Tuple[Set[int], Dict[str, object]]:
    aigc_indices = [idx for idx in train_indices if int(dataset.records[idx].label) == 1]
    if max_samples > 0 and len(aigc_indices) > max_samples:
        rng = random.Random(seed)
        candidate_indices = sorted(rng.sample(aigc_indices, max_samples))
    else:
        candidate_indices = sorted(aigc_indices)
    anchor_rows = _collect_anchor_photo_features(
        model=model,
        device=device,
        photos_dir=photos_dir,
        anchor_names=anchor_names,
        image_size=image_size,
    )
    if not anchor_rows or not candidate_indices:
        summary = {
            "candidate_count": int(len(candidate_indices)),
            "selected_count": 0,
            "top_k_per_anchor": int(top_k_per_anchor),
            "max_probability": float(max_probability),
            "anchor_count": int(len(anchor_rows)),
            "anchors": [dict(anchor_name=str(item["anchor_name"]), image_path=str(item["image_path"])) for item in anchor_rows],
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return set(), summary

    candidate_rows = _collect_feature_records_for_indices(
        model=model,
        device=device,
        dataset=dataset,
        candidate_indices=candidate_indices,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    merged: Dict[int, Dict[str, object]] = {}
    per_anchor: List[Dict[str, object]] = []
    for anchor in anchor_rows:
        ranked: List[Dict[str, object]] = []
        anchor_feat = anchor["base_feat"]
        for item in candidate_rows:
            base_prob = float(item["base_probability"])
            if base_prob > float(max_probability):
                continue
            similarity = float(torch.dot(anchor_feat, item["base_feat"]).item())
            similarity = max(similarity, 0.0)
            uncertainty = 4.0 * base_prob * (1.0 - base_prob)
            agreement = torch.sigmoid(
                torch.tensor(0.5 * (float(item["semantic_logit"]) + float(item["frequency_logit"])), dtype=torch.float32)
            ).item()
            score = similarity * uncertainty * (0.5 + float(agreement))
            if score <= 0.0:
                continue
            ranked.append(
                {
                    "anchor_name": str(anchor["anchor_name"]),
                    "dataset_index": int(item["dataset_index"]),
                    "image_name": str(item["image_name"]),
                    "image_path": str(item["image_path"]),
                    "score": float(score),
                    "similarity": float(similarity),
                    "base_probability": base_prob,
                    "base_logit": float(item["base_logit"]),
                    "semantic_logit": float(item["semantic_logit"]),
                    "frequency_logit": float(item["frequency_logit"]),
                }
            )
        ranked.sort(key=lambda row: float(row["score"]), reverse=True)
        selected = ranked[: max(int(top_k_per_anchor), 0)]
        per_anchor.append(
            {
                "anchor_name": str(anchor["anchor_name"]),
                "selected_count": int(len(selected)),
                "preview": selected[:25],
            }
        )
        for row in selected:
            existing = merged.get(int(row["dataset_index"]))
            if existing is None or float(row["score"]) > float(existing["score"]):
                merged[int(row["dataset_index"])] = row
    selected_records = sorted(merged.values(), key=lambda row: float(row["score"]), reverse=True)
    selected_indices = {int(row["dataset_index"]) for row in selected_records}
    summary = {
        "candidate_count": int(len(candidate_indices)),
        "selected_count": int(len(selected_records)),
        "top_k_per_anchor": int(top_k_per_anchor),
        "max_probability": float(max_probability),
        "anchor_count": int(len(anchor_rows)),
        "avg_score": float(sum(float(row["score"]) for row in selected_records) / max(len(selected_records), 1)),
        "avg_similarity": float(sum(float(row["similarity"]) for row in selected_records) / max(len(selected_records), 1)),
        "anchors": per_anchor,
        "selected_preview": selected_records[:50],
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return selected_indices, summary


def build_train_dataset(
    base_dataset: NTIRETrainDataset,
    image_size: int,
    jpeg_aligned_crop_p: float,
    disable_defocus: bool,
    disable_webp_compression: bool,
    disable_to_gray: bool,
    hard_real_indices: Optional[Sequence[int]] = None,
    anchor_hard_real_indices: Optional[Sequence[int]] = None,
    fragile_aigc_indices: Optional[Sequence[int]] = None,
) -> BufferedTransformDataset:
    train_transform = build_train_transform(
        image_size=image_size,
        jpeg_aligned_crop_p=jpeg_aligned_crop_p,
        enable_grayscale=not disable_to_gray,
        enable_defocus=not disable_defocus,
        enable_webp=not disable_webp_compression,
        chain_mix=True,
        chain_mix_strength="aigc_focus",
    )
    real_clean_transform = build_real_clean_transform(
        image_size=image_size,
        jpeg_aligned_crop_p=jpeg_aligned_crop_p,
        enable_webp=not disable_webp_compression,
    )
    real_mild_transform = build_real_balanced_negative_focus_transform(
        image_size=image_size,
        jpeg_aligned_crop_p=jpeg_aligned_crop_p,
        enable_defocus=not disable_defocus,
        enable_webp=not disable_webp_compression,
    )
    real_hard_transform = build_real_hard_negative_transform(
        image_size=image_size,
        jpeg_aligned_crop_p=jpeg_aligned_crop_p,
        enable_defocus=not disable_defocus,
        enable_webp=not disable_webp_compression,
    )
    aigc_clean_transform = build_aigc_clean_transform(
        image_size=image_size,
        jpeg_aligned_crop_p=jpeg_aligned_crop_p,
        enable_webp=not disable_webp_compression,
    )
    aigc_mild_transform = build_aigc_mild_transform(
        image_size=image_size,
        jpeg_aligned_crop_p=jpeg_aligned_crop_p,
        enable_webp=not disable_webp_compression,
    )
    aigc_hard_transform = build_aigc_hard_transform(
        image_size=image_size,
        jpeg_aligned_crop_p=jpeg_aligned_crop_p,
        enable_grayscale=not disable_to_gray,
        enable_defocus=not disable_defocus,
        enable_webp=not disable_webp_compression,
    )
    real_probs = get_real_profile_probabilities("standard_v10")
    hard_real_probs = get_real_profile_probabilities("hard_real_v10")
    anchor_hard_real_probs = get_real_profile_probabilities("anchor_hard_real_v101")
    aigc_probs = get_aigc_profile_probabilities("standard_v10")
    fragile_probs = get_aigc_profile_probabilities("fragile_v101")
    return BufferedTransformDataset(
        base_dataset=base_dataset,
        transform=train_transform,
        real_clean_transform=real_clean_transform,
        real_mild_transform=real_mild_transform,
        real_hard_transform=real_hard_transform,
        real_clean_prob=real_probs[0],
        real_mild_prob=real_probs[1],
        real_hard_prob=real_probs[2],
        hard_real_indices=hard_real_indices or set(),
        hard_real_clean_prob=hard_real_probs[0],
        hard_real_mild_prob=hard_real_probs[1],
        hard_real_hard_prob=hard_real_probs[2],
        anchor_hard_real_indices=anchor_hard_real_indices or set(),
        anchor_hard_real_clean_prob=anchor_hard_real_probs[0],
        anchor_hard_real_mild_prob=anchor_hard_real_probs[1],
        anchor_hard_real_hard_prob=anchor_hard_real_probs[2],
        aigc_clean_transform=aigc_clean_transform,
        aigc_mild_transform=aigc_mild_transform,
        aigc_hard_transform=aigc_hard_transform,
        aigc_clean_prob=aigc_probs[0],
        aigc_mild_prob=aigc_probs[1],
        aigc_hard_prob=aigc_probs[2],
        fragile_aigc_indices=fragile_aigc_indices or set(),
        fragile_aigc_clean_prob=fragile_probs[0],
        fragile_aigc_mild_prob=fragile_probs[1],
        fragile_aigc_hard_prob=fragile_probs[2],
    )


def evaluate_epoch_outputs(
    epoch: int,
    trainer: V10Trainer,
    val_loader: DataLoader,
    photos_dir: Path,
    photos_labels: Path,
    output_root: Path,
    image_size: int,
    default_threshold: float,
    evaluate_hybrid: bool,
) -> Dict[str, object]:
    eval_dir = output_root / f"eval_epoch_{epoch:03d}"
    base_eval = evaluate_candidate(
        model=trainer.model,
        device=trainer.device,
        mode="base_only",
        val_loader=val_loader,
        photos_dir=photos_dir,
        photos_labels=photos_labels,
        output_dir=eval_dir / "candidate_base_only",
        image_size=image_size,
        default_threshold=default_threshold,
    )
    summary: Dict[str, object] = {
        "epoch": int(epoch),
        "candidate_base_only": base_eval["report"],
    }
    if evaluate_hybrid:
        hybrid_eval = evaluate_candidate(
            model=trainer.model,
            device=trainer.device,
            mode="hybrid_optional",
            val_loader=val_loader,
            photos_dir=photos_dir,
            photos_labels=photos_labels,
            output_dir=eval_dir / "candidate_hybrid",
            image_size=image_size,
            default_threshold=default_threshold,
        )
        summary["candidate_hybrid"] = hybrid_eval["report"]
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _get_fixed_threshold_metrics(report: Dict[str, object], threshold: float) -> Dict[str, object]:
    fixed = report.get("photos_fixed_thresholds", {})
    return fixed.get(f"{float(threshold):.2f}", {})


def build_final_polish_best_selector() -> Any:
    def _selector(epoch_record: Dict[str, object]) -> Dict[str, object]:
        evaluation = epoch_record.get("evaluation") or {}
        candidate = evaluation.get("candidate_base_only") or {}
        photos_020 = (candidate.get("photos_fixed_thresholds") or {}).get("0.20", candidate.get("photos_default", {}))
        photos_030 = (candidate.get("photos_fixed_thresholds") or {}).get("0.30", {})
        val_default = candidate.get("val_default", {})
        key = [
            -float(photos_020.get("fn", 1e9)),
            -float(photos_020.get("fp", 1e9)),
            float(photos_020.get("f1", 0.0)),
            float(photos_030.get("f1", 0.0)),
            float(val_default.get("f1", 0.0)),
            float(val_default.get("auroc", 0.0)),
        ]
        return {
            "metric_key": "final_polish_lexicographic",
            "score": float(photos_020.get("f1", 0.0)),
            "key": key,
            "summary": {
                "photos_0.20": {
                    "precision": float(photos_020.get("precision", 0.0)),
                    "recall": float(photos_020.get("recall", 0.0)),
                    "f1": float(photos_020.get("f1", 0.0)),
                    "fp": int(float(photos_020.get("fp", 0.0))),
                    "fn": int(float(photos_020.get("fn", 0.0))),
                },
                "photos_0.30": {
                    "precision": float(photos_030.get("precision", 0.0)),
                    "recall": float(photos_030.get("recall", 0.0)),
                    "f1": float(photos_030.get("f1", 0.0)),
                    "fp": int(float(photos_030.get("fp", 0.0))),
                    "fn": int(float(photos_030.get("fn", 0.0))),
                },
                "val_default": {
                    "precision": float(val_default.get("precision", 0.0)),
                    "recall": float(val_default.get("recall", 0.0)),
                    "f1": float(val_default.get("f1", 0.0)),
                    "auroc": float(val_default.get("auroc", 0.0)),
                },
            },
        }

    return _selector


def build_final_polish_early_stop(patience: int = 2) -> Any:
    patience = max(int(patience), 1)

    def _early_stop(
        epoch_record: Dict[str, object],
        is_best: bool,
        no_improve_epochs: int,
        best_selection: Optional[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        del is_best, best_selection
        evaluation = epoch_record.get("evaluation") or {}
        candidate = evaluation.get("candidate_base_only") or {}
        photos_020 = (candidate.get("photos_fixed_thresholds") or {}).get("0.20", candidate.get("photos_default", {}))
        photos_030 = (candidate.get("photos_fixed_thresholds") or {}).get("0.30", {})
        if int(float(photos_020.get("fn", 1.0))) == 0 and int(float(photos_020.get("fp", 1e9))) <= 1:
            return {
                "reason": "photos_threshold_0.20_target_met",
                "epoch": int(epoch_record["epoch"]),
                "photos_0.20": photos_020,
            }
        if float(photos_030.get("precision", 0.0)) >= 0.90 and float(photos_030.get("recall", 0.0)) >= 1.0:
            return {
                "reason": "photos_threshold_0.30_balanced_target_met",
                "epoch": int(epoch_record["epoch"]),
                "photos_0.30": photos_030,
            }
        if no_improve_epochs >= patience:
            return {
                "reason": "no_improvement_patience_exhausted",
                "epoch": int(epoch_record["epoch"]),
                "no_improve_epochs": int(no_improve_epochs),
            }
        return None

    return _early_stop


def build_threshold_recommendation(
    report: Dict[str, object],
    best_checkpoint: Path,
) -> Dict[str, object]:
    val_thresholds = report.get("val_thresholds", {})
    return {
        "recommended_threshold_recall": 0.20,
        "recommended_threshold_balanced": float(
            ((val_thresholds.get("recall_ge_0_88") or {}).get("threshold", 0.30))
        ),
        "recommended_threshold_precision": float(
            ((val_thresholds.get("precision_ge_0_80") or {}).get("threshold", 0.55))
        ),
        "winner_checkpoint": str(best_checkpoint),
        "winner_mode": "base_only",
    }


def finalize_final_polish_artifacts(
    save_dir: Path,
    best_epoch: int,
    best_checkpoint: Path,
) -> Dict[str, object]:
    best_eval_dir = save_dir / f"eval_epoch_{int(best_epoch):03d}"
    summary_path = best_eval_dir / "summary.json"
    if not summary_path.exists():
        return {}
    best_eval_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate_base = dict(best_eval_summary.get("candidate_base_only", {}))
    export_files = [
        "photos_predictions.csv",
        "photos_threshold_sweep.csv",
        "val_predictions.csv",
        "val_threshold_sweep.csv",
        "report.json",
    ]
    candidate_dir = best_eval_dir / "candidate_base_only"
    copied_files: List[str] = []
    for name in export_files:
        src = candidate_dir / name
        if src.exists():
            shutil.copy2(src, save_dir / name)
            copied_files.append(name)
    root_summary = {
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_checkpoint),
        "candidate_base_only": candidate_base,
    }
    (save_dir / "summary.json").write_text(json.dumps(root_summary, indent=2), encoding="utf-8")
    threshold_recommendation = build_threshold_recommendation(candidate_base, best_checkpoint=best_checkpoint)
    candidate_summary = {
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_checkpoint),
        "winner_mode": "base_only",
        "photos_test_default": candidate_base.get("photos_default", {}),
        "photos_test_best_f1": (candidate_base.get("photos_thresholds") or {}).get("best_f1"),
        "heldout_val_default": candidate_base.get("val_default", {}),
        "threshold_recommendation": threshold_recommendation,
    }
    (save_dir / "candidate_summary.json").write_text(json.dumps(candidate_summary, indent=2), encoding="utf-8")
    (save_dir / "threshold_recommendation.json").write_text(
        json.dumps(threshold_recommendation, indent=2),
        encoding="utf-8",
    )
    return {
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_checkpoint),
        "copied_files": copied_files,
        "threshold_recommendation": threshold_recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V10 reset training pipeline.")
    parser.add_argument(
        "--phase",
        type=str,
        default="phase1_warmup",
        choices=["phase1_warmup", "phase2_curriculum", "phase3_competition", "phase4_final_polish"],
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--shards", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--semantic-trainable-layers", type=int, default=2)
    parser.add_argument("--phase3-semantic-trainable-layers", type=int, default=4)
    parser.add_argument("--frequency-dim", type=int, default=256)
    parser.add_argument("--noise-dim", type=int, default=256)
    parser.add_argument("--fused-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fusion-gate-input-dropout", type=float, default=0.1)
    parser.add_argument("--fusion-feature-dropout", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=0.35)
    parser.add_argument("--jpeg-aligned-crop-p", type=float, default=0.3)
    parser.add_argument("--disable-defocus", action="store_true")
    parser.add_argument("--disable-webp-compression", action="store_true")
    parser.add_argument("--disable-to-gray", action="store_true")
    parser.add_argument("--disable-hybrid", action="store_true")
    parser.add_argument("--hard-real-top-k", type=int, default=1200)
    parser.add_argument("--hard-real-max-samples", type=int, default=12000)
    parser.add_argument("--hard-real-min-probability", type=float, default=0.0)
    parser.add_argument("--hard-real-buffer-ratio", type=float, default=0.18)
    parser.add_argument("--anchor-hard-real-names", type=str, default="real1.jpg,real8.jpg")
    parser.add_argument("--anchor-hard-real-top-k-per-anchor", type=int, default=180)
    parser.add_argument("--anchor-hard-real-max-samples", type=int, default=12000)
    parser.add_argument("--anchor-hard-real-min-probability", type=float, default=0.15)
    parser.add_argument("--anchor-hard-real-buffer-ratio", type=float, default=0.12)
    parser.add_argument("--fragile-aigc-top-k", type=int, default=600)
    parser.add_argument("--fragile-aigc-max-samples", type=int, default=12000)
    parser.add_argument("--fragile-aigc-max-probability", type=float, default=0.80)
    parser.add_argument("--fragile-aigc-buffer-ratio", type=float, default=0.10)
    parser.add_argument("--fragile-anchor-names", type=str, default="aigc6.png,aigc7.png")
    parser.add_argument("--fragile-anchor-top-k-per-anchor", type=int, default=120)
    parser.add_argument("--fragile-anchor-max-samples", type=int, default=12000)
    parser.add_argument("--fragile-anchor-max-probability", type=float, default=0.80)
    parser.add_argument("--base-bce-weight", type=float, default=1.0)
    parser.add_argument("--semantic-aux-weight", type=float, default=0.20)
    parser.add_argument("--frequency-aux-weight", type=float, default=0.20)
    parser.add_argument("--noise-aux-weight", type=float, default=0.05)
    parser.add_argument("--hard-real-margin-weight", type=float, default=0.10)
    parser.add_argument("--hard-real-margin", type=float, default=0.25)
    parser.add_argument("--anchor-real-margin-weight", type=float, default=0.14)
    parser.add_argument("--anchor-real-margin", type=float, default=0.30)
    parser.add_argument("--prototype-margin-weight", type=float, default=0.05)
    parser.add_argument("--prototype-margin", type=float, default=0.15)
    parser.add_argument("--fragile-aigc-weight", type=float, default=0.05)
    parser.add_argument("--fragile-aigc-target-prob", type=float, default=0.20)
    parser.add_argument("--hybrid-main-weight", type=float, default=1.0)
    parser.add_argument("--base-support-weight", type=float, default=0.50)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--reset-best-metric-on-resume", action="store_true")
    parser.add_argument("--reset-history-on-resume", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--photos-dir", type=str, default=str(PROJECT_ROOT / "photos_test"))
    parser.add_argument("--photos-labels", type=str, default=str(PROJECT_ROOT / "photos_test" / "labels.csv"))
    parser.add_argument("--default-threshold", type=float, default=0.20)
    args = parser.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_dataset = NTIRETrainDataset(
        root_dir=args.data_root,
        shard_ids=parse_shards(args.shards),
        transform=None,
        strict=False,
    )
    print_dataset_sanity(base_dataset, max_rows=3)
    train_indices, val_indices, val_mode = build_train_val_indices(
        base_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Validation mode: {val_mode}")
    print(f"Train samples: {len(train_indices)} | Val samples: {len(val_indices)}")

    model = V10CompetitionResetModel(
        backbone_name=args.backbone_name,
        pretrained_backbone=args.pretrained_backbone,
        semantic_trainable_layers=0,
        image_size=args.image_size,
        frequency_dim=args.frequency_dim,
        noise_dim=args.noise_dim,
        fused_dim=args.fused_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        fusion_gate_input_dropout=args.fusion_gate_input_dropout,
        fusion_feature_dropout=args.fusion_feature_dropout,
        alpha_max=args.alpha_max,
        enable_noise_expert=not args.disable_hybrid,
    )
    trainer = V10Trainer(
        model=model,
        device=device,
        save_dir=save_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        base_bce_weight=args.base_bce_weight,
        semantic_aux_weight=args.semantic_aux_weight,
        frequency_aux_weight=args.frequency_aux_weight,
        noise_aux_weight=args.noise_aux_weight,
        hard_real_margin_weight=args.hard_real_margin_weight,
        hard_real_margin=args.hard_real_margin,
        anchor_real_margin_weight=args.anchor_real_margin_weight,
        anchor_real_margin=args.anchor_real_margin,
        prototype_margin_weight=args.prototype_margin_weight,
        prototype_margin=args.prototype_margin,
        fragile_aigc_weight=args.fragile_aigc_weight,
        fragile_aigc_target_prob=args.fragile_aigc_target_prob,
        hybrid_main_weight=args.hybrid_main_weight,
        base_support_weight=args.base_support_weight,
        checkpoint_interval=args.checkpoint_interval,
    )

    start_epoch = 0
    if args.resume:
        resume_meta = trainer.resume(args.resume)
        start_epoch = int(resume_meta["epoch"])
        print(f"Resumed from {args.resume}: {json.dumps(resume_meta, ensure_ascii=False)}")
        trainer.epochs = start_epoch + int(args.epochs)
        if args.reset_best_metric_on_resume or args.reset_history_on_resume:
            trainer.reset_tracking(clear_history=args.reset_history_on_resume)
            print(
                "Reset phase tracking after resume: "
                + json.dumps(
                    {
                        "reset_best_metric_on_resume": bool(args.reset_best_metric_on_resume),
                        "reset_history_on_resume": bool(args.reset_history_on_resume),
                    },
                    ensure_ascii=False,
                )
            )

    if args.phase == "phase1_warmup":
        semantic_layers = 0
    elif args.phase in {"phase2_curriculum", "phase4_final_polish"}:
        semantic_layers = int(args.semantic_trainable_layers)
    else:
        semantic_layers = int(args.phase3_semantic_trainable_layers)
    phase_summary = trainer.set_phase(args.phase, semantic_trainable_layers=semantic_layers)
    print(f"Phase config: {json.dumps(phase_summary, ensure_ascii=False)}")

    hard_real_indices: Set[int] = set()
    hard_real_summary: Dict[str, object] = {"selected_count": 0}
    anchor_hard_real_indices: Set[int] = set()
    anchor_hard_real_summary: Dict[str, object] = {"selected_count": 0}
    fragile_aigc_indices: Set[int] = set()
    fragile_aigc_summary: Dict[str, object] = {"selected_count": 0}
    photos_dir = Path(args.photos_dir)
    photos_labels = Path(args.photos_labels)
    anchor_real_names = [name.strip() for name in str(args.anchor_hard_real_names).split(",") if name.strip()]
    fragile_anchor_names = [name.strip() for name in str(args.fragile_anchor_names).split(",") if name.strip()]
    if args.phase in {"phase2_curriculum", "phase3_competition", "phase4_final_polish"}:
        hard_real_indices, hard_real_summary = mine_base_hard_real_indices(
            model=trainer.model,
            device=trainer.device,
            dataset=base_dataset,
            train_indices=train_indices,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.hard_real_max_samples,
            top_k=args.hard_real_top_k,
            seed=args.seed,
            min_probability=args.hard_real_min_probability,
            output_path=save_dir / "hard_real_buffer.json",
        )
        hard_real_indices = set(train_indices).intersection(hard_real_indices)
        if args.phase == "phase4_final_polish":
            anchor_hard_real_indices, anchor_hard_real_summary = mine_anchor_guided_real_indices(
                model=trainer.model,
                device=trainer.device,
                dataset=base_dataset,
                train_indices=train_indices,
                photos_dir=photos_dir,
                anchor_names=anchor_real_names,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.anchor_hard_real_max_samples,
                top_k_per_anchor=args.anchor_hard_real_top_k_per_anchor,
                seed=args.seed,
                min_probability=args.anchor_hard_real_min_probability,
                output_path=save_dir / "anchor_hard_real_buffer.json",
            )
            anchor_hard_real_indices = set(train_indices).intersection(anchor_hard_real_indices)
            fragile_aigc_indices, fragile_aigc_summary = mine_anchor_guided_fragile_aigc_indices(
                model=trainer.model,
                device=trainer.device,
                dataset=base_dataset,
                train_indices=train_indices,
                photos_dir=photos_dir,
                anchor_names=fragile_anchor_names,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.fragile_anchor_max_samples,
                top_k_per_anchor=args.fragile_anchor_top_k_per_anchor,
                seed=args.seed,
                max_probability=args.fragile_anchor_max_probability,
                output_path=save_dir / "fragile_aigc_buffer.json",
            )
        else:
            fragile_aigc_indices, fragile_aigc_summary = mine_fragile_aigc_indices(
                model=trainer.model,
                device=trainer.device,
                dataset=base_dataset,
                train_indices=train_indices,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.fragile_aigc_max_samples,
                top_k=args.fragile_aigc_top_k,
                seed=args.seed,
                max_probability=args.fragile_aigc_max_probability,
                output_path=save_dir / "fragile_aigc_buffer.json",
            )
        fragile_aigc_indices = set(train_indices).intersection(fragile_aigc_indices)

    train_dataset = build_train_dataset(
        base_dataset=base_dataset,
        image_size=args.image_size,
        jpeg_aligned_crop_p=args.jpeg_aligned_crop_p,
        disable_defocus=args.disable_defocus,
        disable_webp_compression=args.disable_webp_compression,
        disable_to_gray=args.disable_to_gray,
        hard_real_indices=hard_real_indices,
        anchor_hard_real_indices=anchor_hard_real_indices,
        fragile_aigc_indices=fragile_aigc_indices,
    )
    eval_dataset = BufferedTransformDataset(
        base_dataset=base_dataset,
        transform=build_eval_transform(args.image_size),
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False,
    )

    if args.phase == "phase1_warmup":
        train_loader = DataLoader(
            Subset(train_dataset, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if args.num_workers > 0 else False,
            drop_last=False,
        )
    else:
        curriculum_sampler = CurriculumBatchSampler(
            primary_indices=train_indices,
            hard_real_indices=sorted(hard_real_indices),
            anchor_hard_real_indices=sorted(anchor_hard_real_indices),
            fragile_aigc_indices=sorted(fragile_aigc_indices),
            batch_size=args.batch_size,
            hard_real_ratio=args.hard_real_buffer_ratio,
            anchor_hard_real_ratio=args.anchor_hard_real_buffer_ratio,
            fragile_aigc_ratio=args.fragile_aigc_buffer_ratio,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=curriculum_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if args.num_workers > 0 else False,
        )

    train_config = vars(args).copy()
    train_config["device"] = str(device)
    train_config["phase_summary"] = phase_summary
    train_config["hard_real_summary"] = hard_real_summary
    train_config["anchor_hard_real_summary"] = anchor_hard_real_summary
    train_config["fragile_aigc_summary"] = fragile_aigc_summary
    train_config["anchor_hard_real_names"] = anchor_real_names
    train_config["fragile_anchor_names"] = fragile_anchor_names
    train_config["train_count"] = len(train_indices)
    train_config["val_count"] = len(val_indices)
    train_config["val_mode"] = val_mode
    train_config["hard_real_ratio"] = float(args.hard_real_buffer_ratio)
    train_config["anchor_hard_real_ratio"] = float(args.anchor_hard_real_buffer_ratio)
    train_config["fragile_aigc_ratio"] = float(args.fragile_aigc_buffer_ratio)
    (save_dir / "train_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    def _eval_callback(epoch: int, trainer: V10Trainer) -> Optional[Dict[str, object]]:
        if epoch % max(int(args.eval_every), 1) != 0:
            return None
        if (not photos_dir.exists()) or (not photos_labels.exists()):
            return None
        return evaluate_epoch_outputs(
            epoch=epoch,
            trainer=trainer,
            val_loader=val_loader,
            photos_dir=photos_dir,
            photos_labels=photos_labels,
            output_root=save_dir,
            image_size=args.image_size,
            default_threshold=args.default_threshold,
            evaluate_hybrid=(args.phase == "phase3_competition" and not args.disable_hybrid),
        )

    best_selector = build_final_polish_best_selector() if args.phase == "phase4_final_polish" else None
    early_stop_fn = (
        build_final_polish_early_stop(patience=args.early_stop_patience)
        if args.phase == "phase4_final_polish"
        else None
    )
    fit_summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=start_epoch,
        eval_callback=_eval_callback,
        best_selector=best_selector,
        early_stop_fn=early_stop_fn,
    )

    final_export: Dict[str, object] = {}
    if args.phase == "phase4_final_polish" and trainer.best_epoch > 0:
        final_export = finalize_final_polish_artifacts(
            save_dir=save_dir,
            best_epoch=trainer.best_epoch,
            best_checkpoint=save_dir / "best.pth",
        )

    final_summary = {
        "phase": args.phase,
        "resume": args.resume,
        "fit_summary": fit_summary,
        "phase_summary": phase_summary,
        "hard_real_summary": hard_real_summary,
        "anchor_hard_real_summary": anchor_hard_real_summary,
        "fragile_aigc_summary": fragile_aigc_summary,
        "final_export": final_export,
    }
    (save_dir / "phase_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
