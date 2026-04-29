from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire.augmentations import build_eval_transform
from ai_image_detector.ntire.dataset import (  # noqa: E402
    BufferedTransformDataset,
    NTIRETrainDataset,
    build_train_val_indices,
    compute_base_hard_real_score,
)
from ai_image_detector.ntire.model import HybridAIGCDetector  # noqa: E402


def parse_shards(shards: Optional[str]) -> Optional[List[int]]:
    if shards is None or not str(shards).strip():
        return None
    parsed = []
    for part in str(shards).split(","):
        part = part.strip()
        if part:
            parsed.append(int(part))
    return parsed or None


def build_v9_model_from_checkpoint(
    checkpoint: Path,
    device: torch.device,
    backbone_name: str = "vit_base_patch16_clip_224.openai",
    image_size: int = 224,
    fused_dim: int = 512,
    head_hidden_dim: int = 256,
    dropout: float = 0.3,
) -> Tuple[HybridAIGCDetector, Dict[str, object]]:
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model = HybridAIGCDetector(
        backbone_name=backbone_name,
        pretrained_backbone=False,
        backbone_trainable_layers=0,
        image_size=image_size,
        fused_dim=fused_dim,
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
        use_aux_heads=True,
        enable_base_residual_fusion=True,
        v8_stage="residual_finetune",
    )
    current_state = model.state_dict()
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    load_result = model.load_state_dict(filtered_state, strict=False)
    model.to(device).eval()
    meta = {
        "epoch": int(ckpt.get("epoch", 0)),
        "temperature": float(ckpt.get("temperature", 1.0)),
        "loaded_keys": int(len(filtered_state)),
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
    }
    return model, meta


def mine_base_hard_real_indices(
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
    min_probability: float = 0.0,
    output_path: Optional[Path] = None,
) -> Tuple[Set[int], Dict[str, object]]:
    real_indices = [idx for idx in train_indices if int(dataset.records[idx].label) == 0]
    if not real_indices:
        summary = {
            "selected_count": 0,
            "candidate_count": 0,
            "avg_score": 0.0,
            "avg_base_probability": 0.0,
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return set(), summary

    if max_samples > 0 and len(real_indices) > max_samples:
        rng = random.Random(seed)
        candidate_indices = sorted(rng.sample(real_indices, max_samples))
    else:
        candidate_indices = sorted(real_indices)

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
    original_mode = getattr(base_model, "inference_mode", "legacy")
    if hasattr(base_model, "set_inference_mode"):
        base_model.set_inference_mode("tri_fusion")

    ranked: List[Dict[str, object]] = []
    model.eval()
    try:
        with torch.no_grad():
            for images, _, metadata in loader:
                images = images.to(device, non_blocking=True)
                out = model(images)
                base_logit = out.get("tri_fusion_logit", out["logit"]).detach().view(-1)
                semantic_logit = out.get("semantic_logit")
                frequency_logit = out.get("freq_logit")
                noise_logit = out.get("noise_logit")
                scores = compute_base_hard_real_score(
                    base_logit=base_logit,
                    semantic_logit=semantic_logit,
                    frequency_logit=frequency_logit,
                    noise_logit=noise_logit,
                    min_probability=min_probability,
                ).cpu()
                base_probability = torch.sigmoid(base_logit).cpu()
                semantic_list = (
                    semantic_logit.detach().view(-1).cpu().tolist()
                    if semantic_logit is not None
                    else [None] * len(scores)
                )
                frequency_list = (
                    frequency_logit.detach().view(-1).cpu().tolist()
                    if frequency_logit is not None
                    else [None] * len(scores)
                )
                noise_list = (
                    noise_logit.detach().view(-1).cpu().tolist()
                    if noise_logit is not None
                    else [None] * len(scores)
                )
                dataset_indices = metadata.get("dataset_index")
                if isinstance(dataset_indices, torch.Tensor):
                    dataset_indices = dataset_indices.cpu().tolist()
                image_names = list(metadata.get("image_name", []))
                image_paths = list(metadata.get("image_path", []))
                for dataset_index, image_name, image_path, score, base_prob, base_logit_i, semantic_i, frequency_i, noise_i in zip(
                    dataset_indices,
                    image_names,
                    image_paths,
                    scores.tolist(),
                    base_probability.tolist(),
                    base_logit.cpu().tolist(),
                    semantic_list,
                    frequency_list,
                    noise_list,
                ):
                    disagreement = None
                    branch_values = [float(value) for value in (semantic_i, frequency_i, noise_i) if value is not None]
                    if len(branch_values) >= 2:
                        disagreement = max(branch_values) - min(branch_values)
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
                            "noise_logit": None if noise_i is None else float(noise_i),
                            "disagreement": disagreement,
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
        "min_probability": float(min_probability),
        "avg_score": float(sum(float(item["score"]) for item in selected_records) / max(len(selected_records), 1)),
        "avg_base_probability": float(
            sum(float(item["base_probability"]) for item in selected_records) / max(len(selected_records), 1)
        ),
        "avg_disagreement": float(
            sum(float(item["disagreement"] or 0.0) for item in selected_records) / max(len(selected_records), 1)
        ),
        "selected_preview": selected_records[:50],
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return selected_indices, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine V9 hard-real buffer candidates from base logit.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--shards", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--top-k", type=int, default=1200)
    parser.add_argument("--min-probability", type=float, default=0.0)
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--fused-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NTIRETrainDataset(
        root_dir=args.data_root,
        shard_ids=parse_shards(args.shards),
        transform=None,
        strict=False,
    )
    train_indices, _, _ = build_train_val_indices(dataset, val_ratio=args.val_ratio, seed=args.seed)
    model, checkpoint_meta = build_v9_model_from_checkpoint(
        checkpoint=Path(args.checkpoint),
        device=device,
        backbone_name=args.backbone_name,
        image_size=args.image_size,
        fused_dim=args.fused_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    )
    selected_indices, summary = mine_base_hard_real_indices(
        model=model,
        device=device,
        dataset=dataset,
        train_indices=train_indices,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        top_k=args.top_k,
        seed=args.seed,
        min_probability=args.min_probability,
        output_path=Path(args.output),
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": checkpoint_meta,
        "selected_count": int(len(selected_indices)),
        "summary": summary,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
