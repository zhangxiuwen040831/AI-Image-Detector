from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire.augmentations import build_eval_transform  # noqa: E402
from ai_image_detector.ntire.dataset import BufferedTransformDataset, NTIRETrainDataset, build_train_val_indices  # noqa: E402
from ai_image_detector.ntire.metrics import compute_metrics  # noqa: E402
from mine_base_hard_reals import build_v9_model_from_checkpoint, parse_shards  # noqa: E402


def threshold_sweep(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold_start: float = 0.10,
    threshold_end: float = 0.60,
    threshold_step: float = 0.05,
) -> pd.DataFrame:
    thresholds = np.arange(threshold_start, threshold_end + 1e-9, threshold_step)
    rows: List[Dict[str, float]] = []
    y_true_list = [int(x) for x in y_true]
    y_prob_list = [float(x) for x in y_prob]
    for threshold in thresholds:
        metrics = compute_metrics(y_true_list, y_prob_list, threshold=float(threshold))
        preds = [1 if float(prob) >= float(threshold) else 0 for prob in y_prob_list]
        tp = sum(int(yt == 1 and yp == 1) for yt, yp in zip(y_true_list, preds))
        fp = sum(int(yt == 0 and yp == 1) for yt, yp in zip(y_true_list, preds))
        tn = sum(int(yt == 0 and yp == 0) for yt, yp in zip(y_true_list, preds))
        fn = sum(int(yt == 1 and yp == 0) for yt, yp in zip(y_true_list, preds))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "auroc": float(metrics.get("auroc", float("nan"))),
                "auprc": float(metrics.get("auprc", float("nan"))),
                "ece": float(metrics.get("ece", float("nan"))),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
    return pd.DataFrame(rows)


def summarize_thresholds(sweep_df: pd.DataFrame) -> Dict[str, object]:
    if sweep_df.empty:
        return {
            "best_f1": None,
            "recall_ge_0_88": None,
            "precision_ge_0_80": None,
        }
    best_f1_row = sweep_df.iloc[int(sweep_df["f1"].idxmax())]
    recall_candidates = sweep_df[sweep_df["recall"] >= 0.88]
    precision_candidates = sweep_df[sweep_df["precision"] >= 0.80]
    recall_row = (
        recall_candidates.sort_values(["precision", "f1", "threshold"], ascending=[False, False, True]).iloc[0]
        if not recall_candidates.empty
        else None
    )
    precision_row = (
        precision_candidates.sort_values(["f1", "recall", "threshold"], ascending=[False, False, True]).iloc[0]
        if not precision_candidates.empty
        else None
    )
    return {
        "best_f1": best_f1_row.to_dict(),
        "recall_ge_0_88": None if recall_row is None else recall_row.to_dict(),
        "precision_ge_0_80": None if precision_row is None else precision_row.to_dict(),
    }


def records_to_metrics(
    records: List[Dict[str, object]],
    threshold: float,
) -> Dict[str, object]:
    y_true = [int(record["label"]) for record in records]
    y_prob = [float(record["probability"]) for record in records]
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    preds = [1 if float(prob) >= threshold else 0 for prob in y_prob]
    false_positive = [record["image_name"] for record, pred in zip(records, preds) if int(record["label"]) == 0 and pred == 1]
    false_negative = [record["image_name"] for record, pred in zip(records, preds) if int(record["label"]) == 1 and pred == 0]
    tp = sum(int(yt == 1 and yp == 1) for yt, yp in zip(y_true, preds))
    fp = sum(int(yt == 0 and yp == 1) for yt, yp in zip(y_true, preds))
    tn = sum(int(yt == 0 and yp == 0) for yt, yp in zip(y_true, preds))
    fn = sum(int(yt == 1 and yp == 0) for yt, yp in zip(y_true, preds))
    return {
        "threshold": float(threshold),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
        "auroc": float(metrics.get("auroc", float("nan"))),
        "auprc": float(metrics.get("auprc", float("nan"))),
        "ece": float(metrics.get("ece", float("nan"))),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def set_model_mode(model: torch.nn.Module, mode: str) -> None:
    base_model = model.module if hasattr(model, "module") else model
    if hasattr(base_model, "set_inference_mode"):
        base_model.set_inference_mode(mode)


def collect_records_from_subset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    temperature: float,
) -> List[Dict[str, object]]:
    set_model_mode(model, mode)
    model.eval()
    records: List[Dict[str, object]] = []
    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device, non_blocking=True)
            out = model(images)
            logits_tensor = out["logit"].detach().view(-1)
            base_logits_tensor = out.get("base_only_logit", out.get("base_logit", out["logit"])).detach().view(-1)
            hybrid_logits_tensor = out.get("hybrid_logit", out["logit"]).detach().view(-1)
            probs = torch.sigmoid(logits_tensor / max(float(temperature), 1e-6)).cpu().tolist()
            logits = logits_tensor.cpu().tolist()
            base_probs = torch.sigmoid(base_logits_tensor / max(float(temperature), 1e-6)).cpu().tolist()
            hybrid_probs = torch.sigmoid(hybrid_logits_tensor / max(float(temperature), 1e-6)).cpu().tolist()
            base_logits = base_logits_tensor.cpu().tolist()
            hybrid_logits = hybrid_logits_tensor.cpu().tolist()
            alpha_active = out.get("alpha_active", out.get("alpha", torch.zeros_like(out["logit"]))).detach().view(-1).cpu().tolist()
            noise_delta = out.get("noise_delta_logit", torch.zeros_like(out["logit"])).detach().view(-1).cpu().tolist()
            semantic_logits = out.get("semantic_logit")
            frequency_logits = out.get("freq_logit")
            fusion_weights = out.get("fusion_weights")
            semantic_values = semantic_logits.detach().view(-1).cpu().tolist() if semantic_logits is not None else [None] * len(probs)
            frequency_values = frequency_logits.detach().view(-1).cpu().tolist() if frequency_logits is not None else [None] * len(probs)
            fusion_values = fusion_weights.detach().cpu().tolist() if fusion_weights is not None else [[None, None, None]] * len(probs)
            image_names = list(metadata.get("image_name", []))
            image_paths = list(metadata.get("image_path", []))
            label_values = labels.detach().view(-1).cpu().tolist()
            for image_name, image_path, label, prob, logit, base_prob, hybrid_prob, base_logit, hybrid_logit, alpha, noise_delta_logit, semantic_logit, frequency_logit, fusion in zip(
                image_names,
                image_paths,
                label_values,
                probs,
                logits,
                base_probs,
                hybrid_probs,
                base_logits,
                hybrid_logits,
                alpha_active,
                noise_delta,
                semantic_values,
                frequency_values,
                fusion_values,
            ):
                records.append(
                    {
                        "image_name": str(image_name),
                        "image_path": str(image_path),
                        "label": int(label),
                        "probability": float(prob),
                        "logit": float(logit),
                        "base_probability": float(base_prob),
                        "hybrid_probability": float(hybrid_prob),
                        "base_logit": float(base_logit),
                        "hybrid_logit": float(hybrid_logit),
                        "alpha": float(alpha),
                        "noise_delta_logit": float(noise_delta_logit),
                        "semantic_logit": None if semantic_logit is None else float(semantic_logit),
                        "frequency_logit": None if frequency_logit is None else float(frequency_logit),
                        "fusion_semantic": None if fusion[0] is None else float(fusion[0]),
                        "fusion_frequency": None if fusion[1] is None else float(fusion[1]),
                        "fusion_noise": None if fusion[2] is None else float(fusion[2]),
                        "mode": mode,
                    }
                )
    return records


def collect_records_from_folder(
    model: torch.nn.Module,
    folder: Path,
    labels_csv: Path,
    device: torch.device,
    mode: str,
    image_size: int,
    temperature: float,
) -> List[Dict[str, object]]:
    labels_df = pd.read_csv(labels_csv)
    labels_map = {str(row["image_name"]): int(row["label"]) for _, row in labels_df.iterrows()}
    transform = build_eval_transform(image_size=image_size)
    set_model_mode(model, mode)
    model.eval()
    records: List[Dict[str, object]] = []
    image_paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        image_paths.extend(sorted(folder.glob(pattern)))
    with torch.no_grad():
        for image_path in image_paths:
            if image_path.name not in labels_map:
                continue
            arr = np.array(Image.open(image_path).convert("RGB"))
            tensor = transform(image=arr)["image"].unsqueeze(0).to(device)
            out = model(tensor)
            final_logit = out["logit"].view(-1)
            base_logit = out.get("base_only_logit", out.get("base_logit", out["logit"])).view(-1)
            hybrid_logit = out.get("hybrid_logit", out["logit"]).view(-1)
            fusion = out.get("fusion_weights")
            fusion_row = fusion.detach().cpu().view(-1, 3).tolist()[0] if fusion is not None else [None, None, None]
            records.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "label": int(labels_map[image_path.name]),
                    "probability": float(torch.sigmoid(final_logit / max(float(temperature), 1e-6)).item()),
                    "logit": float(final_logit.item()),
                    "base_probability": float(torch.sigmoid(base_logit / max(float(temperature), 1e-6)).item()),
                    "hybrid_probability": float(torch.sigmoid(hybrid_logit / max(float(temperature), 1e-6)).item()),
                    "base_logit": float(base_logit.item()),
                    "hybrid_logit": float(hybrid_logit.item()),
                    "alpha": float(out.get("alpha_active", out.get("alpha", torch.zeros_like(out["logit"]))).item()),
                    "noise_delta_logit": float(out.get("noise_delta_logit", torch.zeros_like(out["logit"])).item()),
                    "semantic_logit": float(out["semantic_logit"].item()) if "semantic_logit" in out else None,
                    "frequency_logit": float(out["freq_logit"].item()) if "freq_logit" in out else None,
                    "fusion_semantic": None if fusion_row[0] is None else float(fusion_row[0]),
                    "fusion_frequency": None if fusion_row[1] is None else float(fusion_row[1]),
                    "fusion_noise": None if fusion_row[2] is None else float(fusion_row[2]),
                    "mode": mode,
                }
            )
    return records


def evaluate_candidate(
    model: torch.nn.Module,
    device: torch.device,
    mode: str,
    val_loader: DataLoader,
    photos_dir: Path,
    photos_labels: Path,
    output_dir: Path,
    image_size: int,
    default_threshold: float,
    temperature: float,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    val_records = collect_records_from_subset(
        model=model,
        loader=val_loader,
        device=device,
        mode=mode,
        temperature=temperature,
    )
    photos_records = collect_records_from_folder(
        model=model,
        folder=photos_dir,
        labels_csv=photos_labels,
        device=device,
        mode=mode,
        image_size=image_size,
        temperature=temperature,
    )

    val_predictions = pd.DataFrame(val_records)
    photos_predictions = pd.DataFrame(photos_records)
    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    photos_predictions.to_csv(output_dir / "photos_predictions.csv", index=False)

    val_y_true = [int(record["label"]) for record in val_records]
    val_y_prob = [float(record["probability"]) for record in val_records]
    photos_y_true = [int(record["label"]) for record in photos_records]
    photos_y_prob = [float(record["probability"]) for record in photos_records]
    val_sweep = threshold_sweep(val_y_true, val_y_prob)
    photos_sweep = threshold_sweep(photos_y_true, photos_y_prob)
    val_sweep.to_csv(output_dir / "val_threshold_sweep.csv", index=False)
    photos_sweep.to_csv(output_dir / "photos_threshold_sweep.csv", index=False)

    val_default = records_to_metrics(val_records, threshold=default_threshold)
    photos_default = records_to_metrics(photos_records, threshold=default_threshold)
    report = {
        "mode": mode,
        "default_threshold": float(default_threshold),
        "val_default": val_default,
        "photos_default": photos_default,
        "val_thresholds": summarize_thresholds(val_sweep),
        "photos_thresholds": summarize_thresholds(photos_sweep),
        "photos_false_positive": photos_default["false_positive"],
        "photos_false_negative": photos_default["false_negative"],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def choose_industrial_candidate(
    base_report: Dict[str, object],
    hybrid_report: Dict[str, object],
) -> Dict[str, object]:
    candidates = [
        {"mode": "base_only", "report": base_report},
        {"mode": "hybrid", "report": hybrid_report},
    ]
    candidates.sort(
        key=lambda item: (
            int(item["report"]["photos_default"]["fp"]),  # type: ignore[index]
            -float(item["report"]["photos_default"]["precision"]),  # type: ignore[index]
            -float(item["report"]["photos_default"]["recall"]),  # type: ignore[index]
            -float(item["report"]["photos_default"]["f1"]),  # type: ignore[index]
        )
    )
    winner = candidates[0]
    return {
        "mode": winner["mode"],
        "photos_default": winner["report"]["photos_default"],
        "reason": "lower photos_test false positives with better precision/recall tradeoff",
    }


def choose_competition_candidate(
    base_report: Dict[str, object],
    hybrid_report: Dict[str, object],
) -> Dict[str, object]:
    base_best_f1 = base_report["photos_thresholds"]["best_f1"]  # type: ignore[index]
    hybrid_best_f1 = hybrid_report["photos_thresholds"]["best_f1"]  # type: ignore[index]
    if hybrid_best_f1 is not None and base_best_f1 is not None:
        hybrid_f1 = float(hybrid_best_f1["f1"])
        base_f1 = float(base_best_f1["f1"])
        hybrid_precision = float(hybrid_best_f1["precision"])
        base_precision = float(base_best_f1["precision"])
        if hybrid_f1 > base_f1 and hybrid_precision >= (base_precision - 0.03):
            return {
                "mode": "hybrid",
                "photos_best_f1": hybrid_best_f1,
                "reason": "hybrid improves best-F1 without paying a large precision penalty",
            }
    return {
        "mode": "base_only",
        "photos_best_f1": base_best_f1,
        "reason": "hybrid does not deliver a clear best-F1 gain over base-only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate V9 base_only vs hybrid candidates.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--photos-dir", type=str, default=str(PROJECT_ROOT / "photos_test"))
    parser.add_argument("--photos-labels", type=str, default=str(PROJECT_ROOT / "photos_test" / "labels.csv"))
    parser.add_argument("--shards", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--default-threshold", type=float, default=0.2)
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--fused-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = NTIRETrainDataset(
        root_dir=args.data_root,
        shard_ids=parse_shards(args.shards),
        transform=None,
        strict=False,
    )
    _, val_indices, val_mode = build_train_val_indices(dataset, val_ratio=args.val_ratio, seed=args.seed)
    eval_dataset = BufferedTransformDataset(base_dataset=dataset, transform=build_eval_transform(args.image_size))
    val_loader = DataLoader(
        Subset(eval_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False,
    )

    model, checkpoint_meta = build_v9_model_from_checkpoint(
        checkpoint=Path(args.checkpoint),
        device=device,
        backbone_name=args.backbone_name,
        image_size=args.image_size,
        fused_dim=args.fused_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    )
    base_report = evaluate_candidate(
        model=model,
        device=device,
        mode="base_only",
        val_loader=val_loader,
        photos_dir=Path(args.photos_dir),
        photos_labels=Path(args.photos_labels),
        output_dir=output_dir / "base_only",
        image_size=args.image_size,
        default_threshold=args.default_threshold,
        temperature=float(checkpoint_meta.get("temperature", 1.0)),
    )
    hybrid_report = evaluate_candidate(
        model=model,
        device=device,
        mode="hybrid",
        val_loader=val_loader,
        photos_dir=Path(args.photos_dir),
        photos_labels=Path(args.photos_labels),
        output_dir=output_dir / "hybrid",
        image_size=args.image_size,
        default_threshold=args.default_threshold,
        temperature=float(checkpoint_meta.get("temperature", 1.0)),
    )
    winner_industrial = choose_industrial_candidate(base_report=base_report, hybrid_report=hybrid_report)
    winner_competition = choose_competition_candidate(base_report=base_report, hybrid_report=hybrid_report)
    industrial_default = winner_industrial["photos_default"]
    ready_for_large_scale_train = bool(
        int(industrial_default["fp"]) <= 5
        and float(industrial_default["precision"]) > 0.5294
        and float(industrial_default["recall"]) >= 0.88
    )
    summary = {
        "v9_plan": {
            "bottleneck": "base_path_boundary",
            "industrial_goal": "base_only_or_base_dominant",
            "competition_goal": "hybrid_only_if_it_truly_beats_base_only",
        },
        "checkpoint_meta": checkpoint_meta,
        "validation_mode": val_mode,
        "candidate_base_only": base_report,
        "candidate_hybrid": hybrid_report,
        "winner_industrial": winner_industrial,
        "winner_competition": winner_competition,
        "decision": {
            "ready_for_large_scale_train": ready_for_large_scale_train,
            "ready_for_industrial_export": bool(
                winner_industrial["mode"] == "base_only"
                and int(industrial_default["fp"]) <= 5
                and float(industrial_default["precision"]) >= 0.60
            ),
            "reason": (
                "Industrial candidate meets FP/precision/recall gate."
                if ready_for_large_scale_train
                else "Need more base-path debiasing before longer training or export."
            ),
        },
    }
    summary_path = output_dir / "candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
