from __future__ import annotations

import argparse
import json
import math
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
from ai_image_detector.ntire.model_v10 import V10CompetitionResetModel  # noqa: E402


def parse_shards(shards: Optional[str]) -> Optional[List[int]]:
    if shards is None or not str(shards).strip():
        return None
    parsed: List[int] = []
    for part in str(shards).split(","):
        part = part.strip()
        if part:
            parsed.append(int(part))
    return parsed or None


def load_v10_model_from_checkpoint(
    checkpoint: Path,
    device: torch.device,
    backbone_name: str,
    image_size: int,
    frequency_dim: int,
    noise_dim: int,
    fused_dim: int,
    head_hidden_dim: int,
    dropout: float,
    fusion_gate_input_dropout: float,
    fusion_feature_dropout: float,
    alpha_max: float,
    enable_noise_expert: bool,
) -> Tuple[V10CompetitionResetModel, Dict[str, object]]:
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model = V10CompetitionResetModel(
        backbone_name=backbone_name,
        pretrained_backbone=False,
        semantic_trainable_layers=0,
        image_size=image_size,
        frequency_dim=frequency_dim,
        noise_dim=noise_dim,
        fused_dim=fused_dim,
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
        fusion_gate_input_dropout=fusion_gate_input_dropout,
        fusion_feature_dropout=fusion_feature_dropout,
        alpha_max=alpha_max,
        enable_noise_expert=enable_noise_expert,
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
        "phase": str(ckpt.get("phase", "unknown")),
        "loaded_keys": int(len(filtered_state)),
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
    }
    return model, meta


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
            "precision_ge_0_80": None,
            "recall_ge_0_88": None,
        }
    best_f1_row = sweep_df.sort_values(["f1", "precision", "recall"], ascending=[False, False, False]).iloc[0]
    precision_row = None
    precision_candidates = sweep_df[sweep_df["precision"] >= 0.80]
    if not precision_candidates.empty:
        precision_row = precision_candidates.sort_values(["f1", "recall", "threshold"], ascending=[False, False, True]).iloc[0]
    recall_row = None
    recall_candidates = sweep_df[sweep_df["recall"] >= 0.88]
    if not recall_candidates.empty:
        recall_row = recall_candidates.sort_values(["precision", "f1", "threshold"], ascending=[False, False, True]).iloc[0]
    return {
        "best_f1": best_f1_row.to_dict(),
        "precision_ge_0_80": None if precision_row is None else precision_row.to_dict(),
        "recall_ge_0_88": None if recall_row is None else recall_row.to_dict(),
    }


def summarize_fixed_thresholds(
    records: List[Dict[str, object]],
    thresholds: Sequence[float] = (0.20, 0.25, 0.30, 0.35, 0.55),
) -> Dict[str, object]:
    return {
        f"{float(threshold):.2f}": records_to_metrics(records=records, threshold=float(threshold))
        for threshold in thresholds
    }


def records_to_metrics(records: List[Dict[str, object]], threshold: float) -> Dict[str, object]:
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


def _extract_sf_triplet(out: Dict[str, torch.Tensor]) -> Tuple[float, float, float]:
    fusion = out.get("fusion_weights")
    if fusion is None:
        return 0.0, 0.0, 0.0
    row = fusion.detach().cpu().view(-1, fusion.shape[-1])[0].tolist()
    if len(row) == 2:
        return float(row[0]), float(row[1]), 0.0
    if len(row) >= 3:
        return float(row[0]), float(row[1]), float(row[2])
    return 0.0, 0.0, 0.0


def collect_records_from_subset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
) -> List[Dict[str, object]]:
    set_model_mode(model, mode)
    model.eval()
    records: List[Dict[str, object]] = []
    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device, non_blocking=True)
            out = model(images)
            final_logits = out["logit"].detach().view(-1).cpu().tolist()
            base_logits = out["base_logit"].detach().view(-1).cpu().tolist()
            hybrid_logits = out.get("hybrid_logit", out["logit"]).detach().view(-1).cpu().tolist()
            semantic_logits = out["semantic_logit"].detach().view(-1).cpu().tolist()
            frequency_logits = out["freq_logit"].detach().view(-1).cpu().tolist()
            alpha_values = out.get("alpha_used", out.get("alpha", torch.zeros_like(out["logit"]))).detach().view(-1).cpu().tolist()
            noise_delta = out.get("noise_delta_logit", torch.zeros_like(out["logit"])).detach().view(-1).cpu().tolist()
            sf_weights = out.get("sf_weights")
            fusion_weights = out.get("fusion_weights")
            if sf_weights is not None:
                sf_rows = sf_weights.detach().cpu().tolist()
            else:
                sf_rows = [[None, None] for _ in final_logits]
            if fusion_weights is not None:
                fusion_rows = fusion_weights.detach().cpu().tolist()
            else:
                fusion_rows = [[None, None, None] for _ in final_logits]
            image_names = list(metadata.get("image_name", []))
            image_paths = list(metadata.get("image_path", []))
            label_values = labels.detach().view(-1).cpu().tolist()
            for idx, image_name in enumerate(image_names):
                records.append(
                    {
                        "image_name": str(image_name),
                        "image_path": str(image_paths[idx]),
                        "label": int(label_values[idx]),
                        "probability": float(torch.sigmoid(torch.tensor(final_logits[idx])).item()),
                        "logit": float(final_logits[idx]),
                        "base_probability": float(torch.sigmoid(torch.tensor(base_logits[idx])).item()),
                        "base_logit": float(base_logits[idx]),
                        "hybrid_probability": float(torch.sigmoid(torch.tensor(hybrid_logits[idx])).item()),
                        "hybrid_logit": float(hybrid_logits[idx]),
                        "semantic_logit": float(semantic_logits[idx]),
                        "frequency_logit": float(frequency_logits[idx]),
                        "alpha": float(alpha_values[idx]),
                        "noise_delta_logit": float(noise_delta[idx]),
                        "sf_semantic": None if sf_rows[idx][0] is None else float(sf_rows[idx][0]),
                        "sf_frequency": None if sf_rows[idx][1] is None else float(sf_rows[idx][1]),
                        "fusion_semantic": None if fusion_rows[idx][0] is None else float(fusion_rows[idx][0]),
                        "fusion_frequency": None if fusion_rows[idx][1] is None else float(fusion_rows[idx][1]),
                        "fusion_noise": None if len(fusion_rows[idx]) < 3 or fusion_rows[idx][2] is None else float(fusion_rows[idx][2]),
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
) -> List[Dict[str, object]]:
    labels_df = pd.read_csv(labels_csv)
    labels_map = {str(row["image_name"]): int(row["label"]) for _, row in labels_df.iterrows()}
    transform = build_eval_transform(image_size=image_size)
    set_model_mode(model, mode)
    model.eval()
    image_paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        image_paths.extend(sorted(folder.glob(pattern)))
    records: List[Dict[str, object]] = []
    with torch.no_grad():
        for image_path in image_paths:
            if image_path.name not in labels_map:
                continue
            arr = np.array(Image.open(image_path).convert("RGB"))
            tensor = transform(image=arr)["image"].unsqueeze(0).to(device)
            out = model(tensor)
            sf_semantic, sf_frequency, fusion_noise = _extract_sf_triplet(out)
            records.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "label": int(labels_map[image_path.name]),
                    "probability": float(torch.sigmoid(out["logit"].view(-1)).item()),
                    "logit": float(out["logit"].view(-1).item()),
                    "base_probability": float(torch.sigmoid(out["base_logit"].view(-1)).item()),
                    "base_logit": float(out["base_logit"].view(-1).item()),
                    "hybrid_probability": float(torch.sigmoid(out.get("hybrid_logit", out["logit"]).view(-1)).item()),
                    "hybrid_logit": float(out.get("hybrid_logit", out["logit"]).view(-1).item()),
                    "semantic_logit": float(out["semantic_logit"].view(-1).item()),
                    "frequency_logit": float(out["freq_logit"].view(-1).item()),
                    "alpha": float(out.get("alpha_used", out.get("alpha", torch.zeros_like(out["logit"]))).view(-1).item()),
                    "noise_delta_logit": float(out.get("noise_delta_logit", torch.zeros_like(out["logit"])).view(-1).item()),
                    "sf_semantic": sf_semantic,
                    "sf_frequency": sf_frequency,
                    "fusion_semantic": sf_semantic,
                    "fusion_frequency": sf_frequency,
                    "fusion_noise": fusion_noise,
                    "mode": mode,
                }
            )
    return records


def evaluate_records(
    records: List[Dict[str, object]],
    default_threshold: float,
) -> Dict[str, object]:
    sweep = threshold_sweep(
        y_true=[int(record["label"]) for record in records],
        y_prob=[float(record["probability"]) for record in records],
    )
    return {
        "default": records_to_metrics(records=records, threshold=default_threshold),
        "thresholds": summarize_thresholds(sweep),
        "fixed_thresholds": summarize_fixed_thresholds(records=records),
        "sweep": sweep,
    }


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
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    val_records = collect_records_from_subset(model=model, loader=val_loader, device=device, mode=mode)
    photos_records = collect_records_from_folder(
        model=model,
        folder=photos_dir,
        labels_csv=photos_labels,
        device=device,
        mode=mode,
        image_size=image_size,
    )
    val_predictions = pd.DataFrame(val_records)
    photos_predictions = pd.DataFrame(photos_records)
    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    photos_predictions.to_csv(output_dir / "photos_predictions.csv", index=False)
    val_eval = evaluate_records(records=val_records, default_threshold=default_threshold)
    photos_eval = evaluate_records(records=photos_records, default_threshold=default_threshold)
    val_eval["sweep"].to_csv(output_dir / "val_threshold_sweep.csv", index=False)
    photos_eval["sweep"].to_csv(output_dir / "photos_threshold_sweep.csv", index=False)
    report = {
        "mode": mode,
        "val_default": val_eval["default"],
        "photos_default": photos_eval["default"],
        "val_thresholds": val_eval["thresholds"],
        "photos_thresholds": photos_eval["thresholds"],
        "val_fixed_thresholds": val_eval["fixed_thresholds"],
        "photos_fixed_thresholds": photos_eval["fixed_thresholds"],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "report": report,
        "val_records": val_records,
        "photos_records": photos_records,
    }


def _ensemble_records(
    base_records: List[Dict[str, object]],
    hybrid_records: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    hybrid_map = {str(record["image_name"]): record for record in hybrid_records}
    ensemble: List[Dict[str, object]] = []
    for base_record in base_records:
        name = str(base_record["image_name"])
        hybrid_record = hybrid_map.get(name)
        if hybrid_record is None:
            ensemble.append(dict(base_record))
            continue
        base_prob = float(base_record["probability"])
        hybrid_prob = float(hybrid_record["probability"])
        prob = 0.7 * base_prob + 0.3 * hybrid_prob
        prob = min(max(prob, 1e-6), 1.0 - 1e-6)
        logit = math.log(prob / (1.0 - prob))
        row = dict(base_record)
        row["probability"] = float(prob)
        row["logit"] = float(logit)
        row["mode"] = "ensemble"
        ensemble.append(row)
    return ensemble


def _candidate_summary(name: str, evaluation: Dict[str, object]) -> Dict[str, object]:
    report = evaluation["report"]
    return {
        "name": name,
        "val_default": report["val_default"],
        "photos_default": report["photos_default"],
        "val_thresholds": report["val_thresholds"],
        "photos_thresholds": report["photos_thresholds"],
        "val_fixed_thresholds": report.get("val_fixed_thresholds", {}),
        "photos_fixed_thresholds": report.get("photos_fixed_thresholds", {}),
    }


def choose_industrial_winner(
    candidates: List[Dict[str, object]],
) -> Dict[str, object]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["photos_default"].get("precision", 0.0)),
            -float(item["photos_default"].get("recall", 0.0)),
            -float(item["photos_default"].get("f1", 0.0)),
        ),
    )
    return ranked[0]


def choose_competition_winner(
    candidates: List[Dict[str, object]],
) -> Dict[str, object]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float((item.get("photos_thresholds", {}).get("best_f1") or {}).get("f1", item["photos_default"].get("f1", 0.0))),
            -float(item["photos_default"].get("auroc", 0.0)),
            -float(item["photos_default"].get("precision", 0.0)),
        ),
    )
    return ranked[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate V10 industrial and competition candidates.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--photos-dir", type=str, default=str(PROJECT_ROOT / "photos_test"))
    parser.add_argument("--photos-labels", type=str, default=str(PROJECT_ROOT / "photos_test" / "labels.csv"))
    parser.add_argument("--shards", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--default-threshold", type=float, default=0.20)
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--frequency-dim", type=int, default=256)
    parser.add_argument("--noise-dim", type=int, default=256)
    parser.add_argument("--fused-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fusion-gate-input-dropout", type=float, default=0.1)
    parser.add_argument("--fusion-feature-dropout", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=0.35)
    parser.add_argument("--disable-hybrid", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_meta = load_v10_model_from_checkpoint(
        checkpoint=Path(args.checkpoint),
        device=device,
        backbone_name=args.backbone_name,
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_tri = evaluate_candidate(
        model=model,
        device=device,
        mode="tri_fusion",
        val_loader=val_loader,
        photos_dir=Path(args.photos_dir),
        photos_labels=Path(args.photos_labels),
        output_dir=output_dir / "candidate_tri_fusion",
        image_size=args.image_size,
        default_threshold=args.default_threshold,
    )

    candidate_summaries = [
        _candidate_summary("candidate_tri_fusion", candidate_tri),
    ]
    final_report: Dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_meta": checkpoint_meta,
        "val_mode": val_mode,
        "candidate_tri_fusion": candidate_summaries[0],
        "candidate_hybrid": None,
        "candidate_ensemble": None,
    }

    final_report["winner_industrial"] = choose_industrial_winner(candidate_summaries)
    final_report["winner_competition"] = choose_competition_winner(candidate_summaries)
    (output_dir / "candidate_comparison.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    candidate_summary = {
        "checkpoint": final_report["checkpoint"],
        "checkpoint_meta": final_report["checkpoint_meta"],
        "val_mode": final_report["val_mode"],
        "winner_industrial": final_report["winner_industrial"],
        "winner_competition": final_report["winner_competition"],
    }
    (output_dir / "candidate_summary.json").write_text(json.dumps(candidate_summary, indent=2), encoding="utf-8")
    print(json.dumps(final_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
