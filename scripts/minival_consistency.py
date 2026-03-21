#!/usr/bin/env python3
"""
最小验证集一致性检查脚本 - 升级版
输出: Precision/Recall/F1、漏检样本列表、阈值扫描
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire.augmentations import build_eval_transform, build_train_transform
from ai_image_detector.ntire.model import HybridAIGCDetector


def load_model(
    checkpoint: Path,
    device: torch.device,
    v8_stage: str = "residual_finetune",
) -> Tuple[torch.nn.Module, float]:
    """Load model and return with temperature."""
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt["model"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    is_v8 = any(
        key.startswith("primary_fusion_sf.")
        or key.startswith("noise_controller.")
        or key.startswith("noise_delta_head.")
        for key in state_dict.keys()
    )
    model = HybridAIGCDetector(
        backbone_name="vit_base_patch16_clip_224.openai",
        pretrained_backbone=False,
        enable_base_residual_fusion=is_v8,
        v8_stage=v8_stage,
    )
    model_state = model.state_dict()
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(filtered_state, strict=False)
    model.to(device).eval()
    temperature = float(ckpt.get("temperature", 1.0))
    return model, temperature


def predict_image(model: torch.nn.Module, image: Image.Image, transform, device: torch.device, temperature: float) -> Dict:
    """Predict single image and return detailed results."""
    arr = np.array(image.convert("RGB"))
    x = transform(image=arr)["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(x)
        logit = out["logit"].item()
        calibrated_logit = logit / max(temperature, 1e-6)
        prob = torch.sigmoid(torch.tensor(calibrated_logit)).item()
        fusion_weights = out["fusion_weights"].cpu().numpy()[0]
        base_logit = float(out["base_logit"].item()) if "base_logit" in out else logit
        base_prob = torch.sigmoid(torch.tensor(base_logit / max(temperature, 1e-6))).item()
        noise_delta_logit = float(out["noise_delta_logit"].item()) if "noise_delta_logit" in out else 0.0
        alpha = float(out["alpha_used"].item()) if "alpha_used" in out else (float(out["alpha"].item()) if "alpha" in out else 0.0)
        semantic_logit = float(out["semantic_logit"].item()) if "semantic_logit" in out else float("nan")
        frequency_logit = float(out["freq_logit"].item()) if "freq_logit" in out else float("nan")
        noise_logit = float(out["noise_logit"].item()) if "noise_logit" in out else float("nan")
    
    return {
        "logit": logit,
        "calibrated_logit": calibrated_logit,
        "probability": prob,
        "base_logit": base_logit,
        "base_probability": base_prob,
        "noise_delta_logit": noise_delta_logit,
        "alpha": alpha,
        "semantic_logit": semantic_logit,
        "frequency_logit": frequency_logit,
        "noise_logit": noise_logit,
        "fusion_semantic": fusion_weights[0],
        "fusion_frequency": fusion_weights[1],
        "fusion_noise": fusion_weights[2],
    }


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict:
    """Compute classification metrics."""
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    # BCE Loss
    bce_loss = 0.0
    for yt, yp in zip(y_true, y_prob):
        yp_clipped = max(min(yp, 1 - 1e-7), 1e-7)
        bce_loss += -(yt * np.log(yp_clipped) + (1 - yt) * np.log(1 - yp_clipped))
    bce_loss /= len(y_true) if len(y_true) > 0 else 1
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "bce_loss": bce_loss,
    }


def threshold_sweep(y_true: List[int], y_prob: List[float], thresholds: np.ndarray) -> pd.DataFrame:
    """Sweep thresholds and compute metrics."""
    results = []
    for th in thresholds:
        y_pred = [1 if p >= th else 0 for p in y_prob]
        metrics = compute_metrics(y_true, y_pred, y_prob)
        results.append({"threshold": th, **metrics})
    return pd.DataFrame(results)


def compute_consistency_loss(model: torch.nn.Module, image: Image.Image, transform1, transform2, device: torch.device, temperature: float) -> float:
    """Compute consistency loss between two augmented views."""
    arr = np.array(image.convert("RGB"))
    
    x1 = transform1(image=arr)["image"].unsqueeze(0).to(device)
    x2 = transform2(image=arr)["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
        
        logit1 = out1["logit"].item() / max(temperature, 1e-6)
        logit2 = out2["logit"].item() / max(temperature, 1e-6)
        
        prob1 = torch.sigmoid(torch.tensor(logit1)).item()
        prob2 = torch.sigmoid(torch.tensor(logit2)).item()
        
        # MSE consistency loss
        consistency_loss = (prob1 - prob2) ** 2
    
    return consistency_loss


def per_sample_bce(label: int, prob: float) -> float:
    yp = max(min(float(prob), 1 - 1e-7), 1e-7)
    yt = float(label)
    return float(-(yt * np.log(yp) + (1 - yt) * np.log(1 - yp)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--folder", type=str, default="photos_test")
    parser.add_argument("--labels", type=str, default="photos_test/labels.csv")
    parser.add_argument("--profile", type=str, default="balanced", choices=["f1", "balanced", "recall"])
    parser.add_argument("--output-dir", type=str, default="outputs/minival_v4")
    parser.add_argument("--default-threshold", type=float, default=0.2)
    parser.add_argument("--compute-consistency", action="store_true", help="Compute dual-view consistency")
    parser.add_argument(
        "--v8-stage",
        type=str,
        default="residual_finetune",
        choices=["debias_base", "residual_finetune"],
        help="Inference stage for V8 residual fusion checkpoints.",
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, temperature = load_model(Path(args.checkpoint), device, v8_stage=args.v8_stage)
    print(f"Model loaded from: {args.checkpoint}")
    print(f"Temperature: {temperature:.4f}")
    
    # Load labels
    labels_df = pd.read_csv(args.labels)
    labels_map = dict(zip(labels_df["image_name"], labels_df["label"]))
    
    # Get image paths
    folder = Path(args.folder)
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths.extend(folder.glob(ext))
    
    # Transforms
    eval_transform = build_eval_transform(224)
    train_transform1 = build_train_transform(224, chain_mix=True, chain_mix_strength="aigc_focus")
    train_transform2 = build_train_transform(224, chain_mix=True, chain_mix_strength="aigc_focus")
    
    # Inference
    results = []
    y_true = []
    y_prob = []
    
    print(f"\nProcessing {len(image_paths)} images...")
    for img_path in sorted(image_paths):
        img_name = img_path.name
        if img_name not in labels_map:
            continue
        
        label = labels_map[img_name]
        image = Image.open(img_path)
        
        # Eval transform prediction
        pred = predict_image(model, image, eval_transform, device, temperature)
        
        # Consistency loss
        consistency_loss = 0.0
        if args.compute_consistency:
            consistency_loss = compute_consistency_loss(model, image, train_transform1, train_transform2, device, temperature)
        
        bce_i = per_sample_bce(label, pred["probability"])
        results.append({
            "image_name": img_name,
            "label": label,
            "probability": pred["probability"],
            "logit": pred["logit"],
            "calibrated_logit": pred["calibrated_logit"],
            "base_probability": pred["base_probability"],
            "base_logit": pred["base_logit"],
            "noise_delta_logit": pred["noise_delta_logit"],
            "alpha": pred["alpha"],
            "semantic_logit": pred["semantic_logit"],
            "frequency_logit": pred["frequency_logit"],
            "noise_logit": pred["noise_logit"],
            "bce_loss": bce_i,
            "prediction_at_default": int(pred["probability"] >= args.default_threshold),
            "fusion_semantic": pred["fusion_semantic"],
            "fusion_frequency": pred["fusion_frequency"],
            "fusion_noise": pred["fusion_noise"],
            "consistency_loss": consistency_loss,
        })
        
        y_true.append(label)
        y_prob.append(pred["probability"])
    
    # Compute metrics with default threshold (0.5)
    y_pred = [1 if p >= args.default_threshold else 0 for p in y_prob]
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    # Threshold sweep
    thresholds = np.arange(0.1, 0.6001, 0.05)
    sweep_df = threshold_sweep(y_true, y_prob, thresholds)
    
    # Find optimal thresholds
    best_f1_idx = sweep_df["f1"].idxmax()
    best_recall_idx = sweep_df["recall"].idxmax()
    best_precision_idx = sweep_df["precision"].idxmax()
    
    # Find missed AIGC samples (false negatives)
    missed_aigc = [r for r in results if r["label"] == 1 and r["probability"] < args.default_threshold]
    false_positive = [r for r in results if r["label"] == 0 and r["probability"] >= args.default_threshold]
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "predictions.csv", index=False)
    results_df[["image_name", "fusion_semantic", "fusion_frequency", "fusion_noise"]].to_csv(
        output_dir / "fusion_weights.csv", index=False
    )
    
    # Save sweep results
    sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False)
    
    threshold_020 = sweep_df[np.isclose(sweep_df["threshold"], 0.2)].head(1)
    recall_at_020 = float(threshold_020["recall"].iloc[0]) if not threshold_020.empty else None
    precision_at_020 = float(threshold_020["precision"].iloc[0]) if not threshold_020.empty else None
    f1_at_020 = float(threshold_020["f1"].iloc[0]) if not threshold_020.empty else None
    semantic_ok = bool((results_df["fusion_semantic"] <= 0.45 + 1e-6).all()) if len(results_df) else False
    noise_ok = bool((results_df["fusion_noise"] >= 0.20 - 1e-6).all()) if len(results_df) else False
    sum_ok = bool(
        np.allclose(
            results_df["fusion_semantic"] + results_df["fusion_frequency"] + results_df["fusion_noise"],
            1.0,
            atol=1e-5,
        )
    ) if len(results_df) else False
    numeric_stable = bool(np.isfinite(results_df[["probability", "bce_loss", "consistency_loss"]].to_numpy()).all()) if len(results_df) else False

    # Save metrics JSON
    report = {
        "metrics": metrics,
        "default_threshold": float(args.default_threshold),
        "threshold_0_20_metrics": {
            "precision": precision_at_020,
            "recall": recall_at_020,
            "f1": f1_at_020,
        },
        "optimal_thresholds": {
            "best_f1": {
                "threshold": float(sweep_df.loc[best_f1_idx, "threshold"]),
                "f1": float(sweep_df.loc[best_f1_idx, "f1"]),
                "precision": float(sweep_df.loc[best_f1_idx, "precision"]),
                "recall": float(sweep_df.loc[best_f1_idx, "recall"]),
            },
            "best_recall": {
                "threshold": float(sweep_df.loc[best_recall_idx, "threshold"]),
                "recall": float(sweep_df.loc[best_recall_idx, "recall"]),
                "precision": float(sweep_df.loc[best_recall_idx, "precision"]),
                "f1": float(sweep_df.loc[best_recall_idx, "f1"]),
            },
            "best_precision": {
                "threshold": float(sweep_df.loc[best_precision_idx, "threshold"]),
                "precision": float(sweep_df.loc[best_precision_idx, "precision"]),
                "recall": float(sweep_df.loc[best_precision_idx, "recall"]),
                "f1": float(sweep_df.loc[best_precision_idx, "f1"]),
            },
        },
        "missed_aigc": missed_aigc,
        "false_positive": false_positive,
        "fusion_constraints": {
            "semantic_le_0_45": semantic_ok,
            "noise_ge_0_20": noise_ok,
            "weights_sum_to_1": sum_ok,
        },
        "alpha_distribution": {
            "mean": float(results_df["alpha"].mean()) if len(results_df) else None,
            "real_mean": float(results_df.loc[results_df["label"] == 0, "alpha"].mean()) if len(results_df) else None,
            "aigc_mean": float(results_df.loc[results_df["label"] == 1, "alpha"].mean()) if len(results_df) else None,
            "max": float(results_df["alpha"].max()) if len(results_df) else None,
        },
        "noise_delta_distribution": {
            "mean": float(results_df["noise_delta_logit"].mean()) if len(results_df) else None,
            "real_mean": float(results_df.loc[results_df["label"] == 0, "noise_delta_logit"].mean()) if len(results_df) else None,
            "aigc_mean": float(results_df.loc[results_df["label"] == 1, "noise_delta_logit"].mean()) if len(results_df) else None,
        },
        "numerical_stability": {
            "finite_prob_bce_consistency": numeric_stable,
        },
        "summary": {
            "total_samples": len(results),
            "aigc_samples": sum(r["label"] == 1 for r in results),
            "real_samples": sum(r["label"] == 0 for r in results),
            "missed_aigc_count": len(missed_aigc),
            "false_positive_count": len(false_positive),
            "mean_bce_loss": float(results_df["bce_loss"].mean()) if len(results_df) else None,
            "mean_consistency_loss": float(results_df["consistency_loss"].mean()) if len(results_df) else None,
            "mean_base_probability": float(results_df["base_probability"].mean()) if len(results_df) else None,
            "mean_alpha": float(results_df["alpha"].mean()) if len(results_df) else None,
        }
    }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    report_native = convert_to_native(report)
    
    with open(output_dir / "report.json", "w") as f:
        json.dump(report_native, f, indent=2)
    
    # Print report
    print("\n" + "="*60)
    print("最小验证集一致性检查报告")
    print("="*60)
    
    print(f"\n样本统计:")
    print(f"  总样本: {len(results)}")
    print(f"  AIGC (label=1): {sum(r['label'] == 1 for r in results)}")
    print(f"  真实 (label=0): {sum(r['label'] == 0 for r in results)}")
    
    print(f"\n默认阈值 ({args.default_threshold:.2f}) 指标:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  BCE Loss:  {metrics['bce_loss']:.4f}")
    
    print(f"\n最优阈值:")
    print(f"  Best F1:     threshold={sweep_df.loc[best_f1_idx, 'threshold']:.2f}, "
          f"F1={sweep_df.loc[best_f1_idx, 'f1']:.4f}")
    print(f"  Best Recall: threshold={sweep_df.loc[best_recall_idx, 'threshold']:.2f}, "
          f"Recall={sweep_df.loc[best_recall_idx, 'recall']:.4f}")
    print(f"  Best Prec:   threshold={sweep_df.loc[best_precision_idx, 'threshold']:.2f}, "
          f"Precision={sweep_df.loc[best_precision_idx, 'precision']:.4f}")
    
    if missed_aigc:
        print(f"\n漏检的 AIGC 样本 ({len(missed_aigc)} 个):")
        for r in missed_aigc:
            print(f"  {r['image_name']}: prob={r['probability']:.4f}, "
                  f"fusion=({r['fusion_semantic']:.3f}, {r['fusion_frequency']:.3f}, {r['fusion_noise']:.3f})")
    
    if false_positive:
        print(f"\n误报的真实样本 ({len(false_positive)} 个):")
        for r in false_positive:
            print(f"  {r['image_name']}: prob={r['probability']:.4f}")
    
    print(f"\n融合权重统计:")
    print(f"  Semantic: mean={results_df['fusion_semantic'].mean():.4f}, std={results_df['fusion_semantic'].std():.4f}")
    print(f"  Frequency: mean={results_df['fusion_frequency'].mean():.4f}, std={results_df['fusion_frequency'].std():.4f}")
    print(f"  Noise: mean={results_df['fusion_noise'].mean():.4f}, std={results_df['fusion_noise'].std():.4f}")
    
    if recall_at_020 is not None:
        print(f"\n阈值 0.20 指标: Precision={precision_at_020:.4f}, Recall={recall_at_020:.4f}, F1={f1_at_020:.4f}")
    print(f"\n输出文件:")
    print(f"  {output_dir / 'predictions.csv'}")
    print(f"  {output_dir / 'fusion_weights.csv'}")
    print(f"  {output_dir / 'threshold_sweep.csv'}")
    print(f"  {output_dir / 'report.json'}")
    print("="*60)


if __name__ == "__main__":
    main()
