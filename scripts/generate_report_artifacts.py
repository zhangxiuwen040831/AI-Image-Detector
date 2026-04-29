from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from openpyxl import load_workbook
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEPLOYMENT_THRESHOLDS = {
    "recall-first": 0.20,
    "balanced": 0.35,
    "precision-first": 0.35,
}
ANALYSIS_THRESHOLDS = {
    "precision-first-analysis": 0.55,
}
PHOTOS_TEST_SCOPE = (
    "photos_test is a small-sample diagnostic set for boundary inspection; "
    "it is not a large-scale generalization benchmark."
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compute_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float) -> Dict[str, Any]:
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_prob_arr = np.asarray(list(y_prob), dtype=np.float64)
    y_pred = (y_prob_arr >= threshold).astype(np.int64)

    tp = int(np.sum((y_true_arr == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = float((tp + tn) / len(y_true_arr)) if len(y_true_arr) else 0.0

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _load_checkpoint_history(checkpoint_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = checkpoint.get("history", [])
    history_rows: List[Dict[str, Any]] = []
    milestone_rows: List[Dict[str, Any]] = []

    for item in history:
        row: Dict[str, Any] = {
            "epoch": item.get("epoch"),
            "phase": item.get("phase"),
        }
        for split in ("train", "val"):
            metrics = item.get(split, {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[f"{split}_{key}"] = value
        history_rows.append(row)

        evaluation = item.get("evaluation") or {}
        candidate = evaluation.get("candidate_base_only") if isinstance(evaluation, dict) else None
        photos_default = candidate.get("photos_default") if isinstance(candidate, dict) else None
        if photos_default:
            milestone_rows.append(
                {
                    "epoch": item.get("epoch"),
                    "phase": item.get("phase"),
                    "precision": photos_default.get("precision"),
                    "recall": photos_default.get("recall"),
                    "f1": photos_default.get("f1"),
                    "accuracy": photos_default.get("accuracy"),
                    "tp": photos_default.get("tp"),
                    "fp": photos_default.get("fp"),
                    "tn": photos_default.get("tn"),
                    "fn": photos_default.get("fn"),
                    "source": f"checkpoint_history_epoch_{item.get('epoch')}",
                }
            )

    return pd.DataFrame(history_rows), pd.DataFrame(milestone_rows)


def _prepare_photos_predictions(raw_predictions_csv: Path, labels_csv: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_predictions_csv)
    labels_df = pd.read_csv(labels_csv)
    labels_df = labels_df.rename(columns={"label": "ground_truth"})
    raw_df["image_name"] = raw_df["path"].map(lambda value: Path(str(value)).name)
    merged = raw_df.merge(labels_df, on="image_name", how="left")
    merged["is_labeled_diagnostic_sample"] = merged["ground_truth"].notna()
    merged = merged[merged["is_labeled_diagnostic_sample"]].copy()
    merged["ground_truth"] = merged["ground_truth"].astype(int)
    merged["label_name"] = merged["ground_truth"].map({0: "REAL", 1: "AIGC"})
    for threshold_name, threshold in {
        "pred_recall_first": DEPLOYMENT_THRESHOLDS["recall-first"],
        "pred_balanced": DEPLOYMENT_THRESHOLDS["balanced"],
        "pred_precision_first": DEPLOYMENT_THRESHOLDS["precision-first"],
        "pred_precision_first_analysis": ANALYSIS_THRESHOLDS["precision-first-analysis"],
    }.items():
        merged[threshold_name] = np.where(merged["probability"] >= threshold, "AIGC", "REAL")
    merged["evaluation_scope"] = PHOTOS_TEST_SCOPE
    return merged


def _ensure_raw_predictions(
    project_root: Path,
    checkpoint_path: Path,
    photos_dir: Path,
    raw_predictions_csv: Path,
) -> None:
    if raw_predictions_csv.exists():
        return
    raw_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(project_root / "scripts" / "infer_ntire.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--folder",
        str(photos_dir),
        "--out-csv",
        str(raw_predictions_csv),
        "--device",
        "cpu",
        "--threshold-profile",
        "balanced",
    ]
    subprocess.run(command, check=True, cwd=project_root)


def _build_threshold_sweep(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for threshold in np.arange(0.10, 0.601, 0.01):
        metrics = _compute_metrics(
            predictions_df["ground_truth"],
            predictions_df["probability"],
            float(round(threshold, 2)),
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _build_version_comparison(
    milestone_df: pd.DataFrame,
    v9_report_json: Path,
    final_predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    v8_row = milestone_df[milestone_df["epoch"] == 2].head(1).to_dict("records")
    if not v8_row:
        raise RuntimeError("Missing historical V8 baseline (epoch 2) in checkpoint history.")

    v9_report = json.loads(v9_report_json.read_text(encoding="utf-8"))
    v9_default = v9_report["photos_default"]
    v10_default = _compute_metrics(final_predictions_df["ground_truth"], final_predictions_df["probability"], 0.20)

    rows = [
        {
            "version": "V8",
            "fp": int(v8_row[0]["fp"]),
            "fn": int(v8_row[0]["fn"]),
            "precision": float(v8_row[0]["precision"]),
            "recall": float(v8_row[0]["recall"]),
            "f1": float(v8_row[0]["f1"]),
            "source": "Internal report artifact from final checkpoint history epoch 2 (threshold=0.20)",
            "evidence_level": "checkpoint-history-derived",
            "reproducibility": "partial: no standalone V8 checkpoint is present in the current workspace",
        },
        {
            "version": "V9",
            "fp": int(v9_default["fp"]),
            "fn": int(v9_default["fn"]),
            "precision": float(v9_default["precision"]),
            "recall": float(v9_default["recall"]),
            "f1": float(v9_default["f1"]),
            "source": "docs/reference/v9_reference_report.json (threshold=0.20)",
            "evidence_level": "reference-json",
            "reproducibility": "partial: no standalone V9 checkpoint is present in the current workspace",
        },
        {
            "version": "V10",
            "fp": int(v10_default["fp"]),
            "fn": int(v10_default["fn"]),
            "precision": float(v10_default["precision"]),
            "recall": float(v10_default["recall"]),
            "f1": float(v10_default["f1"]),
            "source": "Current checkpoints/best.pth inference on photos_test (threshold=0.20)",
            "evidence_level": "current-checkpoint-inference",
            "reproducibility": "yes: reproducible with checkpoints/best.pth and labeled photos_test samples",
        },
    ]
    return pd.DataFrame(rows)


def _build_threshold_profiles(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile, threshold, scope, note in [
        ("recall-first", DEPLOYMENT_THRESHOLDS["recall-first"], "deployment", "runtime API profile"),
        ("balanced", DEPLOYMENT_THRESHOLDS["balanced"], "deployment", "current default deployment profile"),
        ("precision-first", DEPLOYMENT_THRESHOLDS["precision-first"], "deployment", "runtime API profile; aligned with balanced in the current UI"),
        (
            "precision-first-analysis",
            ANALYSIS_THRESHOLDS["precision-first-analysis"],
            "analysis",
            "photos_test diagnostic sweep only; not a deployment default",
        ),
    ]:
        metrics = _compute_metrics(predictions_df["ground_truth"], predictions_df["probability"], threshold)
        metrics["profile"] = profile
        metrics["scope"] = scope
        metrics["note"] = note
        rows.append(metrics)
    return pd.DataFrame(rows)[
        ["profile", "scope", "threshold", "precision", "recall", "f1", "accuracy", "tp", "fp", "tn", "fn", "note"]
    ]


def _plot_training_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Loss")
    axes[0, 0].plot(history_df["epoch"], history_df["val_loss"], marker="s", label="Val Loss")
    axes[0, 0].set_title("Training And Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(history_df["epoch"], history_df["val_base_f1"], marker="o", color="#1f77b4")
    axes[0, 1].set_title("Validation Base F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("F1")

    axes[1, 0].plot(history_df["epoch"], history_df["val_base_precision"], marker="o", color="#2ca02c")
    axes[1, 0].set_title("Validation Base Precision")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Precision")

    axes[1, 1].plot(history_df["epoch"], history_df["val_base_recall"], marker="o", color="#d62728")
    axes[1, 1].set_title("Validation Base Recall")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_sweep(sweep_df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", linewidth=2)
    ax1.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", linewidth=2)
    ax1.plot(sweep_df["threshold"], sweep_df["f1"], label="F1", linewidth=2)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Threshold Sweep On photos_test")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(sweep_df["threshold"], sweep_df["fp"], label="False Positives", color="#ff7f0e", linestyle="--")
    ax2.set_ylabel("FP Count")
    ax2.legend(loc="upper right")

    for threshold, label in [
        (DEPLOYMENT_THRESHOLDS["recall-first"], "deploy recall-first"),
        (DEPLOYMENT_THRESHOLDS["balanced"], "deploy balanced / precision-first"),
        (ANALYSIS_THRESHOLDS["precision-first-analysis"], "analysis only"),
    ]:
        ax1.axvline(threshold, color="#999999", linestyle=":", linewidth=1)
        ax1.text(threshold, 0.04, label, rotation=90, va="bottom", ha="right", fontsize=8, color="#555555")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(predictions_df: pd.DataFrame, threshold: float, output_path: Path) -> None:
    y_true = predictions_df["ground_truth"].to_numpy(dtype=np.int64)
    y_pred = (predictions_df["probability"].to_numpy(dtype=np.float64) >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred REAL", "Pred AIGC"],
        yticklabels=["True REAL", "True AIGC"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_precision_recall_curve(predictions_df: pd.DataFrame, output_path: Path) -> None:
    y_true = predictions_df["ground_truth"].to_numpy(dtype=np.int64)
    y_prob = predictions_df["probability"].to_numpy(dtype=np.float64)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(recall, precision, linewidth=2.2, color="#1f77b4", label=f"PR Curve (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve On photos_test")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_workbook(
    output_path: Path,
    predictions_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    version_df: pd.DataFrame,
    threshold_profiles_df: pd.DataFrame,
    history_df: pd.DataFrame,
    milestone_df: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        predictions_df.to_excel(writer, sheet_name="photos_predictions", index=False)
        sweep_df.to_excel(writer, sheet_name="threshold_sweep", index=False)
        version_df.to_excel(writer, sheet_name="version_comparison", index=False)
        threshold_profiles_df.to_excel(writer, sheet_name="threshold_profiles", index=False)
        history_df.to_excel(writer, sheet_name="checkpoint_history", index=False)
        milestone_df.to_excel(writer, sheet_name="milestones", index=False)

    workbook = load_workbook(output_path)
    for sheet in workbook.worksheets:
        for column_cells in sheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter
            for cell in column_cells:
                value = "" if cell.value is None else str(cell.value)
                max_length = max(max_length, len(value))
            sheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 42)
    workbook.save(output_path)


def generate_report_artifacts(project_root: Path | None = None) -> Dict[str, Any]:
    root = project_root or PROJECT_ROOT
    figures_dir = _ensure_dir(root / "figures")
    tmp_dir = _ensure_dir(root / "tmp" / "report_generation")
    spreadsheet_dir = _ensure_dir(root / "output" / "spreadsheet")

    checkpoint_path = root / "checkpoints" / "best.pth"
    raw_predictions_csv = tmp_dir / "photos_predictions_raw.csv"
    photos_dir = root / "photos_test"
    labels_csv = root / "photos_test" / "labels.csv"
    v9_report_json = root / "docs" / "reference" / "v9_reference_report.json"

    history_df, milestone_df = _load_checkpoint_history(checkpoint_path)
    _ensure_raw_predictions(
        project_root=root,
        checkpoint_path=checkpoint_path,
        photos_dir=photos_dir,
        raw_predictions_csv=raw_predictions_csv,
    )
    predictions_df = _prepare_photos_predictions(raw_predictions_csv, labels_csv)
    sweep_df = _build_threshold_sweep(predictions_df)
    version_df = _build_version_comparison(milestone_df, v9_report_json, predictions_df)
    threshold_profiles_df = _build_threshold_profiles(predictions_df)

    history_csv = tmp_dir / "checkpoint_history_metrics.csv"
    predictions_csv = tmp_dir / "photos_predictions.csv"
    sweep_csv = tmp_dir / "threshold_sweep.csv"
    versions_csv = tmp_dir / "version_comparison.csv"
    threshold_profiles_csv = tmp_dir / "threshold_profiles.csv"
    milestone_csv = tmp_dir / "milestone_metrics.csv"

    history_df.to_csv(history_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)
    sweep_df.to_csv(sweep_csv, index=False)
    version_df.to_csv(versions_csv, index=False)
    threshold_profiles_df.to_csv(threshold_profiles_csv, index=False)
    milestone_df.to_csv(milestone_csv, index=False)

    _plot_training_curves(history_df, figures_dir / "training_curves.png")
    _plot_threshold_sweep(sweep_df, figures_dir / "threshold_sweep.png")
    _plot_confusion_matrix(predictions_df, 0.20, figures_dir / "confusion_matrix_recall_first.png")
    _plot_precision_recall_curve(predictions_df, figures_dir / "precision_recall_curve.png")

    workbook_path = spreadsheet_dir / "report_tables.xlsx"
    _write_workbook(
        workbook_path,
        predictions_df=predictions_df,
        sweep_df=sweep_df,
        version_df=version_df,
        threshold_profiles_df=threshold_profiles_df,
        history_df=history_df,
        milestone_df=milestone_df,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "predictions_csv": str(predictions_csv),
        "threshold_sweep_csv": str(sweep_csv),
        "version_comparison_csv": str(versions_csv),
        "threshold_profiles_csv": str(threshold_profiles_csv),
        "workbook": str(workbook_path),
        "figures": {
            "training_curves": str(figures_dir / "training_curves.png"),
            "threshold_sweep": str(figures_dir / "threshold_sweep.png"),
            "confusion_matrix": str(figures_dir / "confusion_matrix_recall_first.png"),
            "precision_recall_curve": str(figures_dir / "precision_recall_curve.png"),
        },
        "final_metrics_recall_first": threshold_profiles_df.loc[
            threshold_profiles_df["profile"] == "recall-first"
        ].iloc[0].to_dict(),
        "final_metrics_balanced": threshold_profiles_df.loc[
            threshold_profiles_df["profile"] == "balanced"
        ].iloc[0].to_dict(),
        "final_metrics_precision_first_deploy": threshold_profiles_df.loc[
            threshold_profiles_df["profile"] == "precision-first"
        ].iloc[0].to_dict(),
        "final_metrics_precision_first_analysis": threshold_profiles_df.loc[
            threshold_profiles_df["profile"] == "precision-first-analysis"
        ].iloc[0].to_dict(),
        "threshold_definitions": {
            "deployment": DEPLOYMENT_THRESHOLDS,
            "analysis_only": ANALYSIS_THRESHOLDS,
        },
        "photos_test_scope": PHOTOS_TEST_SCOPE,
        "version_evidence_note": (
            "V8/V9/V10 rows are internal version-iteration records. "
            "Only V10 is fully reproducible from the current checkpoint; V8/V9 rely on retained report artifacts."
        ),
        "version_comparison": version_df.to_dict(orient="records"),
    }
    summary_path = tmp_dir / "artifact_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    summary = generate_report_artifacts(PROJECT_ROOT)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
