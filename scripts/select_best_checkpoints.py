from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"


def get_file_md5(file_path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_duplicates(ckpt_paths: List[Path]) -> Dict[str, List[Path]]:
    md5_map: Dict[str, List[Path]] = {}
    for path in ckpt_paths:
        if path.exists():
            md5 = get_file_md5(path)
            if md5 not in md5_map:
                md5_map[md5] = []
            md5_map[md5].append(path)
    return md5_map


def evaluate_single_checkpoint(
    ckpt_path: Path,
    data_root: Path,
    out_csv: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    backbone_name: str,
) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "evaluate_ntire.py"),
        "--data-root",
        str(data_root),
        "--checkpoint",
        str(ckpt_path),
        "--out-csv",
        str(out_csv),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--image-size",
        str(image_size),
        "--backbone-name",
        backbone_name,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            check=True
        )
        if out_csv.exists():
            df = pd.read_csv(out_csv)
            return True, df, None
        else:
            return False, None, "Output CSV not created"
    except subprocess.CalledProcessError as e:
        return False, None, f"Command failed: {e.stderr}"
    except Exception as e:
        return False, None, str(e)


def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    metrics = {}
    
    conditions = ["clean"] + [f"jpeg_q{q}" for q in range(30, 91, 10)]
    
    for condition in conditions:
        row = df[df["condition"] == condition]
        if not row.empty:
            metrics[f"{condition}_auroc"] = float(row["auroc"].iloc[0])
            metrics[f"{condition}_f1"] = float(row["f1"].iloc[0])
            metrics[f"{condition}_ece"] = float(row["ece"].iloc[0])
    
    aurocs = [metrics[f"{c}_auroc"] for c in conditions if f"{c}_auroc" in metrics]
    f1s = [metrics[f"{c}_f1"] for c in conditions if f"{c}_f1" in metrics]
    eces = [metrics[f"{c}_ece"] for c in conditions if f"{c}_ece" in metrics]
    
    if aurocs:
        metrics["robust_auroc_mean"] = sum(aurocs) / len(aurocs)
    if f1s:
        metrics["robust_f1_mean"] = sum(f1s) / len(f1s)
    if eces:
        metrics["robust_ece_mean"] = sum(eces) / len(eces)
    
    return metrics


def rank_competition(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(
        by=["robust_auroc_mean", "clean_auroc", "robust_f1_mean"],
        ascending=[False, False, False]
    )
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted["rank_a"] = df_sorted.index + 1
    return df_sorted


def rank_deployment(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(
        by=["robust_auroc_mean", "robust_f1_mean", "robust_ece_mean"],
        ascending=[False, False, True]
    )
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted["rank_b"] = df_sorted.index + 1
    return df_sorted


def generate_markdown_report(
    all_results: List[Dict],
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    model_names = [r["model_name"] for r in all_results if r["success"]]
    
    lines = []
    lines.append("# Checkpoint 对比报告\n")
    
    lines.append("## 1. 参与评估的模型\n")
    for name in model_names:
        lines.append(f"- {name}")
    lines.append("")
    
    lines.append("## 2. 各模型关键指标表\n")
    key_cols = ["model_name", "clean_auroc", "clean_f1", "clean_ece", 
                "robust_auroc_mean", "robust_f1_mean", "robust_ece_mean"]
    display_df = summary_df[key_cols].copy()
    for col in key_cols[1:]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    lines.append(display_df.to_markdown(index=False))
    lines.append("")
    
    lines.append("## 3. 排名 A（竞赛型）\n")
    rank_a_df = summary_df.sort_values("rank_a")[["rank_a", "model_name", "robust_auroc_mean", "clean_auroc", "robust_f1_mean"]]
    for _, row in rank_a_df.iterrows():
        lines.append(f"{int(row['rank_a'])}. {row['model_name']} (robust_auroc={row['robust_auroc_mean']:.4f}, clean_auroc={row['clean_auroc']:.4f}, robust_f1={row['robust_f1_mean']:.4f})")
    lines.append("")
    
    lines.append("## 4. 排名 B（部署型）\n")
    rank_b_df = summary_df.sort_values("rank_b")[["rank_b", "model_name", "robust_auroc_mean", "robust_f1_mean", "robust_ece_mean"]]
    for _, row in rank_b_df.iterrows():
        lines.append(f"{int(row['rank_b'])}. {row['model_name']} (robust_auroc={row['robust_auroc_mean']:.4f}, robust_f1={row['robust_f1_mean']:.4f}, robust_ece={row['robust_ece_mean']:.4f})")
    lines.append("")
    
    lines.append("## 5. 最终推荐\n")
    primary_model = rank_a_df.iloc[0]["model_name"]
    secondary_model = rank_b_df.iloc[0]["model_name"] if rank_b_df.iloc[0]["model_name"] != primary_model else rank_b_df.iloc[1]["model_name"]
    
    lines.append(f"- 主模型：{primary_model}")
    lines.append(f"- 第二备选模型：{secondary_model}")
    lines.append("- 推荐理由：")
    lines.append("  - 主模型在竞赛型排名中第一，适合leaderboard提交，AUROC优先")
    lines.append("  - 第二备选模型在部署型排名中表现优秀，更稳健，F1/ECE更平衡")
    
    report = "\n".join(lines)
    
    report_path = output_dir / "checkpoint_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_clip_224.openai")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="best.pth,epoch_018.pth,epoch_019.pth,epoch_020.pth,latest.pth"
    )
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_names = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    ckpt_paths = [ckpt_dir / name for name in ckpt_names]
    
    print(f"Checking {len(ckpt_paths)} checkpoints...")
    md5_map = check_duplicates(ckpt_paths)
    
    unique_ckpts: List[Tuple[Path, str]] = []
    for md5, paths in md5_map.items():
        primary_path = min(paths, key=lambda p: p.name)
        unique_ckpts.append((primary_path, md5))
        if len(paths) > 1:
            print(f"  Duplicate (md5={md5[:12]}...): {[p.name for p in paths]} - using {primary_path.name}")
    
    all_results = []
    summary_data = []
    
    for ckpt_path, md5 in unique_ckpts:
        model_name = ckpt_path.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        out_csv = output_dir / f"eval_{model_name.replace('.', '_')}.csv"
        
        success, df, error = evaluate_single_checkpoint(
            ckpt_path=ckpt_path,
            data_root=data_root,
            out_csv=out_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            backbone_name=args.backbone_name,
        )
        
        result = {
            "model_name": model_name,
            "ckpt_path": str(ckpt_path),
            "md5": md5,
            "success": success,
            "error": error,
        }
        
        if success and df is not None:
            metrics = extract_metrics(df)
            result["metrics"] = metrics
            summary_row = {
                "model_name": model_name,
                **metrics
            }
            summary_data.append(summary_row)
        
        all_results.append(result)
    
    summary_df = pd.DataFrame(summary_data)
    
    if not summary_df.empty:
        summary_df = rank_competition(summary_df)
        summary_df = rank_deployment(summary_df)
        
        summary_csv = output_dir / "checkpoint_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nSummary saved to: {summary_csv}")
        
        report = generate_markdown_report(all_results, summary_df, output_dir)
        print(f"\n{report}")
    else:
        print("\nNo successful evaluations!")
    
    results_json = output_dir / "all_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {results_json}")


if __name__ == "__main__":
    main()
