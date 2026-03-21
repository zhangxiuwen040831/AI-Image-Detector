"""
分析 photos_test 推理结果
- 读取 photos_test/labels.csv
- 合并三档推理 CSV
- 计算 per-profile prob/logit 统计
- 生成 predictions_with_labels_all_profiles.csv
- 生成 summary_table.csv
- 生成 anomaly_flags_by_profile.csv
- 生成 analysis_report.json
- 生成 strategy_suggestions.txt
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("photos_test_outputs")
LABELS_CSV = Path("photos_test/labels.csv")

def compute_metrics(df: pd.DataFrame, label_col: str = "label", pred_col: str = "pred_label"):
    """计算 precision, recall, f1, accuracy"""
    tp = ((df[label_col] == 1) & (df[pred_col] == 1)).sum()
    fp = ((df[label_col] == 0) & (df[pred_col] == 1)).sum()
    tn = ((df[label_col] == 0) & (df[pred_col] == 0)).sum()
    fn = ((df[label_col] == 1) & (df[pred_col] == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(df) if len(df) > 0 else 0.0
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

def main():
    # 读取 labels
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df["image_name"] = labels_df["image_name"].astype(str)
    
    # 读取三档推理结果
    profiles = ["f1", "balanced", "recall"]
    profile_dfs = {}
    for profile in profiles:
        csv_path = OUTPUT_DIR / f"inference_{profile}.csv"
        df = pd.read_csv(csv_path)
        df["image_name"] = df["path"].apply(lambda x: Path(x).name)
        profile_dfs[profile] = df
    
    # 合并所有 profile 数据
    merged = labels_df.copy()
    for profile in profiles:
        df = profile_dfs[profile][["image_name", "probability", "raw_logit", "calibrated_logit", "pred_label"]]
        df = df.rename(columns={
            "probability": f"prob_{profile}",
            "raw_logit": f"logit_raw_{profile}",
            "calibrated_logit": f"logit_calib_{profile}",
            "pred_label": f"pred_{profile}",
        })
        merged = merged.merge(df, on="image_name", how="left")
    
    # 保存合并后的预测结果
    merged.to_csv(OUTPUT_DIR / "predictions_with_labels_all_profiles.csv", index=False)
    print(f"Saved: predictions_with_labels_all_profiles.csv")
    
    # 计算每个 profile 的统计信息
    summary_rows = []
    for profile in profiles:
        metrics = compute_metrics(merged, "label", f"pred_{profile}")
        prob_stats = {
            "prob_mean": float(merged[f"prob_{profile}"].mean()),
            "prob_std": float(merged[f"prob_{profile}"].std()),
            "prob_min": float(merged[f"prob_{profile}"].min()),
            "prob_max": float(merged[f"prob_{profile}"].max()),
        }
        row = {
            "profile": profile,
            **metrics,
            **prob_stats,
        }
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary_table.csv", index=False)
    print(f"Saved: summary_table.csv")
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    # 生成 anomaly flags（预测错误的样本）
    anomaly_rows = []
    for _, row in merged.iterrows():
        flags = {}
        for profile in profiles:
            flags[f"anomaly_{profile}"] = int(row["label"] != row[f"pred_{profile}"])
        anomaly_rows.append({
            "image_name": row["image_name"],
            "label": row["label"],
            **{f"prob_{p}": row[f"prob_{p}"] for p in profiles},
            **flags,
        })
    
    anomaly_df = pd.DataFrame(anomaly_rows)
    anomaly_df.to_csv(OUTPUT_DIR / "anomaly_flags_by_profile.csv", index=False)
    print(f"\nSaved: anomaly_flags_by_profile.csv")
    
    # 生成 analysis_report.json
    report = {
        "total_samples": int(len(merged)),
        "positive_samples": int((merged["label"] == 1).sum()),
        "negative_samples": int((merged["label"] == 0).sum()),
        "profiles": {},
    }
    for profile in profiles:
        metrics = compute_metrics(merged, "label", f"pred_{profile}")
        report["profiles"][profile] = {
            "metrics": metrics,
            "prob_stats": {
                "mean": float(merged[f"prob_{profile}"].mean()),
                "std": float(merged[f"prob_{profile}"].std()),
                "min": float(merged[f"prob_{profile}"].min()),
                "max": float(merged[f"prob_{profile}"].max()),
            },
        }
    
    with open(OUTPUT_DIR / "analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved: analysis_report.json")
    
    # 生成 strategy_suggestions.txt
    suggestions = []
    suggestions.append("=" * 60)
    suggestions.append("策略建议 (Strategy Suggestions)")
    suggestions.append("=" * 60)
    suggestions.append("")
    
    # 找出最佳 profile
    best_f1_profile = summary_df.loc[summary_df["f1"].idxmax(), "profile"]
    best_acc_profile = summary_df.loc[summary_df["accuracy"].idxmax(), "profile"]
    
    suggestions.append(f"1. F1 最优的 Profile: {best_f1_profile}")
    suggestions.append(f"   - F1: {summary_df.loc[summary_df['profile']==best_f1_profile, 'f1'].values[0]:.4f}")
    suggestions.append(f"   - Precision: {summary_df.loc[summary_df['profile']==best_f1_profile, 'precision'].values[0]:.4f}")
    suggestions.append(f"   - Recall: {summary_df.loc[summary_df['profile']==best_f1_profile, 'recall'].values[0]:.4f}")
    suggestions.append("")
    
    suggestions.append(f"2. 准确率最优的 Profile: {best_acc_profile}")
    suggestions.append(f"   - Accuracy: {summary_df.loc[summary_df['profile']==best_acc_profile, 'accuracy'].values[0]:.4f}")
    suggestions.append("")
    
    # 分析概率分布
    suggestions.append("3. 概率分布分析:")
    for profile in profiles:
        prob_mean = summary_df.loc[summary_df["profile"]==profile, "prob_mean"].values[0]
        prob_std = summary_df.loc[summary_df["profile"]==profile, "prob_std"].values[0]
        suggestions.append(f"   - {profile}: mean={prob_mean:.4f}, std={prob_std:.4f}")
    suggestions.append("")
    
    # 找出异常样本
    suggestions.append("4. 预测错误样本:")
    for profile in profiles:
        anomalies = anomaly_df[anomaly_df[f"anomaly_{profile}"] == 1]
        if len(anomalies) > 0:
            suggestions.append(f"   - {profile} profile 错误样本:")
            for _, row in anomalies.iterrows():
                suggestions.append(f"     * {row['image_name']} (label={row['label']}, prob={row[f'prob_{profile}']:.4f})")
        else:
            suggestions.append(f"   - {profile} profile: 无错误样本")
    suggestions.append("")
    
    suggestions.append("5. 建议:")
    suggestions.append(f"   - 推荐使用 {best_f1_profile} profile 以获得最佳 F1 分数")
    suggestions.append("   - 如需更高召回率，请使用 recall profile")
    suggestions.append("   - 如需平衡 precision/recall，请使用 balanced profile")
    
    with open(OUTPUT_DIR / "strategy_suggestions.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(suggestions))
    print(f"Saved: strategy_suggestions.txt")
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
