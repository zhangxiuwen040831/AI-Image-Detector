"""
阈值扫描脚本
- 读取 predictions_with_labels_all_profiles.csv
- 对 prob_balanced 扫描 threshold=[0.2,0.6] 步长0.02
- 输出 sweep_best_official_full.csv (threshold, precision, recall, f1)
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def compute_metrics_at_threshold(df: pd.DataFrame, prob_col: str, threshold: float, label_col: str = "label"):
    """计算指定阈值下的 precision, recall, f1"""
    pred = (df[prob_col] >= threshold).astype(int)
    tp = ((df[label_col] == 1) & (pred == 1)).sum()
    fp = ((df[label_col] == 0) & (pred == 1)).sum()
    tn = ((df[label_col] == 0) & (pred == 0)).sum()
    fn = ((df[label_col] == 1) & (pred == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(df) if len(df) > 0 else 0.0
    
    return {
        "threshold": float(threshold),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", type=str, required=True, help="Path to predictions CSV")
    parser.add_argument("--out-csv", type=str, required=True, help="Output sweep CSV path")
    parser.add_argument("--prob-col", type=str, default="prob_balanced", help="Probability column to sweep")
    parser.add_argument("--th-min", type=float, default=0.2, help="Minimum threshold")
    parser.add_argument("--th-max", type=float, default=0.6, help="Maximum threshold")
    parser.add_argument("--th-step", type=float, default=0.02, help="Threshold step")
    args = parser.parse_args()
    
    # 读取预测结果
    df = pd.read_csv(args.pred_csv)
    
    # 生成阈值范围
    thresholds = np.arange(args.th_min, args.th_max + args.th_step, args.th_step)
    
    # 扫描每个阈值
    results = []
    for th in thresholds:
        metrics = compute_metrics_at_threshold(df, args.prob_col, th)
        results.append(metrics)
    
    # 保存结果
    results_df = pd.DataFrame(results)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    
    # 打印最佳结果
    best_f1 = results_df.loc[results_df["f1"].idxmax()]
    best_balance = results_df.loc[(results_df["precision"] - results_df["recall"]).abs().idxmin()]
    best_recall = results_df.loc[results_df["recall"].idxmax()]
    
    print(f"\n阈值扫描完成: {args.out_csv}")
    print(f"扫描范围: [{args.th_min}, {args.th_max}], 步长: {args.th_step}")
    print(f"\n最佳结果:")
    print(f"  - F1 最优: threshold={best_f1['threshold']:.2f}, f1={best_f1['f1']:.4f}, precision={best_f1['precision']:.4f}, recall={best_f1['recall']:.4f}")
    print(f"  - 平衡: threshold={best_balance['threshold']:.2f}, f1={best_balance['f1']:.4f}, precision={best_balance['precision']:.4f}, recall={best_balance['recall']:.4f}")
    print(f"  - 最高召回: threshold={best_recall['threshold']:.2f}, f1={best_recall['f1']:.4f}, precision={best_recall['precision']:.4f}, recall={best_recall['recall']:.4f}")

if __name__ == "__main__":
    main()
