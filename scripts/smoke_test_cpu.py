from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_image_detector.ntire.augmentations import build_eval_transform, build_train_transform
from ai_image_detector.ntire.dataset import NTIRETrainDataset, print_dataset_sanity
from ai_image_detector.ntire.losses import HybridDetectionLoss
from ai_image_detector.ntire.metrics import compute_metrics
from ai_image_detector.ntire.model import HybridAIGCDetector
from make_tiny_subset import make_tiny_subset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_smoke(args: argparse.Namespace) -> None:
    print(f"\n# NTIRE 本地 CPU 最小规模测试报告")
    print(f"\n## 1. 测试配置")
    print(f"- data root: {args.data_root}")
    print(f"- tiny subset csv: {args.tiny_csv}")
    print(f"- subset size: {args.subset_size}")
    print(f"- batch size: {args.batch_size}")
    print(f"- image size: {args.image_size}")
    print(f"- freeze backbone: {args.freeze_backbone}")
    print(f"- epoch: 1")
    print(f"- device: cpu")
    set_seed(args.seed)
    tiny_csv = Path(args.tiny_csv)
    if not tiny_csv.exists():
        make_tiny_subset(
            data_root=Path(args.data_root),
            out_csv=tiny_csv,
            subset_size=args.subset_size,
            seed=args.seed,
        )
    train_ds = NTIRETrainDataset(
        root_dir=args.data_root,
        subset_csv=tiny_csv,
        transform=build_train_transform(args.image_size),
    )
    eval_ds = NTIRETrainDataset(
        root_dir=args.data_root,
        subset_csv=tiny_csv,
        transform=build_eval_transform(args.image_size),
    )
    total_samples = 0
    real_count = 0
    fake_count = 0
    shards_used = set()
    
    print(f"\n## 2. 数据检查")
    for i in range(len(train_ds)):
        item = train_ds[i]
        label = item[1]
        meta = item[2]
        total_samples += 1
        if label < 0.5:
            real_count += 1
        else:
            fake_count += 1
        shards_used.add(meta.get("shard_name", "unknown"))
    
    print(f"- total: {total_samples}")
    print(f"- real: {real_count}")
    print(f"- fake: {fake_count}")
    print(f"- shards used: {len(shards_used)}")
    print(f"- shard distribution: {list(shards_used)[:5]}...")

    n = len(train_ds)
    split = max(int(n * 0.8), 1)
    train_idx = list(range(split))
    val_idx = list(range(split, n))
    
    # Use Subset for splitting
    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(eval_ds, val_idx)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    device = torch.device("cpu")
    model = HybridAIGCDetector(
        backbone_name=args.backbone_name,
        fused_dim=args.fused_dim,
        head_hidden_dim=args.head_hidden_dim,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = HybridDetectionLoss()

    print("\n## 3. 运行结果")
    import time
    start_time = time.time()
    
    total_loss_sum = 0.0
    main_loss_sum = 0.0
    aux_global_sum = 0.0
    aux_freq_sum = 0.0
    aux_noise_sum = 0.0
    
    fusion_global_w = []
    fusion_freq_w = []
    fusion_noise_w = []
    
    step_times = []

    for epoch in range(1):
        losses = []
        y_true = []
        y_prob = []
        
        for step, (images, labels, _) in enumerate(train_loader):
            step_start = time.time()
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            loss_dict = loss_fn(out, labels)
            
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()
            
            # Record losses
            total_loss_sum += loss.item()
            main_loss_sum += loss_dict.get("main_bce_loss", torch.tensor(0.0)).item()
            aux_global_sum += loss_dict.get("semantic_logit_loss", torch.tensor(0.0)).item()
            aux_freq_sum += loss_dict.get("freq_logit_loss", torch.tensor(0.0)).item()
            aux_noise_sum += loss_dict.get("noise_logit_loss", torch.tensor(0.0)).item()
            
            # Record fusion weights
            w = out["fusion_weights"].detach().cpu().numpy() # [B, 3]
            fusion_global_w.extend(w[:, 0].tolist())
            fusion_freq_w.extend(w[:, 1].tolist())
            fusion_noise_w.extend(w[:, 2].tolist())
            
            losses.append(float(loss.detach().item()))
            y_true.extend(labels.detach().cpu().numpy().tolist())
            prob_list = torch.sigmoid(out["logit"]).detach().view(-1).cpu().numpy().tolist()
            y_prob.extend(prob_list)
            
            step_end = time.time()
            step_times.append(step_end - step_start)

    total_runtime = time.time() - start_time
    avg_step_time = np.mean(step_times) if step_times else 0.0
    
    num_steps = len(train_loader)
    print(f"- total loss: {total_loss_sum / num_steps:.4f}")
    print(f"- main loss: {main_loss_sum / num_steps:.4f}")
    print(f"- aux/global loss: {aux_global_sum / num_steps:.4f}")
    print(f"- aux/freq loss: {aux_freq_sum / num_steps:.4f}")
    print(f"- aux/noise loss: {aux_noise_sum / num_steps:.4f}")

    train_metrics = compute_metrics(y_true=y_true, y_prob=y_prob)
    print(f"- AUROC: {train_metrics['auroc']:.4f}")
    print(f"- AUPRC: {train_metrics['auprc']:.4f}")
    print(f"- F1: {train_metrics['f1']:.4f}")
    print(f"- Precision: {train_metrics['precision']:.4f}")
    print(f"- Recall: {train_metrics['recall']:.4f}")
    print(f"- ECE: {train_metrics['ece']:.4f}")
    
    print("\n## 4. Fusion 权重统计")
    mean_g = np.mean(fusion_global_w) if fusion_global_w else 0.0
    mean_f = np.mean(fusion_freq_w) if fusion_freq_w else 0.0
    mean_n = np.mean(fusion_noise_w) if fusion_noise_w else 0.0
    print(f"- mean global weight: {mean_g:.4f}")
    print(f"- mean freq weight: {mean_f:.4f}")
    print(f"- mean noise weight: {mean_n:.4f}")
    
    collapse_msg = "No collapse detected"
    if mean_g < 0.05 or mean_f < 0.05 or mean_n < 0.05:
        collapse_msg = "WARNING: Potential collapse detected (<0.05)"
    print(f"- collapse check: {collapse_msg}")
    
    print("\n## 5. 数值稳定性检查")
    logits_finite = np.all(np.isfinite(y_prob)) # prob is sigmoid(logit), checking prob is enough for nan/inf usually
    probs_valid = np.all((np.array(y_prob) >= 0) & (np.array(y_prob) <= 1))
    print(f"- logits finite: {logits_finite}")
    print(f"- probs valid: {probs_valid}")
    print(f"- nan/inf: {'Found' if not logits_finite else 'None'}")
    
    print("\n## 6. 耗时")
    print(f"- total runtime: {total_runtime:.2f}s")
    print(f"- avg step runtime: {avg_step_time:.4f}s")


    model.eval()
    with torch.no_grad():
        y_true = []
        y_prob = []
        for images, labels, _ in val_loader:
            out = model(images.to(device))
            y_true.extend(labels.numpy().tolist())
            y_prob.extend(torch.sigmoid(out["logit"]).view(-1).cpu().numpy().tolist())
        val_metrics = compute_metrics(y_true=y_true, y_prob=y_prob)
        print(
            "Smoke Val | "
            f"auroc={val_metrics['auroc']:.4f} "
            f"auprc={val_metrics['auprc']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"precision={val_metrics['precision']:.4f} "
            f"recall={val_metrics['recall']:.4f} "
            f"ece={val_metrics['ece']:.4f}"
        )
    print("Smoke test completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"C:\Users\32902\Desktop\ai-image-detector\NTIRE-RobustAIGenDetection-train",
    )
    parser.add_argument("--tiny-csv", type=str, default=str(PROJECT_ROOT / "tiny_subset_500.csv"))
    parser.add_argument("--subset-size", type=int, default=320)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fused-dim", type=int, default=256)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--backbone-name", type=str, default="resnet18")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_smoke(parse_args())
