#!/usr/bin/env python3
"""
小 batch forward/backward 烟测脚本 V3 - 最终优化版
- 学习率: 5e-6 (进一步降低)
- 使用 aigc_focus 增强 (clean:mild:hard = 0.50:0.35:0.15)
- 对漏检样本使用 aigc_conservative 模式 (0.65:0.25:0.10)
- 漏检样本加权 loss (3x)
- 融合分支权重约束 (max_semantic=0.45, min_noise=0.20)
- Warmup: 5 steps
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire.augmentations import build_train_transform
from ai_image_detector.ntire.model import HybridAIGCDetector


class SmokeTestDatasetV3(torch.utils.data.Dataset):
    """Small dataset for smoke testing V3 with per-sample transform."""
    
    def __init__(self, image_paths: List[Path], labels: List[int], transform, sample_weights: List[float] = None, missed_samples: set = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.sample_weights = sample_weights if sample_weights else [1.0] * len(labels)
        self.missed_samples = missed_samples or set()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        weight = self.sample_weights[idx]
        img_name = str(img_path.name)
        
        image = Image.open(img_path).convert("RGB")
        arr = np.array(image)
        
        # Use conservative transform for missed AIGC samples
        if img_name in self.missed_samples and label == 1:
            conservative_transform = build_train_transform(224, chain_mix=True, chain_mix_strength="aigc_conservative")
            image = conservative_transform(image=arr)["image"]
        else:
            if self.transform:
                image = self.transform(image=arr)["image"]
        
        return image, torch.tensor(label, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32), img_name


class WarmupSchedulerV3:
    """Linear warmup scheduler with 5 steps."""
    def __init__(self, optimizer, warmup_steps: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return self.optimizer.param_groups[0]['lr']


def compute_bce_loss_v3(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute weighted BCE loss with temperature scaling."""
    calibrated_logits = logits / max(temperature, 1e-6)
    loss = F.binary_cross_entropy_with_logits(calibrated_logits.squeeze(), labels, reduction='none')
    weighted_loss = (loss * weights).mean()
    return weighted_loss


def compute_consistency_loss_v3(logits1: torch.Tensor, logits2: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute consistency loss between two views."""
    prob1 = torch.sigmoid(logits1 / max(temperature, 1e-6))
    prob2 = torch.sigmoid(logits2 / max(temperature, 1e-6))
    loss = F.mse_loss(prob1, prob2)
    return loss


def smoke_test_v3(args):
    """Run smoke test with V3 optimized settings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model with weight constraints (V3 settings)
    model = HybridAIGCDetector(
        backbone_name="vit_base_patch16_clip_224.openai",
        pretrained_backbone=False,
        max_semantic_weight=0.45,
        min_noise_weight=0.20,
        use_weight_regularization=True,
        weight_reg_lambda=0.01,
    )
    
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        temperature = float(ckpt.get("temperature", 1.0))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        temperature = 1.0
        print(f"No checkpoint found, using random weights")
    
    model.to(device)
    model.train()  # Set to train mode for backward pass
    
    # Optimizer with V3 lower learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # V3 Warmup scheduler with 5 steps
    warmup_scheduler = WarmupSchedulerV3(optimizer, args.warmup_steps, args.lr)
    
    # Load data
    labels_df = pd.read_csv(args.labels)
    folder = Path(args.folder)
    
    # V3: Identify missed AIGC samples for special handling
    missed_aigc_samples = {'aigc1.jpg', 'aigc2.png', 'aigc3.png', 'aigc4.png', 'aigc7.png', 'aigc9.png'}
    
    image_paths = []
    labels = []
    sample_weights = []
    
    for _, row in labels_df.iterrows():
        img_path = folder / row["image_name"]
        if img_path.exists():
            image_paths.append(img_path)
            labels.append(row["label"])
            
            # V3: Higher weight for missed AIGC samples (3x)
            if row["image_name"] in missed_aigc_samples and row["label"] == 1:
                sample_weights.append(args.missed_sample_weight)
            else:
                sample_weights.append(1.0)
    
    # Limit batch size
    image_paths = image_paths[:args.batch_size]
    labels = labels[:args.batch_size]
    sample_weights = sample_weights[:args.batch_size]
    
    print(f"\n{'='*60}")
    print("Smoke Test V3 - Final Optimization")
    print("="*60)
    print(f"Images: {len(image_paths)} (AIGC={sum(labels)}, Real={len(labels)-sum(labels)})")
    print(f"Learning rate: {args.lr} (with {args.warmup_steps} warmup steps)")
    print(f"Missed AIGC weight: {args.missed_sample_weight}x")
    print(f"Enhancement: aigc_focus (0.50:0.35:0.15)")
    print(f"Missed samples: aigc_conservative (0.65:0.25:0.10)")
    
    # V3: Transforms for dual views using aigc_focus mode
    transform1 = build_train_transform(224, chain_mix=True, chain_mix_strength="aigc_focus")
    transform2 = build_train_transform(224, chain_mix=True, chain_mix_strength="aigc_focus")
    
    # V3: Create datasets with missed sample handling
    dataset1 = SmokeTestDatasetV3(image_paths, labels, transform1, sample_weights, missed_aigc_samples)
    dataset2 = SmokeTestDatasetV3(image_paths, labels, transform2, sample_weights, missed_aigc_samples)
    
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=len(image_paths), shuffle=False)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=len(image_paths), shuffle=False)
    
    results = []
    all_losses = []
    
    # Multiple forward/backward iterations
    print(f"\n{'='*60}")
    print("Training Iterations")
    print("="*60)
    
    for iteration in range(args.num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{args.num_iterations} ---")
        
        for batch_idx, ((images1, labels1, weights1, names1), (images2, labels2, weights2, names2)) in enumerate(zip(loader1, loader2)):
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels_tensor = labels1.to(device)
            weights_tensor = weights1.to(device)
            
            # Warmup step
            current_lr = warmup_scheduler.step()
            if iteration == 0 and batch_idx == 0:
                print(f"Current LR: {current_lr:.2e}")
            
            # View 1
            out1 = model(images1)
            logits1 = out1["logit"].squeeze()
            fusion_weights1 = out1["fusion_weights"]
            fusion_reg_loss1 = out1.get("fusion_reg_loss", torch.tensor(0.0))
            
            # View 2
            out2 = model(images2)
            logits2 = out2["logit"].squeeze()
            fusion_weights2 = out2["fusion_weights"]
            fusion_reg_loss2 = out2.get("fusion_reg_loss", torch.tensor(0.0))
            
            # Compute losses
            bce_loss = compute_bce_loss_v3(logits1, labels_tensor, weights_tensor, temperature)
            consistency_loss = compute_consistency_loss_v3(logits1, logits2, temperature)
            fusion_reg_loss = (fusion_reg_loss1 + fusion_reg_loss2) / 2
            
            # Total loss
            total_loss = bce_loss + args.consistency_weight * consistency_loss + fusion_reg_loss
            
            # Print per-sample info (only first iteration)
            if iteration == 0:
                probs1 = torch.sigmoid(logits1 / max(temperature, 1e-6)).cpu().detach().numpy()
                probs2 = torch.sigmoid(logits2 / max(temperature, 1e-6)).cpu().detach().numpy()
                
                print(f"\n  Losses:")
                print(f"    BCE: {bce_loss.item():.6f}")
                print(f"    Consistency: {consistency_loss.item():.6f}")
                print(f"    Fusion Reg: {fusion_reg_loss.item():.6f}")
                print(f"    Total: {total_loss.item():.6f}")
                
                print(f"\n  Per-sample results:")
                print(f"  {'Image':<15} {'Label':>6} {'Weight':>7} {'Prob':>8} {'Fusion (S/F/N)':>20} {'Status':>10}")
                print(f"  {'-'*70}")
                
                for i, name in enumerate(names1):
                    status = "[MISSED]" if name in missed_aigc_samples else ""
                    fusion_str = f"({fusion_weights1[i][0]:.2f}/{fusion_weights1[i][1]:.2f}/{fusion_weights1[i][2]:.2f})"
                    print(f"  {name:<15} {int(labels[i]):>6} {sample_weights[i]:>7.1f} {probs1[i]:>8.4f} {fusion_str:>20} {status:>10}")
                    
                    if iteration == 0:
                        results.append({
                            "iteration": iteration + 1,
                            "image_name": name,
                            "label": int(labels[i]),
                            "sample_weight": sample_weights[i],
                            "prob_view1": float(probs1[i]),
                            "prob_view2": float(probs2[i]),
                            "prob_diff": float(abs(probs1[i] - probs2[i])),
                            "fusion_semantic": float(fusion_weights1[i][0]),
                            "fusion_frequency": float(fusion_weights1[i][1]),
                            "fusion_noise": float(fusion_weights1[i][2]),
                        })
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Check gradients
            total_grad_norm = 0.0
            max_grad = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    max_grad = max(max_grad, param.grad.abs().max().item())
            
            total_grad_norm = total_grad_norm ** 0.5
            
            # Optimizer step
            optimizer.step()
            
            # Store losses
            all_losses.append({
                "iteration": iteration + 1,
                "bce_loss": float(bce_loss.item()),
                "consistency_loss": float(consistency_loss.item()),
                "fusion_reg_loss": float(fusion_reg_loss.item()),
                "total_loss": float(total_loss.item()),
                "grad_norm": float(total_grad_norm),
                "max_grad": float(max_grad),
                "lr": float(current_lr),
            })
            
            # Check for NaN/Inf
            has_nan = False
            has_inf = False
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    has_nan = True
                if torch.isinf(param).any():
                    has_inf = True
            
            if has_nan or has_inf:
                print(f"  ⚠️  NaN/Inf detected!")
                break
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "smoke_test_v3_results.csv", index=False)
    
    losses_df = pd.DataFrame(all_losses)
    losses_df.to_csv(output_dir / "smoke_test_v3_losses.csv", index=False)
    
    # Compute metrics for missed AIGC samples
    if results:
        missed_results = [r for r in results if r["image_name"] in missed_aigc_samples]
        if missed_results:
            avg_prob = np.mean([r["prob_view1"] for r in missed_results])
            print(f"\n  Missed AIGC samples average probability: {avg_prob:.4f}")
    
    # Final summary
    report = {
        "config": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "consistency_weight": args.consistency_weight,
            "missed_sample_weight": args.missed_sample_weight,
            "warmup_steps": args.warmup_steps,
            "num_iterations": args.num_iterations,
            "temperature": temperature,
            "batch_size": len(image_paths),
            "enhancement_focus": "0.50:0.35:0.15",
            "enhancement_conservative": "0.65:0.25:0.10",
        },
        "losses_summary": {
            "final_bce_loss": all_losses[-1]["bce_loss"] if all_losses else 0,
            "final_total_loss": all_losses[-1]["total_loss"] if all_losses else 0,
            "avg_grad_norm": np.mean([l["grad_norm"] for l in all_losses]) if all_losses else 0,
            "max_grad": max([l["max_grad"] for l in all_losses]) if all_losses else 0,
        },
        "checks": {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "gradients_ok": all(l["grad_norm"] < 100 for l in all_losses) if all_losses else True,
        }
    }
    
    with open(output_dir / "smoke_test_v3_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Smoke Test V3 Summary")
    print("="*60)
    print(f"Final BCE Loss: {report['losses_summary']['final_bce_loss']:.6f}")
    print(f"Final Total Loss: {report['losses_summary']['final_total_loss']:.6f}")
    print(f"Avg Gradient Norm: {report['losses_summary']['avg_grad_norm']:.6f}")
    print(f"Max Gradient: {report['losses_summary']['max_grad']:.6f}")
    
    if report["checks"]["gradients_ok"] and not has_nan and not has_inf:
        print("✓ 测试通过 - 数值稳定，可以进行完整训练")
    else:
        print("✗ 测试失败 - 检测到数值问题")
    
    print(f"\n输出文件:")
    print(f"  {output_dir / 'smoke_test_v3_results.csv'}")
    print(f"  {output_dir / 'smoke_test_v3_losses.csv'}")
    print(f"  {output_dir / 'smoke_test_v3_report.json'}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--folder", type=str, default="photos_test")
    parser.add_argument("--labels", type=str, default="photos_test/labels.csv")
    parser.add_argument("--batch-size", type=int, default=19, help="Batch size for smoke test")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate (V3: 5e-6)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--consistency-weight", type=float, default=0.1, help="Consistency loss weight")
    parser.add_argument("--missed-sample-weight", type=float, default=3.0, help="Weight for missed AIGC samples (V3: 3x)")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps (V3: 5)")
    parser.add_argument("--num-iterations", type=int, default=3, help="Number of forward/backward iterations")
    parser.add_argument("--output-dir", type=str, default="outputs/smoke_test_v3")
    args = parser.parse_args()
    
    smoke_test_v3(args)


if __name__ == "__main__":
    main()
