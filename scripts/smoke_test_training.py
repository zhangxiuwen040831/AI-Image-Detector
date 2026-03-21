#!/usr/bin/env python3
"""
小 batch forward/backward 烟测脚本
测试 BCE + 一致性 loss 的数值合理性
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


class SmokeTestDataset(torch.utils.data.Dataset):
    """Small dataset for smoke testing."""
    
    def __init__(self, image_paths: List[Path], labels: List[int], transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        arr = np.array(image)
        
        if self.transform:
            image = self.transform(image=arr)["image"]
        
        return image, torch.tensor(label, dtype=torch.float32), str(img_path.name)


def compute_bce_loss(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute BCE loss with temperature scaling."""
    calibrated_logits = logits / max(temperature, 1e-6)
    loss = F.binary_cross_entropy_with_logits(calibrated_logits.squeeze(), labels)
    return loss


def compute_consistency_loss(logits1: torch.Tensor, logits2: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute consistency loss between two views."""
    prob1 = torch.sigmoid(logits1 / max(temperature, 1e-6))
    prob2 = torch.sigmoid(logits2 / max(temperature, 1e-6))
    loss = F.mse_loss(prob1, prob2)
    return loss


def smoke_test(args):
    """Run smoke test with small batch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = HybridAIGCDetector(
        backbone_name="vit_base_patch16_clip_224.openai",
        pretrained_backbone=False
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
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Load data
    labels_df = pd.read_csv(args.labels)
    folder = Path(args.folder)
    
    image_paths = []
    labels = []
    for _, row in labels_df.iterrows():
        img_path = folder / row["image_name"]
        if img_path.exists():
            image_paths.append(img_path)
            labels.append(row["label"])
    
    # Limit batch size
    image_paths = image_paths[:args.batch_size]
    labels = labels[:args.batch_size]
    
    print(f"\nSmoke test with {len(image_paths)} images")
    print(f"Batch composition: AIGC={sum(labels)}, Real={len(labels)-sum(labels)}")
    
    # Transforms for dual views
    transform1 = build_train_transform(224, chain_mix=True, chain_mix_strength="aigc_focus")
    transform2 = build_train_transform(224, chain_mix=True, chain_mix_strength="aigc_focus")
    
    # Create datasets
    dataset1 = SmokeTestDataset(image_paths, labels, transform1)
    dataset2 = SmokeTestDataset(image_paths, labels, transform2)
    
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=len(image_paths), shuffle=False)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=len(image_paths), shuffle=False)
    
    results = []
    
    # Forward pass
    print("\n" + "="*60)
    print("Forward Pass")
    print("="*60)
    
    for batch_idx, ((images1, labels1, names1), (images2, labels2, names2)) in enumerate(zip(loader1, loader2)):
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels_tensor = labels1.to(device)
        
        # View 1
        out1 = model(images1)
        logits1 = out1["logit"].squeeze()
        fusion_weights1 = out1["fusion_weights"]
        
        # View 2
        out2 = model(images2)
        logits2 = out2["logit"].squeeze()
        fusion_weights2 = out2["fusion_weights"]
        
        # Compute losses
        bce_loss = compute_bce_loss(logits1, labels_tensor, temperature)
        consistency_loss = compute_consistency_loss(logits1, logits2, temperature)
        
        # Total loss
        total_loss = bce_loss + args.consistency_weight * consistency_loss
        
        # Print per-sample info
        probs1 = torch.sigmoid(logits1 / max(temperature, 1e-6)).cpu().detach().numpy()
        probs2 = torch.sigmoid(logits2 / max(temperature, 1e-6)).cpu().detach().numpy()
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  BCE Loss: {bce_loss.item():.6f}")
        print(f"  Consistency Loss: {consistency_loss.item():.6f}")
        print(f"  Total Loss: {total_loss.item():.6f}")
        
        print(f"\n  Per-sample details:")
        for i, name in enumerate(names1):
            print(f"    {name}: label={int(labels[i])}, "
                  f"prob1={probs1[i]:.4f}, prob2={probs2[i]:.4f}, "
                  f"diff={abs(probs1[i]-probs2[i]):.4f}, "
                  f"fusion=({fusion_weights1[i][0]:.3f}, {fusion_weights1[i][1]:.3f}, {fusion_weights1[i][2]:.3f})")
            
            results.append({
                "image_name": name,
                "label": int(labels[i]),
                "prob_view1": float(probs1[i]),
                "prob_view2": float(probs2[i]),
                "prob_diff": float(abs(probs1[i] - probs2[i])),
                "fusion_semantic": float(fusion_weights1[i][0]),
                "fusion_frequency": float(fusion_weights1[i][1]),
                "fusion_noise": float(fusion_weights1[i][2]),
            })
        
        # Backward pass
        print(f"\n" + "-"*60)
        print("Backward Pass")
        print("-"*60)
        
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
        
        print(f"  Gradient norm: {total_grad_norm:.6f}")
        print(f"  Max gradient: {max_grad:.6f}")
        
        # Optimizer step
        optimizer.step()
        
        print(f"  Optimizer step completed")
        
        # Check for NaN/Inf
        has_nan = False
        has_inf = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                has_nan = True
                print(f"  WARNING: NaN detected in {name}")
            if torch.isinf(param).any():
                has_inf = True
                print(f"  WARNING: Inf detected in {name}")
        
        if not has_nan and not has_inf:
            print(f"  ✓ No NaN/Inf detected in parameters")
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "smoke_test_results.csv", index=False)
        
        report = {
            "losses": {
                "bce_loss": float(bce_loss.item()),
                "consistency_loss": float(consistency_loss.item()),
                "total_loss": float(total_loss.item()),
            },
            "gradients": {
                "total_norm": float(total_grad_norm),
                "max_grad": float(max_grad),
            },
            "checks": {
                "has_nan": has_nan,
                "has_inf": has_inf,
                "gradients_ok": total_grad_norm < 100 and max_grad < 10,
            },
            "config": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "consistency_weight": args.consistency_weight,
                "temperature": temperature,
                "batch_size": len(image_paths),
            }
        }
        
        with open(output_dir / "smoke_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n输出文件:")
        print(f"  {output_dir / 'smoke_test_results.csv'}")
        print(f"  {output_dir / 'smoke_test_report.json'}")
        
        # Summary
        print(f"\n" + "="*60)
        print("Smoke Test Summary")
        print("="*60)
        
        if report["checks"]["gradients_ok"] and not has_nan and not has_inf:
            print("✓ 测试通过 - 数值稳定，可以进行完整训练")
        else:
            print("✗ 测试失败 - 检测到数值问题:")
            if has_nan:
                print("  - 存在 NaN")
            if has_inf:
                print("  - 存在 Inf")
            if not report["checks"]["gradients_ok"]:
                print("  - 梯度异常")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--folder", type=str, default="photos_test")
    parser.add_argument("--labels", type=str, default="photos_test/labels.csv")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for smoke test")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--consistency-weight", type=float, default=0.1, help="Consistency loss weight")
    parser.add_argument("--output-dir", type=str, default="outputs/smoke_test")
    args = parser.parse_args()
    
    smoke_test(args)


if __name__ == "__main__":
    main()
