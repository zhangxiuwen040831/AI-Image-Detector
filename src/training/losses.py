import torch
from torch import nn
from typing import Dict, Optional
from src.models.osd import OrthogonalSubspaceProjector


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class MetricLoss(nn.Module):
    """
    度量学习损失：Triplet Loss
    构造三元组（锚点：真实图像；正例：另一真实图像；负例：生成图像）
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, features, labels):
        # labels: 0 for real, 1 for fake
        # Anchor: Real (0)
        # Positive: Real (0)
        # Negative: Fake (1)
        
        real_mask = (labels == 0)
        fake_mask = (labels == 1)
        
        real_features = features[real_mask]
        fake_features = features[fake_mask]
        
        if real_features.size(0) < 2 or fake_features.size(0) < 1:
            return torch.tensor(0.0, device=features.device)
            
        # Simple mining: random pairing or all valid triplets
        # For efficiency, we can just pick random positive/negative for each anchor
        # Or use hard mining if possible.
        # Here we implement a simple semi-hard mining or just random sampling for simplicity in this step.
        
        # Strategy: For each real image (anchor), pick another real image (positive) and a fake image (negative)
        # To make it efficient, we can shuffle real_features to get positives
        
        # Ensure we have enough real samples to form pairs
        num_real = real_features.size(0)
        num_fake = fake_features.size(0)
        
        # Create indices for shuffling
        perm_real = torch.randperm(num_real, device=features.device)
        positive_features = real_features[perm_real]
        
        # If we just shuffle, some might be the same image (if batch size is small or duplicates exist, but here index-wise it's fine)
        # Actually if i == perm[i], it's the same image. We should avoid that.
        # But for simplicity, let's just rotate by 1
        positive_features = torch.roll(real_features, shifts=1, dims=0)
        
        # For negative, we need to map each real anchor to a fake negative.
        # We can repeat fake features to match real features count
        if num_fake < num_real:
            repeat_count = (num_real // num_fake) + 1
            negative_features = fake_features.repeat(repeat_count, 1)[:num_real]
        else:
            negative_features = fake_features[:num_real]
            
        # Compute loss
        # Anchor: real_features
        # Positive: positive_features (another real)
        # Negative: negative_features (a fake)
        
        loss = self.triplet_loss(real_features, positive_features, negative_features)
        return loss

class DetectionLoss(nn.Module):
    """
    组合检测损失：
    - FocalLoss 用于二分类 (Hard Sample Mining)
    - 可选 OSD 正交性正则
    - 可选 Metric Loss (Triplet)
    """

    def __init__(self, pos_weight: Optional[float] = None, use_focal: bool = True, use_metric: bool = True) -> None:
        super().__init__()
        if use_focal:
            # Alpha 0.25 fits for class imbalance if fake is minority, adjust if needed
            self.cls_loss = FocalLoss(alpha=0.25, gamma=2.0)
        elif pos_weight is not None:
            self.cls_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        else:
            self.cls_loss = nn.BCEWithLogitsLoss()
            
        self.use_metric = use_metric
        if use_metric:
            self.metric_loss = MetricLoss(margin=0.3)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        osd_projector: Optional[OrthogonalSubspaceProjector] = None,
        osd_lambda_orth: float = 0.0,
        metric_lambda: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        labels_float = labels.float()
        loss_cls = self.cls_loss(logits, labels_float)
        loss = loss_cls

        loss_orth = torch.tensor(0.0, device=logits.device)
        if osd_projector is not None and osd_lambda_orth > 0:
            loss_orth = osd_projector.orthogonality_loss() * osd_lambda_orth
            loss = loss + loss_orth
            
        loss_metric = torch.tensor(0.0, device=logits.device)
        if self.use_metric and features is not None:
            loss_metric = self.metric_loss(features, labels) * metric_lambda
            loss = loss + loss_metric

        return {
            "loss": loss,
            "loss_cls": loss_cls.detach(),
            "loss_orth": loss_orth.detach(),
            "loss_metric": loss_metric.detach(),
        }

