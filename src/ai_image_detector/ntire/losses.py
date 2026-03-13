from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class HybridDetectionLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        focal_weight: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        aux_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.aux_weight = aux_weight

    def _focal(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
        alpha_t = self.focal_alpha * labels + (1.0 - self.focal_alpha) * (1.0 - labels)
        focal_term = (1.0 - p_t).pow(self.focal_gamma)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        return (alpha_t * focal_term * bce).mean()

    def forward(self, model_out: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        labels = labels.view(-1, 1).float()
        main_logit = model_out["logit"]
        bce_loss = self.bce(main_logit, labels)
        focal_loss = self._focal(main_logit, labels) if self.focal_weight > 0 else labels.new_zeros(())
        total = self.bce_weight * bce_loss + self.focal_weight * focal_loss
        aux_total = labels.new_zeros(())
        aux_terms = {}
        for key in ("semantic_logit", "freq_logit", "noise_logit"):
            if key in model_out:
                v = self.bce(model_out[key], labels)
                aux_terms[f"{key}_loss"] = v
                aux_total = aux_total + v
        if aux_terms:
            total = total + self.aux_weight * (aux_total / len(aux_terms))
        result: Dict[str, torch.Tensor] = {
            "total_loss": total,
            "main_bce_loss": bce_loss,
            "focal_loss": focal_loss,
        }
        result.update(aux_terms)
        return result
