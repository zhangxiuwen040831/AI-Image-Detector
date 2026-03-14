import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sampler import SourceBalancedSampler
from src.models.detector import MultiBranchDetector
from src.training.losses import DetectionLoss


class DummyMixedDataset:
    def __init__(self) -> None:
        self.records = []
        for i in range(200):
            self.records.append((f"real_{i}.png", 0, "artifact", "real"))
        for i in range(300):
            self.records.append((f"artifact_fake_{i}.png", 1, "artifact", "fake"))
        for i in range(100):
            self.records.append((f"cifake_fake_{i}.png", 1, "cifake", "fake"))

    def __len__(self) -> int:
        return len(self.records)


def test_forward_and_loss(device: torch.device):
    model = MultiBranchDetector(
        rgb_pretrained=False,
        noise_pretrained=False,
        freq_pretrained=False,
        fused_dim=512,
        classifier_hidden_dim=256,
        dropout=0.1,
    ).to(device)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224, device=device)
        labels = torch.tensor([1.0], device=device)
        out = model(x)
        loss_fn = DetectionLoss(
            bce_weight=0.7,
            focal_weight=0.3,
            lambda_rgb=0.2,
            lambda_freq=0.2,
            lambda_spatial=0.2,
        )
        losses = loss_fn(out, labels)
    total_loss = float(losses["total_loss"].item())
    fusion_shape = tuple(out["fused_feat"].shape)
    gate_shape = tuple(out["fusion_weights"].shape)
    fusion_params = sum(p.numel() for p in model.fusion_module.parameters())
    if not math.isfinite(total_loss):
        raise RuntimeError("loss is not finite")
    return {
        "total_loss": total_loss,
        "main_loss": float(losses["main_loss"].item()),
        "rgb_loss": float(losses["rgb_loss"].item()),
        "freq_loss": float(losses["freq_loss"].item()),
        "spatial_loss": float(losses["spatial_loss"].item()),
        "fusion_shape": fusion_shape,
        "gate_shape": gate_shape,
        "fusion_params": fusion_params,
    }


def test_sampler():
    dataset = DummyMixedDataset()
    sampler = SourceBalancedSampler(
        dataset=dataset,
        artifact_cifake_ratio=(3, 1),
        real_fake_ratio=(1, 1),
        num_samples=80,
    )
    indices = list(iter(sampler))
    composition = sampler.inspect_batch_composition(indices)
    return composition


def main():
    device = torch.device("cpu")
    test1_test2 = test_forward_and_loss(device)
    test3 = test_sampler()
    print("test1_loss", test1_test2["total_loss"])
    print("test1_detail", test1_test2["main_loss"], test1_test2["rgb_loss"], test1_test2["freq_loss"], test1_test2["spatial_loss"])
    print("test2_fusion_shape", test1_test2["fusion_shape"])
    print("test2_gate_shape", test1_test2["gate_shape"])
    print("test2_fusion_params", test1_test2["fusion_params"])
    print("test3_real_fake_ratio", test3["real_ratio"], test3["fake_ratio"])
    print("test3_source_ratio", test3["artifact_ratio_in_fake"], test3["cifake_ratio_in_fake"])


if __name__ == "__main__":
    main()
