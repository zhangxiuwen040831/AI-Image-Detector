from __future__ import annotations

import json
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire.augmentations import get_aigc_profile_probabilities, get_real_profile_probabilities  # noqa: E402
from ai_image_detector.ntire.dataset import (  # noqa: E402
    BufferedTransformDataset,
    CurriculumBatchSampler,
    compute_base_hard_real_score,
    compute_fragile_aigc_score,
)
from ai_image_detector.ntire.model_v10 import V10CompetitionResetModel  # noqa: E402
from ai_image_detector.ntire.trainer_v10 import V10Trainer  # noqa: E402


@dataclass
class DummyRecord:
    image_path: Path
    label: int
    metadata: Dict[str, object]


class DummyBaseDataset:
    def __init__(self, records: List[DummyRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)


class TaggedTransform:
    def __init__(self, tag_value: float) -> None:
        self.tag_value = float(tag_value)

    def __call__(self, image):
        return {"image": torch.full((3, 8, 8), self.tag_value, dtype=torch.float32)}


def build_trainer(save_dir: Path) -> V10Trainer:
    model = V10CompetitionResetModel(
        backbone_name="resnet18",
        pretrained_backbone=False,
        semantic_trainable_layers=0,
        image_size=64,
        frequency_dim=32,
        noise_dim=32,
        fused_dim=64,
        head_hidden_dim=32,
        dropout=0.1,
        enable_noise_expert=True,
    )
    return V10Trainer(
        model=model,
        device=torch.device("cpu"),
        save_dir=save_dir,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        checkpoint_interval=1,
    )


def check_forward_modes() -> Dict[str, object]:
    model = V10CompetitionResetModel(
        backbone_name="resnet18",
        pretrained_backbone=False,
        semantic_trainable_layers=0,
        image_size=64,
        frequency_dim=32,
        noise_dim=32,
        fused_dim=64,
        head_hidden_dim=32,
        dropout=0.1,
        enable_noise_expert=True,
    )
    x = torch.randn(2, 3, 64, 64)
    model.set_inference_mode("base_only")
    out_base = model(x)
    model.set_inference_mode("hybrid_optional")
    out_hybrid = model(x)
    assert torch.allclose(out_base["logit"], out_base["base_logit"]), "base_only must expose base_logit"
    assert out_hybrid["logit"].shape == out_base["logit"].shape, "hybrid logit shape mismatch"
    assert out_hybrid["fusion_weights"].shape[-1] == 3, "fusion weights must expose semantic/frequency/noise view"
    assert torch.isfinite(out_hybrid["alpha"]).all(), "alpha must stay finite"
    return {
        "base_logit_mean": float(out_base["logit"].mean().item()),
        "hybrid_logit_mean": float(out_hybrid["logit"].mean().item()),
    }


def check_phase_freezing() -> Dict[str, object]:
    trainer = build_trainer(PROJECT_ROOT / "outputs" / "tmp_smoke_v10_freeze")
    phase1 = trainer.set_phase("phase1_warmup", semantic_trainable_layers=0)
    model = trainer._unwrap_model()
    assert model.inference_mode == "base_only", "phase1 should stay base_only"
    assert all(not param.requires_grad for param in model.noise_branch.parameters()), "noise branch should freeze in phase1"
    phase2 = trainer.set_phase("phase2_curriculum", semantic_trainable_layers=2)
    semantic_trainable = [name for name, param in model.semantic_branch.encoder.named_parameters() if param.requires_grad]
    assert semantic_trainable, "phase2 should unfreeze semantic backbone tail"
    phase4 = trainer.set_phase("phase4_final_polish", semantic_trainable_layers=2)
    assert model.inference_mode == "base_only", "phase4 should stay base_only"
    phase3 = trainer.set_phase("phase3_competition", semantic_trainable_layers=2)
    assert model.inference_mode == "hybrid_optional", "phase3 should enable optional hybrid inference"
    assert any(param.requires_grad for param in model.noise_controller.parameters()), "noise controller should train in phase3"
    return {
        "phase1_trainable": int(phase1["trainable_param_count"]),
        "phase2_trainable": int(phase2["trainable_param_count"]),
        "phase4_trainable": int(phase4["trainable_param_count"]),
        "phase3_trainable": int(phase3["trainable_param_count"]),
        "semantic_trainable_preview": semantic_trainable[:10],
    }


def check_scores_and_buffers() -> Dict[str, object]:
    base_scores = compute_base_hard_real_score(
        base_logit=torch.tensor([1.0, -0.5, 0.3]),
        semantic_logit=torch.tensor([0.1, -0.3, 0.2]),
        frequency_logit=torch.tensor([0.7, -0.4, 0.8]),
    )
    fragile_scores = compute_fragile_aigc_score(
        base_logit=torch.tensor([-0.1, 2.0, 0.2]),
        semantic_logit=torch.tensor([0.1, 1.5, 0.3]),
        frequency_logit=torch.tensor([0.2, 1.2, 0.4]),
        max_probability=0.85,
    )
    assert base_scores[0] > base_scores[1], "hard-real score should prefer false-positive-like samples"
    assert fragile_scores[0] > fragile_scores[1], "fragile AIGC score should favor uncertain positives"
    sampler = CurriculumBatchSampler(
        primary_indices=list(range(12)),
        hard_real_indices=[8, 9],
        anchor_hard_real_indices=[6, 7],
        fragile_aigc_indices=[10, 11],
        batch_size=6,
        hard_real_ratio=0.17,
        anchor_hard_real_ratio=0.17,
        fragile_aigc_ratio=0.17,
        seed=13,
    )
    batch = next(iter(sampler))
    assert any(idx in {8, 9} for idx in batch), "curriculum sampler should inject hard real"
    assert any(idx in {6, 7} for idx in batch), "curriculum sampler should inject anchor hard real"
    assert any(idx in {10, 11} for idx in batch), "curriculum sampler should inject fragile AIGC"
    return {
        "hard_real_scores": [float(x) for x in base_scores.tolist()],
        "fragile_aigc_scores": [float(x) for x in fragile_scores.tolist()],
        "sampler_first_batch": batch,
    }


def check_routing_profiles() -> Dict[str, object]:
    real_probs = get_real_profile_probabilities("standard_v10")
    hard_real_probs = get_real_profile_probabilities("hard_real_v10")
    anchor_real_probs = get_real_profile_probabilities("anchor_hard_real_v101")
    aigc_probs = get_aigc_profile_probabilities("standard_v10")
    fragile_probs = get_aigc_profile_probabilities("fragile_v101")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        image_path = tmp / "sample.png"
        Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8)).save(image_path)
        dataset = DummyBaseDataset(
            [
                DummyRecord(image_path=image_path, label=0, metadata={"image_name": "real_standard.png"}),
                DummyRecord(image_path=image_path, label=0, metadata={"image_name": "real_hard.png"}),
                DummyRecord(image_path=image_path, label=0, metadata={"image_name": "real_anchor.png"}),
                DummyRecord(image_path=image_path, label=1, metadata={"image_name": "aigc_standard.png"}),
                DummyRecord(image_path=image_path, label=1, metadata={"image_name": "aigc_fragile.png"}),
            ]
        )
        buffered = BufferedTransformDataset(
            base_dataset=dataset,
            transform=TaggedTransform(0.0),
            real_clean_transform=TaggedTransform(1.0),
            real_mild_transform=TaggedTransform(2.0),
            real_hard_transform=TaggedTransform(3.0),
            real_clean_prob=real_probs[0],
            real_mild_prob=real_probs[1],
            real_hard_prob=real_probs[2],
            hard_real_indices={1},
            hard_real_clean_prob=hard_real_probs[0],
            hard_real_mild_prob=hard_real_probs[1],
            hard_real_hard_prob=hard_real_probs[2],
            anchor_hard_real_indices={2},
            anchor_hard_real_clean_prob=anchor_real_probs[0],
            anchor_hard_real_mild_prob=anchor_real_probs[1],
            anchor_hard_real_hard_prob=anchor_real_probs[2],
            aigc_clean_transform=TaggedTransform(4.0),
            aigc_mild_transform=TaggedTransform(5.0),
            aigc_hard_transform=TaggedTransform(6.0),
            aigc_clean_prob=aigc_probs[0],
            aigc_mild_prob=aigc_probs[1],
            aigc_hard_prob=aigc_probs[2],
            fragile_aigc_indices={4},
            fragile_aigc_clean_prob=fragile_probs[0],
            fragile_aigc_mild_prob=fragile_probs[1],
            fragile_aigc_hard_prob=fragile_probs[2],
        )
        random.seed(123)
        counts = {
            "real_standard": {"clean": 0, "mild": 0, "hard": 0},
            "real_hard": {"clean": 0, "mild": 0, "hard": 0},
            "real_anchor": {"clean": 0, "mild": 0, "hard": 0},
            "aigc_standard": {"clean": 0, "mild": 0, "hard": 0},
            "aigc_fragile": {"clean": 0, "mild": 0, "hard": 0},
        }
        for _ in range(300):
            _, _, meta_real_standard = buffered[0]
            _, _, meta_real_hard = buffered[1]
            _, _, meta_real_anchor = buffered[2]
            _, _, meta_aigc_standard = buffered[3]
            _, _, meta_aigc_fragile = buffered[4]
            counts["real_standard"][str(meta_real_standard["real_transform_profile"])] += 1
            counts["real_hard"][str(meta_real_hard["real_transform_profile"])] += 1
            counts["real_anchor"][str(meta_real_anchor["real_transform_profile"])] += 1
            counts["aigc_standard"][str(meta_aigc_standard["aigc_transform_profile"])] += 1
            counts["aigc_fragile"][str(meta_aigc_fragile["aigc_transform_profile"])] += 1
        assert counts["real_hard"]["clean"] > counts["real_standard"]["clean"], "hard real should favor clean profile"
        assert counts["real_anchor"]["clean"] > counts["real_hard"]["clean"], "anchor hard real should favor cleaner profile"
        assert counts["aigc_fragile"]["clean"] > counts["aigc_standard"]["clean"], "fragile AIGC should favor clean profile"
    return {
        "routing_counts": counts,
        "real_probs": real_probs,
        "hard_real_probs": hard_real_probs,
        "anchor_real_probs": anchor_real_probs,
        "aigc_probs": aigc_probs,
        "fragile_probs": fragile_probs,
    }


def check_backward() -> Dict[str, object]:
    trainer = build_trainer(PROJECT_ROOT / "outputs" / "tmp_smoke_v10_backward")
    trainer.set_phase("phase2_curriculum", semantic_trainable_layers=2)
    images = torch.randn(4, 3, 64, 64)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    metadata = {
        "hard_real_buffer_hit": [True, False, True, False],
        "anchor_hard_real_buffer_hit": [True, False, False, False],
        "fragile_aigc_buffer_hit": [False, True, False, True],
    }
    trainer.optimizer.zero_grad(set_to_none=True)
    out, loss_dict = trainer._forward_loss(images=images, labels=labels, metadata=metadata)
    loss = loss_dict["total_loss"]
    loss.backward()
    grad_norm = 0.0
    for param in trainer.model.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.norm().item())
    assert grad_norm > 0.0, "V10 loss must backpropagate"
    assert torch.isfinite(out["logit"]).all(), "forward logits must stay finite"
    assert torch.isfinite(loss.detach()).all(), "loss must stay finite"
    return {
        "loss": float(loss.detach().item()),
        "grad_norm": grad_norm,
        "hard_real_margin_loss": float(loss_dict["hard_real_margin_loss"].detach().item()),
        "anchor_real_margin_loss": float(loss_dict["anchor_real_margin_loss"].detach().item()),
        "prototype_margin_loss": float(loss_dict["prototype_margin_loss"].detach().item()),
        "fragile_aigc_support_loss": float(loss_dict["fragile_aigc_support_loss"].detach().item()),
    }


def main() -> int:
    summary = {
        "forward_modes": check_forward_modes(),
        "phase_freezing": check_phase_freezing(),
        "scores_and_buffers": check_scores_and_buffers(),
        "routing_profiles": check_routing_profiles(),
        "backward": check_backward(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
