import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ImageDataset, build_val_transforms
from src.models import MultiBranchDetector
from src.utils import load_config, setup_logger, save_debug_visualization


def main() -> None:
    cfg = load_config(
        str(ROOT / "configs" / "dev_cpu.yaml"),
        str(ROOT / "configs" / "base_config.yaml"),
    )
    logger = setup_logger("debug_cpu")

    device = torch.device("cpu")
    logger.info("Running CPU debug mode")

    dataset = ImageDataset(
        root_dir=cfg["data"]["train_dir"],
        transform=build_val_transforms(int(cfg["data"]["image_size"])),
    )

    real_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0][:5]
    fake_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1][:5]
    selected = real_indices + fake_indices
    if len(selected) < 10:
        raise RuntimeError("debug_cpu.py requires at least 5 real and 5 fake images")

    subset = Subset(dataset, selected)
    loader = DataLoader(subset, batch_size=2, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    images = images.to(device)

    model = MultiBranchDetector(
        rgb_backbone="resnet18",
        rgb_pretrained=False,
        noise_pretrained=False,
        freq_pretrained=False,
        fused_dim=int(cfg["model"]["fused_dim"]),
        classifier_hidden_dim=int(cfg["model"]["classifier_hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.eval()

    with torch.no_grad():
        out = model(images)
        srm = model.noise_branch.extract_residual(images)
        fft = model.freq_branch.extract_frequency_map(images)

    logger.info(f"Input shape: {tuple(images.shape)}")
    logger.info(f"RGB feature shape: {tuple(out['rgb_feat'].shape)}")
    logger.info(f"Noise feature shape: {tuple(out['noise_feat'].shape)}")
    logger.info(f"Freq feature shape: {tuple(out['freq_feat'].shape)}")
    logger.info(f"Fused feature shape: {tuple(out['fused_feat'].shape)}")
    logger.info(f"Logit shape: {tuple(out['logit'].shape)}")
    logger.info(f"Labels batch: {labels.tolist()}")

    save_debug_visualization(
        original=images[0:1],
        srm_output=srm[0:1],
        fft_output=fft[0:1],
        save_path=str(ROOT / "logs" / "debug_output.png"),
    )
    logger.info("Saved visualization to logs/debug_output.png")


if __name__ == "__main__":
    main()
