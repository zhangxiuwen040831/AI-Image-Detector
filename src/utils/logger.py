import logging
from pathlib import Path

import torch


def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_branch_feature_norms(
    logger: logging.Logger,
    rgb_feat: torch.Tensor,
    noise_feat: torch.Tensor,
    freq_feat: torch.Tensor,
    fused_feat: torch.Tensor,
    epoch: int,
) -> None:
    rgb_norm = rgb_feat.detach().norm(dim=1).mean().item()
    noise_norm = noise_feat.detach().norm(dim=1).mean().item()
    freq_norm = freq_feat.detach().norm(dim=1).mean().item()
    fused_norm = fused_feat.detach().norm(dim=1).mean().item()
    logger.info(
        f"Epoch {epoch} | feature_norms rgb={rgb_norm:.4f} noise={noise_norm:.4f} freq={freq_norm:.4f} fused={fused_norm:.4f}"
    )


def log_gradient_norm(logger: logging.Logger, grad_norm: float, epoch: int) -> None:
    logger.info(f"Epoch {epoch} | gradient_norm={grad_norm:.4f}")


def log_learning_rate(logger: logging.Logger, lr: float, epoch: int) -> None:
    logger.info(f"Epoch {epoch} | learning_rate={lr:.8f}")
