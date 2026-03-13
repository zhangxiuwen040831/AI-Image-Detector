from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _to_numpy_chw(x: torch.Tensor):
    x = x.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    return x


def save_debug_visualization(
    original: torch.Tensor,
    srm_output: torch.Tensor,
    fft_output: torch.Tensor,
    save_path: str,
) -> None:
    original_np = _to_numpy_chw(original).permute(1, 2, 0).clamp(0, 1).numpy()
    srm_np = _to_numpy_chw(srm_output).mean(dim=0).numpy()
    fft_np = _to_numpy_chw(fft_output).mean(dim=0).numpy()

    def to_uint8_gray(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        arr = arr - arr.min()
        denom = max(arr.max(), 1e-6)
        arr = arr / denom
        return (arr * 255.0).astype(np.uint8)

    original_u8 = (np.clip(original_np, 0, 1) * 255.0).astype(np.uint8)
    srm_u8 = to_uint8_gray(srm_np)
    fft_u8 = to_uint8_gray(fft_np)
    srm_rgb = np.stack([srm_u8, srm_u8, srm_u8], axis=2)
    fft_rgb = np.stack([fft_u8, fft_u8, fft_u8], axis=2)
    panel = np.concatenate([original_u8, srm_rgb, fft_rgb], axis=1)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(save_path)
