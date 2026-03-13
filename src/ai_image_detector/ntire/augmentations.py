from __future__ import annotations

import random
from typing import Any

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


class JPEGAlignedRandomCrop(A.ImageOnlyTransform):
    def __init__(self, crop_size: int, always_apply: bool = False, p: float = 0.3) -> None:
        super().__init__(p=p)
        self.crop_size = crop_size

    def get_params_dependent_on_data(self, params, data):
        image = data["image"]
        h, w = image.shape[:2]
        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)
        max_y = max(h - crop_h, 0)
        max_x = max(w - crop_w, 0)
        if max_y == 0:
            y0 = 0
        else:
            y0 = random.randint(0, max_y // 8) * 8
            y0 = min(y0, max_y)
        if max_x == 0:
            x0 = 0
        else:
            x0 = random.randint(0, max_x // 8) * 8
            x0 = min(x0, max_x)
        return {"x0": x0, "y0": y0, "crop_h": crop_h, "crop_w": crop_w}

    def apply(self, img: np.ndarray, x0: int = 0, y0: int = 0, crop_h: int = 224, crop_w: int = 224, **params):
        return img[y0 : y0 + crop_h, x0 : x0 + crop_w]


def build_train_transform(
    image_size: int = 224,
    jpeg_aligned_crop_p: float = 0.3,
    enable_grayscale: bool = True,
    enable_defocus: bool = True,
    enable_webp: bool = True,
) -> Any:
    aligned_p = max(0.0, min(float(jpeg_aligned_crop_p), 1.0))
    rrc_p = 1.0 - aligned_p
    blur_ops = [
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=(3, 5), p=1.0),
    ]
    if enable_defocus and hasattr(A, "Defocus"):
        blur_ops.append(A.Defocus(radius=(3, 6), alias_blur=(0.1, 0.3), p=1.0))
    compression_ops = [A.ImageCompression(quality_range=(35, 95), compression_type="jpeg", p=1.0)]
    if enable_webp:
        compression_ops.append(A.ImageCompression(quality_range=(35, 95), compression_type="webp", p=1.0))
    compression_ops.append(A.NoOp(p=1.0))
    return A.Compose(
        [
            A.OneOf(
                [
                    JPEGAlignedRandomCrop(crop_size=image_size, p=aligned_p),
                    A.RandomResizedCrop(
                        size=(image_size, image_size),
                        scale=(0.65, 1.0),
                        ratio=(0.8, 1.2),
                        p=rrc_p,
                    ),
                ],
                p=1.0,
            ),
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.OneOf(compression_ops, p=0.65),
            A.OneOf(blur_ops + [A.NoOp(p=1.0)], p=0.45),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.01, 0.08), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.05, 0.2), p=1.0),
                    A.NoOp(p=1.0),
                ],
                p=0.45,
            ),
            A.ColorJitter(
                brightness=0.12,
                contrast=0.12,
                saturation=0.1,
                hue=0.04,
                p=0.4,
            ),
            A.ToGray(p=0.1 if enable_grayscale else 0.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_eval_transform(image_size: int = 224) -> Any:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=(0, 0, 0),
            ),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
