from __future__ import annotations

import io
import random
from typing import Any, Tuple, Dict, Optional

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFilter


REAL_PROFILE_MIXES = {
    "standard_v8": (0.40, 0.40, 0.20),
    "anti_shortcut_v8": (0.60, 0.30, 0.10),
    "standard_v81": (0.40, 0.40, 0.20),
    "anti_shortcut_v81": (0.70, 0.20, 0.10),
    "standard_v9": (0.40, 0.40, 0.20),
    "hard_real_v9": (0.70, 0.20, 0.10),
    "standard_v10": (0.40, 0.40, 0.20),
    "hard_real_v10": (0.70, 0.20, 0.10),
    "anchor_hard_real_v101": (0.80, 0.15, 0.05),
}

AIGC_PROFILE_MIXES = {
    "standard_v10": (0.50, 0.30, 0.20),
    "fragile_v10": (0.60, 0.25, 0.15),
    "fragile_v101": (0.70, 0.20, 0.10),
}


def get_real_profile_probabilities(profile: str = "standard_v8") -> Tuple[float, float, float]:
    values = REAL_PROFILE_MIXES.get(profile, REAL_PROFILE_MIXES["standard_v8"])
    probs = np.asarray(values, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 1e-12:
        return 1.0, 0.0, 0.0
    probs = probs / total
    return float(probs[0]), float(probs[1]), float(probs[2])


def get_aigc_profile_probabilities(profile: str = "standard_v10") -> Tuple[float, float, float]:
    values = AIGC_PROFILE_MIXES.get(profile, AIGC_PROFILE_MIXES["standard_v10"])
    probs = np.asarray(values, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 1e-12:
        return 1.0, 0.0, 0.0
    probs = probs / total
    return float(probs[0]), float(probs[1]), float(probs[2])


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


class RandomDownUpResize(A.ImageOnlyTransform):
    def __init__(self, min_scale: float = 0.45, max_scale: float = 0.85, p: float = 0.35) -> None:
        super().__init__(p=p)
        self.min_scale = max(0.1, float(min_scale))
        self.max_scale = max(self.min_scale, float(max_scale))

    def get_params(self):
        return {"scale": random.uniform(self.min_scale, self.max_scale)}

    def apply(self, img: np.ndarray, scale: float = 0.6, **params):
        h, w = img.shape[:2]
        small_h = max(16, int(h * scale))
        small_w = max(16, int(w * scale))
        pil = Image.fromarray(img)
        down = pil.resize((small_w, small_h), Image.Resampling.BILINEAR)
        up = down.resize((w, h), Image.Resampling.BICUBIC)
        return np.array(up)


class RandomFormatRoundTrip(A.ImageOnlyTransform):
    def __init__(self, p: float = 0.35) -> None:
        super().__init__(p=p)

    def get_params(self):
        return {
            "fmt": random.choice(["PNG", "WEBP"]),
            "quality": random.randint(25, 85),
        }

    def apply(self, img: np.ndarray, fmt: str = "PNG", quality: int = 80, **params):
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        save_kwargs = {}
        if fmt == "WEBP":
            save_kwargs["quality"] = int(quality)
            save_kwargs["method"] = 6
        pil.save(buf, format=fmt, **save_kwargs)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        return np.array(out)


def build_train_transform(
    image_size: int = 224,
    jpeg_aligned_crop_p: float = 0.3,
    enable_grayscale: bool = True,
    enable_defocus: bool = True,
    enable_webp: bool = True,
    strong_compression_p: float = 0.65,
    down_up_resize_p: float = 0.35,
    roundtrip_p: float = 0.35,
) -> Any:
    aligned_p = max(0.0, min(float(jpeg_aligned_crop_p), 1.0))
    rrc_p = 1.0 - aligned_p
    blur_ops = [
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=(3, 5), p=1.0),
    ]
    if enable_defocus and hasattr(A, "Defocus"):
        blur_ops.append(A.Defocus(radius=(3, 6), alias_blur=(0.1, 0.3), p=1.0))
    compression_ops = [A.ImageCompression(quality_range=(18, 90), compression_type="jpeg", p=1.0)]
    if enable_webp:
        compression_ops.append(A.ImageCompression(quality_range=(18, 90), compression_type="webp", p=1.0))
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
            A.OneOf(compression_ops, p=max(0.0, min(float(strong_compression_p), 1.0))),
            RandomDownUpResize(min_scale=0.45, max_scale=0.85, p=max(0.0, min(float(down_up_resize_p), 1.0))),
            RandomFormatRoundTrip(p=max(0.0, min(float(roundtrip_p), 1.0))),
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


# =========================
# Robust/hard augmentations
# =========================

class RandomJPEGChain(A.ImageOnlyTransform):
    """
    Simulate multi-platform re-encoding: JPEG -> WebP -> JPEG with random qualities.
    """
    def __init__(self, p: float = 0.35, jpeg_q_range=(25, 85), webp_q_range=(35, 90)) -> None:
        super().__init__(p=p)
        self.jpeg_q_range = tuple(int(x) for x in jpeg_q_range)
        self.webp_q_range = tuple(int(x) for x in webp_q_range)

    def apply(self, img: np.ndarray, **params):
        pil = Image.fromarray(img)
        # JPEG
        buf1 = io.BytesIO()
        q1 = random.randint(*self.jpeg_q_range)
        pil.save(buf1, format="JPEG", quality=q1, subsampling=random.choice([0, 1, 2]))
        buf1.seek(0)
        pil1 = Image.open(buf1).convert("RGB")
        # WebP
        buf2 = io.BytesIO()
        q2 = random.randint(*self.webp_q_range)
        pil1.save(buf2, format="WEBP", quality=q2, method=6)
        buf2.seek(0)
        pil2 = Image.open(buf2).convert("RGB")
        # Back to JPEG
        buf3 = io.BytesIO()
        q3 = random.randint(*self.jpeg_q_range)
        pil2.save(buf3, format="JPEG", quality=q3, subsampling=random.choice([0, 1, 2]))
        buf3.seek(0)
        out = Image.open(buf3).convert("RGB")
        return np.array(out)


class RandomUnsharpMask(A.ImageOnlyTransform):
    def __init__(
        self,
        p: float = 0.5,
        radius_range: Tuple[int, int] = (1, 2),
        percent_range: Tuple[int, int] = (80, 180),
        threshold_range: Tuple[int, int] = (1, 4),
    ) -> None:
        super().__init__(p=p)
        self.radius_range = radius_range
        self.percent_range = percent_range
        self.threshold_range = threshold_range

    def apply(self, img: np.ndarray, **params):
        pil = Image.fromarray(img)
        sharpened = pil.filter(
            ImageFilter.UnsharpMask(
                radius=random.randint(*self.radius_range),
                percent=random.randint(*self.percent_range),
                threshold=random.randint(*self.threshold_range),
            )
        )
        return np.array(sharpened)


class ScreenshotLike(A.ImageOnlyTransform):
    """
    Approximate screenshot/second-shot artifacts:
    - Downscale + nearest upsample (introduce blockiness)
    - Mild gaussian noise
    - Optional sharpen
    """
    def __init__(self, p: float = 0.35) -> None:
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params):
        h, w = img.shape[:2]
        scale = random.uniform(0.35, 0.7)
        small_h = max(16, int(h * scale))
        small_w = max(16, int(w * scale))
        pil = Image.fromarray(img)
        down = pil.resize((small_w, small_h), Image.Resampling.BILINEAR if random.random() < 0.5 else Image.Resampling.NEAREST)
        up = down.resize((w, h), Image.Resampling.NEAREST if random.random() < 0.5 else Image.Resampling.BILINEAR)
        arr = np.array(up)
        # mild noise
        if random.random() < 0.8:
            std = random.uniform(2.0, 6.0)
            noise = np.random.normal(0.0, std, size=arr.shape).astype(np.float32)
            arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # occasional sharpen
        if random.random() < 0.5:
            arr = np.array(Image.fromarray(arr).filter(ImageFilter.SHARPEN))
        return arr


class PostprocessChainSelector(A.ImageOnlyTransform):
    """
    Select exactly one of (clean/mild/hard) postprocess chains per call.

    Albumentations' OneOf expects Transform objects, not Compose pipelines.
    This wrapper makes chain selection behave like a single ImageOnlyTransform.
    """

    def __init__(
        self,
        clean_chain: Any,
        mild_chain: Any,
        hard_chain: Any,
        p_clean: float = 0.35,
        p_mild: float = 0.40,
        p_hard: float = 0.25,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.clean_chain = clean_chain
        self.mild_chain = mild_chain
        self.hard_chain = hard_chain
        probs = np.asarray([p_clean, p_mild, p_hard], dtype=np.float64)
        probs = np.clip(probs, 0.0, None)
        s = float(probs.sum())
        if s <= 1e-12:
            probs = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            probs = probs / s
        self.p_clean, self.p_mild, self.p_hard = (float(probs[0]), float(probs[1]), float(probs[2]))

    def apply(self, img: np.ndarray, **params):
        r = random.random()
        if r < self.p_clean:
            return self.clean_chain(image=img)["image"]
        if r < self.p_clean + self.p_mild:
            return self.mild_chain(image=img)["image"]
        return self.hard_chain(image=img)["image"]


def build_clean_postprocess_chain(
    image_size: int = 224,
    enable_webp: bool = True,
) -> Any:
    compression_ops = [A.ImageCompression(quality_range=(45, 95), compression_type="jpeg", p=1.0)]
    if enable_webp:
        compression_ops.append(A.ImageCompression(quality_range=(50, 95), compression_type="webp", p=1.0))
    compression_ops.append(A.NoOp(p=1.0))
    return A.Compose(
        [
            A.OneOf(compression_ops, p=0.25),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                    A.NoOp(p=1.0),
                ],
                p=0.15,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.005, 0.02), p=1.0),
                    A.NoOp(p=1.0),
                ],
                p=0.15,
            ),
        ]
    )


def build_mild_postprocess_chain(
    image_size: int = 224,
    enable_webp: bool = True,
) -> Any:
    compression_ops = [A.ImageCompression(quality_range=(25, 90), compression_type="jpeg", p=1.0)]
    if enable_webp:
        compression_ops.append(A.ImageCompression(quality_range=(25, 90), compression_type="webp", p=1.0))
    compression_ops.append(A.NoOp(p=1.0))
    blur_ops = [
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=(3, 5), p=1.0),
        A.NoOp(p=1.0),
    ]
    return A.Compose(
        [
            A.OneOf(compression_ops, p=0.55),
            RandomDownUpResize(min_scale=0.55, max_scale=0.9, p=0.35),
            A.OneOf(blur_ops, p=0.35),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.05, 0.18), p=1.0),
                    A.NoOp(p=1.0),
                ],
                p=0.35,
            ),
            A.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.14, hue=0.04, p=0.35),
        ]
    )


def build_hard_postprocess_chain(
    image_size: int = 224,
    enable_grayscale: bool = True,
    enable_defocus: bool = True,
    enable_webp: bool = True,
    strong_compression_p: float = 0.65,
    down_up_resize_p: float = 0.35,
    roundtrip_p: float = 0.35,
) -> Any:
    """
    A stronger, real-world oriented postprocess chain to better simulate platform pipelines.
    """
    blur_ops = [
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=(3, 7), p=1.0),
    ]
    if enable_defocus and hasattr(A, "Defocus"):
        blur_ops.append(A.Defocus(radius=(3, 7), alias_blur=(0.1, 0.4), p=1.0))
    
    compression_ops = [
        RandomJPEGChain(p=1.0),
        A.ImageCompression(quality_range=(15, 85), compression_type="jpeg", p=1.0),
    ]
    if enable_webp:
        compression_ops.append(A.ImageCompression(quality_range=(30, 90), compression_type="webp", p=1.0))
    
    return A.Compose(
        [
            # NOTE: geometric ops are expected to be applied outside in build_train_transform
            A.OneOf(compression_ops, p=max(0.0, min(float(strong_compression_p), 1.0))),
            RandomDownUpResize(min_scale=0.35, max_scale=0.8, p=max(0.0, min(float(down_up_resize_p), 1.0))),
            ScreenshotLike(p=0.45),
            RandomFormatRoundTrip(p=max(0.0, min(float(roundtrip_p), 1.0))),
            A.OneOf(
                [
                    *blur_ops,
                    A.GaussNoise(std_range=(0.01, 0.1), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.06), intensity=(0.05, 0.25), p=1.0),
                    A.NoOp(p=1.0),
                ],
                p=0.55,
            ),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05, p=0.45),
            A.ToGray(p=0.12 if enable_grayscale else 0.0),
        ]
    )


def build_real_hard_negative_chain(
    image_size: int = 224,
    enable_defocus: bool = True,
    enable_webp: bool = True,
) -> Any:
    blur_ops = [
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=(3, 5), p=1.0),
        A.NoOp(p=1.0),
    ]
    if enable_defocus and hasattr(A, "Defocus"):
        blur_ops.insert(2, A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.3), p=1.0))

    compression_ops = [
        RandomJPEGChain(p=1.0, jpeg_q_range=(30, 95), webp_q_range=(35, 95)),
        A.ImageCompression(quality_range=(30, 95), compression_type="jpeg", p=1.0),
    ]
    if enable_webp:
        compression_ops.append(A.ImageCompression(quality_range=(35, 95), compression_type="webp", p=1.0))
    compression_ops.append(A.NoOp(p=1.0))

    sharpen_ops = [
        A.Sharpen(alpha=(0.05, 0.18), lightness=(0.92, 1.10), p=1.0),
        RandomUnsharpMask(
            p=1.0,
            radius_range=(1, 2),
            percent_range=(70, 140),
            threshold_range=(2, 5),
        ),
        A.NoOp(p=1.0),
    ]

    contrast_ops = [
        A.CLAHE(clip_limit=(1.0, 2.4), tile_grid_size=(8, 8), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.06, contrast_limit=0.22, p=1.0),
        A.NoOp(p=1.0),
    ]

    return A.Compose(
        [
            A.OneOf(compression_ops, p=0.80),
            RandomDownUpResize(min_scale=0.45, max_scale=0.92, p=0.60),
            A.OneOf(
                [
                    ScreenshotLike(p=1.0),
                    A.NoOp(p=1.0),
                ],
                p=0.15,
            ),
            A.OneOf(sharpen_ops, p=0.45),
            A.OneOf(
                [
                    RandomJPEGChain(p=1.0, jpeg_q_range=(35, 88), webp_q_range=(40, 90)),
                    A.ImageCompression(quality_range=(35, 90), compression_type="jpeg", p=1.0),
                    A.NoOp(p=1.0),
                ],
                p=0.18,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.004, 0.025), p=1.0),
                    A.ISONoise(color_shift=(0.004, 0.03), intensity=(0.03, 0.12), p=1.0),
                    A.NoOp(p=1.0),
                ],
                p=0.40,
            ),
            A.OneOf(contrast_ops, p=0.45),
            A.OneOf(blur_ops, p=0.22),
            A.ColorJitter(brightness=0.06, contrast=0.10, saturation=0.08, hue=0.02, p=0.30),
        ]
    )


def build_real_balanced_negative_focus_chain(
    image_size: int = 224,
    enable_defocus: bool = True,
    enable_webp: bool = True,
) -> Any:
    blur_ops = [
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=(3, 5), p=1.0),
        A.NoOp(p=1.0),
    ]
    if enable_defocus and hasattr(A, "Defocus"):
        blur_ops.insert(2, A.Defocus(radius=(2, 4), alias_blur=(0.1, 0.25), p=1.0))

    compression_ops = [
        A.ImageCompression(quality_range=(40, 94), compression_type="jpeg", p=1.0),
    ]
    if enable_webp:
        compression_ops.append(A.ImageCompression(quality_range=(45, 95), compression_type="webp", p=1.0))
    compression_ops.append(A.NoOp(p=1.0))

    sharpen_ops = [
        A.Sharpen(alpha=(0.04, 0.14), lightness=(0.95, 1.08), p=1.0),
        RandomUnsharpMask(
            p=1.0,
            radius_range=(1, 2),
            percent_range=(50, 110),
            threshold_range=(2, 6),
        ),
        A.NoOp(p=1.0),
    ]

    tone_ops = [
        A.CLAHE(clip_limit=(1.0, 2.2), tile_grid_size=(8, 8), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.16, p=1.0),
        A.NoOp(p=1.0),
    ]

    return A.Compose(
        [
            A.OneOf(compression_ops, p=0.60),
            RandomDownUpResize(min_scale=0.55, max_scale=0.94, p=0.40),
            A.OneOf(sharpen_ops, p=0.30),
            A.OneOf(
                [
                    RandomFormatRoundTrip(p=1.0),
                    ScreenshotLike(p=1.0),
                    A.NoOp(p=1.0),
                ],
                p=0.12,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.003, 0.02), p=1.0),
                    A.ISONoise(color_shift=(0.003, 0.02), intensity=(0.02, 0.10), p=1.0),
                    A.NoOp(p=1.0),
                ],
                p=0.28,
            ),
            A.OneOf(tone_ops, p=0.40),
            A.OneOf(blur_ops, p=0.15),
            A.ColorJitter(brightness=0.05, contrast=0.08, saturation=0.06, hue=0.02, p=0.22),
        ]
    )


def build_real_clean_transform(
    image_size: int = 224,
    jpeg_aligned_crop_p: float = 0.3,
    enable_webp: bool = True,
) -> Any:
    aligned_p = max(0.0, min(float(jpeg_aligned_crop_p), 1.0))
    rrc_p = 1.0 - aligned_p
    geom = [
        A.OneOf(
            [
                JPEGAlignedRandomCrop(crop_size=image_size, p=aligned_p),
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.70, 1.0),
                    ratio=(0.85, 1.15),
                    p=rrc_p,
                ),
            ],
            p=1.0,
        ),
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
    ]
    return A.Compose(
        [
            *geom,
            build_clean_postprocess_chain(image_size=image_size, enable_webp=enable_webp),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_real_hard_negative_transform(
    image_size: int = 224,
    jpeg_aligned_crop_p: float = 0.3,
    enable_defocus: bool = True,
    enable_webp: bool = True,
) -> Any:
    aligned_p = max(0.0, min(float(jpeg_aligned_crop_p), 1.0))
    rrc_p = 1.0 - aligned_p
    geom = [
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
    ]
    hard_negative_chain = build_real_hard_negative_chain(
        image_size=image_size,
        enable_defocus=enable_defocus,
        enable_webp=enable_webp,
    )
    return A.Compose(
        [
            *geom,
            hard_negative_chain,
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_real_balanced_negative_focus_transform(
    image_size: int = 224,
    jpeg_aligned_crop_p: float = 0.3,
    enable_defocus: bool = True,
    enable_webp: bool = True,
) -> Any:
    aligned_p = max(0.0, min(float(jpeg_aligned_crop_p), 1.0))
    rrc_p = 1.0 - aligned_p
    geom = [
        A.OneOf(
            [
                JPEGAlignedRandomCrop(crop_size=image_size, p=aligned_p),
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.68, 1.0),
                    ratio=(0.82, 1.18),
                    p=rrc_p,
                ),
            ],
            p=1.0,
        ),
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
    ]
    balanced_chain = build_real_balanced_negative_focus_chain(
        image_size=image_size,
        enable_defocus=enable_defocus,
        enable_webp=enable_webp,
    )
    return A.Compose(
        [
            *geom,
            balanced_chain,
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _build_aigc_profile_transform(
    postprocess_chain: Any,
    image_size: int = 224,
    jpeg_aligned_crop_p: float = 0.3,
    scale: Tuple[float, float] = (0.68, 1.0),
    ratio: Tuple[float, float] = (0.82, 1.18),
) -> Any:
    aligned_p = max(0.0, min(float(jpeg_aligned_crop_p), 1.0))
    rrc_p = 1.0 - aligned_p
    geom = [
        A.OneOf(
            [
                JPEGAlignedRandomCrop(crop_size=image_size, p=aligned_p),
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=scale,
                    ratio=ratio,
                    p=rrc_p,
                ),
            ],
            p=1.0,
        ),
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
    ]
    return A.Compose(
        [
            *geom,
            postprocess_chain,
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_aigc_clean_transform(
    image_size: int = 224,
    jpeg_aligned_crop_p: float = 0.3,
    enable_webp: bool = True,
) -> Any:
    return _build_aigc_profile_transform(
        postprocess_chain=build_clean_postprocess_chain(
            image_size=image_size,
            enable_webp=enable_webp,
        ),
        image_size=image_size,
        jpeg_aligned_crop_p=jpeg_aligned_crop_p,
        scale=(0.72, 1.0),
        ratio=(0.85, 1.15),
    )


def build_aigc_mild_transform(
    image_size: int = 224,
    jpeg_aligned_crop_p: float = 0.3,
    enable_webp: bool = True,
) -> Any:
    return _build_aigc_profile_transform(
        postprocess_chain=build_mild_postprocess_chain(
            image_size=image_size,
            enable_webp=enable_webp,
        ),
        image_size=image_size,
        jpeg_aligned_crop_p=jpeg_aligned_crop_p,
        scale=(0.68, 1.0),
        ratio=(0.82, 1.18),
    )


def build_aigc_hard_transform(
    image_size: int = 224,
    jpeg_aligned_crop_p: float = 0.3,
    enable_grayscale: bool = True,
    enable_defocus: bool = True,
    enable_webp: bool = True,
) -> Any:
    return _build_aigc_profile_transform(
        postprocess_chain=build_hard_postprocess_chain(
            image_size=image_size,
            enable_grayscale=enable_grayscale,
            enable_defocus=enable_defocus,
            enable_webp=enable_webp,
        ),
        image_size=image_size,
        jpeg_aligned_crop_p=jpeg_aligned_crop_p,
        scale=(0.65, 1.0),
        ratio=(0.80, 1.20),
    )


def build_dual_view_transforms(
    image_size: int = 224,
    strength: str = "hard",
) -> Dict[str, Any]:
    """
    Return two randomized views for consistency checking/training without changing model API.
    strength: "train" uses existing build_train_transform; "hard" uses build_hard_postprocess_chain
    """
    if strength == "train":
        t1 = build_train_transform(image_size=image_size, jpeg_aligned_crop_p=0.3)
        t2 = build_train_transform(image_size=image_size, jpeg_aligned_crop_p=0.15)
    else:
        # Use full train-time geometry + chain mixing for dual views
        t1 = build_train_transform(image_size=image_size, chain_mix=True, chain_mix_strength="hard")
        t2 = build_train_transform(image_size=image_size, chain_mix=True, chain_mix_strength="hard")
    return {"view1": t1, "view2": t2}


def build_train_transform(
    image_size: int = 224,
    jpeg_aligned_crop_p: float = 0.3,
    enable_grayscale: bool = True,
    enable_defocus: bool = True,
    enable_webp: bool = True,
    strong_compression_p: float = 0.65,
    down_up_resize_p: float = 0.35,
    roundtrip_p: float = 0.35,
    # New options (backwards compatible)
    chain_mix: bool = True,
    chain_mix_strength: str = "default",
    chain_p_clean: float = 0.30,
    chain_p_mild: float = 0.45,
    chain_p_hard: float = 0.25,
) -> Any:
    """
    Train transform with optional clean/mild/hard postprocess chain mixing.
    The goal is to reduce AIGC false negatives under mild real-world perturbations while
    keeping robustness under hard post-processing.
    """
    aligned_p = max(0.0, min(float(jpeg_aligned_crop_p), 1.0))
    rrc_p = 1.0 - aligned_p
    blur_ops = [
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=(3, 5), p=1.0),
    ]
    if enable_defocus and hasattr(A, "Defocus"):
        blur_ops.append(A.Defocus(radius=(3, 6), alias_blur=(0.1, 0.3), p=1.0))
    compression_ops = [A.ImageCompression(quality_range=(18, 90), compression_type="jpeg", p=1.0)]
    if enable_webp:
        compression_ops.append(A.ImageCompression(quality_range=(18, 90), compression_type="webp", p=1.0))
    compression_ops.append(A.NoOp(p=1.0))

    if chain_mix_strength == "hard":
        # harder distribution: favor mild/hard more for AIGC robustness
        p_clean, p_mild, p_hard = 0.15, 0.50, 0.35
    elif chain_mix_strength == "aigc_focus":
        # Focus on mild perturbations to reduce AIGC false negatives
        # V3: More clean, less mild to preserve AIGC features
        p_clean, p_mild, p_hard = 0.50, 0.35, 0.15
    elif chain_mix_strength == "aigc_conservative":
        # Very conservative for previously missed AIGC samples
        # V3: Maximum preservation of AIGC features
        p_clean, p_mild, p_hard = 0.65, 0.25, 0.10
    else:
        p_clean, p_mild, p_hard = float(chain_p_clean), float(chain_p_mild), float(chain_p_hard)

    # Build postprocess chains (no normalize/to-tensor inside; we do it once at end)
    clean_chain = build_clean_postprocess_chain(image_size=image_size, enable_webp=enable_webp)
    mild_chain = build_mild_postprocess_chain(image_size=image_size, enable_webp=enable_webp)
    hard_chain = build_hard_postprocess_chain(
        image_size=image_size,
        enable_grayscale=enable_grayscale,
        enable_defocus=enable_defocus,
        enable_webp=enable_webp,
        strong_compression_p=strong_compression_p,
        down_up_resize_p=down_up_resize_p,
        roundtrip_p=roundtrip_p,
    )

    # Base geometry (shared)
    geom = [
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
    ]

    # If chain_mix disabled, fall back to original single-chain logic
    if not chain_mix:
        post = [
            A.OneOf(compression_ops, p=max(0.0, min(float(strong_compression_p), 1.0))),
            RandomDownUpResize(min_scale=0.45, max_scale=0.85, p=max(0.0, min(float(down_up_resize_p), 1.0))),
            RandomFormatRoundTrip(p=max(0.0, min(float(roundtrip_p), 1.0))),
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
        ]
    else:
        # Mix clean/mild/hard chains to improve stability under mild perturbations
        post = [
            PostprocessChainSelector(
                clean_chain=clean_chain,
                mild_chain=mild_chain,
                hard_chain=hard_chain,
                p_clean=p_clean,
                p_mild=p_mild,
                p_hard=p_hard,
                p=1.0,
            )
        ]

    return A.Compose(
        [
            *geom,
            *post,
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
