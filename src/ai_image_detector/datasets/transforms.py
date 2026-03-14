from typing import Tuple
import io
import random
from PIL import Image

from torchvision import transforms as T


from .artifact_transforms import RandomBlur, RandomNoise, RandomJPEGCompression

# CLIP Normalization (OpenAI)
IMAGENET_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


def build_real_transforms(image_size: int = 224) -> T.Compose:
    """
    真实图像增强：更强的增强策略，包括 RandAugment
    """
    return T.Compose([
        T.Resize(int(image_size * 1.1)),
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandAugment(num_ops=2, magnitude=9), # 强增强
        RandomJPEGCompression(p=0.3),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def build_fake_transforms(image_size: int = 224) -> T.Compose:
    """
    伪造图像增强：
    - 模拟生成器伪影（过度平滑、噪声残留）
    - 保护高频痕迹
    """
    return T.Compose([
        T.Resize(int(image_size * 1.1)),
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)), # 较小的裁剪范围
        T.RandomHorizontalFlip(p=0.5),
        
        # 模拟生成器伪影
        RandomBlur(p=0.3, radius_range=(0.1, 1.0)), # 模拟过度平滑
        RandomNoise(p=0.2, mean=0.0, std=0.05), # 模拟噪声残留
        
        # 模拟网络传输
        RandomJPEGCompression(p=0.2), 
        
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def build_dual_transforms(image_size: int = 224) -> dict:
    return {
        0: build_real_transforms(image_size), # Real
        1: build_fake_transforms(image_size), # Fake
        "default": build_val_transforms(image_size) # Fallback
    }

def build_train_transforms(image_size: int = 224) -> T.Compose:
    """
    (Legacy) 构建训练阶段的数据增强流水线
    """
    return T.Compose(
        [
            T.Resize(int(image_size * 1.1)),
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            # [新增] 增强鲁棒性：高斯模糊和 JPEG 压缩
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            RandomJPEGCompression(p=0.3),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_val_transforms(image_size: int = 224) -> T.Compose:
    """
    构建验证/测试阶段的数据预处理：
    - Resize + CenterCrop
    - ToTensor
    - ImageNet 归一化
    """
    return T.Compose(
        [
            T.Resize(int(image_size * 1.1)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_transforms(split: str, image_size: int = 224) -> T.Compose:
    """
    根据数据划分名称返回合适的 transforms。
    """
    split = split.lower()
    if split in {"train", "training"}:
        return build_train_transforms(image_size)
    return build_val_transforms(image_size)

