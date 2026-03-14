
import random
import torch
from PIL import Image, ImageFilter
import numpy as np
from torchvision import transforms as T

class RandomBlur:
    def __init__(self, p=0.5, radius_range=(0.1, 2.0)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

class RandomNoise:
    def __init__(self, p=0.5, mean=0.0, std=0.1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if random.random() < self.p:
            img_np = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(self.mean, self.std, img_np.shape)
            img_np = img_np + noise
            img_np = np.clip(img_np, 0, 1)
            return Image.fromarray((img_np * 255).astype(np.uint8))
        return img

class RandomJPEGCompression:
    def __init__(self, quality_range=(60, 100), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            import io
            output = io.BytesIO()
            quality = random.randint(*self.quality_range)
            img.save(output, format="JPEG", quality=quality)
            output.seek(0)
            return Image.open(output)
        return img
