
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2
from typing import Tuple
from .artifact_transforms import RandomJPEGCompression

IMAGENET_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

def get_transforms(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def build_train_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(int(image_size * 1.1)),
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
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
    return T.Compose(
        [
            T.Resize(int(image_size * 1.1)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

def get_srm_filter_weights():
    # Standard SRM filters (simplified for visualization)
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    
    # [Out, In, H, W] -> [3, 1, 5, 5]
    weights = torch.tensor([filter1, filter2, filter3], dtype=torch.float32).unsqueeze(1) / 12.0
    return weights

def apply_srm_filter(image_tensor):
    # image_tensor: [1, 3, H, W] (normalized) or [3, H, W]
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Convert to grayscale: 0.299R + 0.587G + 0.114B
    gray = 0.299 * image_tensor[:, 0, :, :] + 0.587 * image_tensor[:, 1, :, :] + 0.114 * image_tensor[:, 2, :, :]
    gray = gray.unsqueeze(1) # [B, 1, H, W]
    
    weights = get_srm_filter_weights()
    # Apply conv2d
    # Pad to keep size: 5x5 filter needs padding=2
    res = F.conv2d(gray, weights, padding=2)
    
    # Combine 3 filters (max or mean) for visualization
    res_map = torch.mean(torch.abs(res), dim=1)
    
    # Normalize to 0-1 for visualization
    min_val = res_map.min()
    max_val = res_map.max()
    if max_val - min_val > 1e-8:
        res_map = (res_map - min_val) / (max_val - min_val)
    
    return res_map.squeeze().numpy()

def get_spectrum_heatmap(image_path):
    # image_path: Path to image
    img = cv2.imread(image_path, 0) # Load as grayscale
    if img is None:
        return None
        
    img = cv2.resize(img, (224, 224))
    
    # FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # Normalize to 0-255
    min_val = magnitude_spectrum.min()
    max_val = magnitude_spectrum.max()
    if max_val - min_val > 1e-8:
        magnitude_spectrum = (magnitude_spectrum - min_val) / (max_val - min_val)
    magnitude_spectrum = (magnitude_spectrum * 255).astype(np.uint8)
    
    # Apply heatmap color map
    heatmap = cv2.applyColorMap(magnitude_spectrum, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(heatmap)
