
import os
import sys
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model import AIGCImageDetector
from backend.transforms import get_transforms

class ForensicDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {model_path} to {self.device}")
        
        # Load configuration (hardcoded or from config file, matching training)
        # Assuming default config from dataset_config.yaml
        model_cfg = {
            "backbone": "resnet18",
            "rgb_pretrained": True,
            "noise_pretrained": False,
            "freq_pretrained": False,
            "fused_dim": 512,
            "classifier_hidden_dim": 256,
            "dropout": 0.3
        }
        
        self.model = AIGCImageDetector(model_cfg)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Handle potential key mismatches (e.g. "module." prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.transforms = get_transforms(image_size=224)
        
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        probability = outputs['probability'].item()
        prediction = "AIGC" if probability > 0.5 else "Real"
        confidence = probability if probability > 0.5 else 1 - probability
        
        # Calculate branch contributions/scores based on feature norms
        # This is a heuristic for visualization
        rgb_norm = torch.norm(outputs['rgb_feat']).item()
        noise_norm = torch.norm(outputs['noise_feat']).item()
        freq_norm = torch.norm(outputs['freq_feat']).item()
        
        total_norm = rgb_norm + noise_norm + freq_norm + 1e-8
        
        branch_scores = {
            "rgb": rgb_norm / total_norm,
            "noise": noise_norm / total_norm,
            "frequency": freq_norm / total_norm
        }
        
        return {
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
            "branch_scores": branch_scores
        }
