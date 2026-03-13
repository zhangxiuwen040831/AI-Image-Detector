import os
import sys
import logging
import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ai_image_detector.models.model import AIGCImageDetector
from src.data.transforms import get_transforms
from ai_image_detector.utils.config import load_config, get_config

class ForensicDetector:
    def __init__(self, model_path, device='cuda', enable_debug=False, config_name="infer/default"):
        self.logger = logging.getLogger("inference.detector")
        self.enable_debug = bool(enable_debug)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger.info("loading_model path=%s device=%s", model_path, self.device)
        
        # Load configuration from config file
        try:
            cfg = load_config(config_name)
            model_cfg = cfg.get("model", {})
            self.logger.info("loaded_config config_name=%s", config_name)
        except Exception as e:
            self.logger.warning("config_load_failed using_defaults error=%s", e)
            # Default configuration if config file not found
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
        self.temperature = float(checkpoint.get("temperature", 1.0)) if isinstance(checkpoint, dict) else 1.0
        self.temperature = max(self.temperature, 1e-3)
            
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
        self.target_layer_name = self._resolve_target_layer_name()
        self.logger.info("model_loaded_successfully target_layer=%s", self.target_layer_name)

    def _resolve_target_layer_name(self):
        backbone_name = getattr(self.model.detector.rgb_branch, "backbone_name", "")
        if backbone_name == "resnet18":
            return "detector.rgb_branch.encoder.layer4"
        if backbone_name == "efficientnet_b0":
            return "detector.rgb_branch.encoder.conv_head"
        if backbone_name == "convnext_tiny":
            return "detector.rgb_branch.encoder.stages.3"
        return "detector.rgb_branch.encoder"

    def _to_base64(self, img):
        if img is None:
            return None
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    def generate_grad_cam(self, image_path, debug=False):
        debug_info = {
            "target_layer": self.target_layer_name,
            "prediction_index": None,
            "activation_shape": None,
            "gradient_shape": None,
            "cam_min": None,
            "cam_max": None,
            "grad_cam_generated": False,
            "overlay_generated": False,
            "error": None,
        }
        try:
            image = Image.open(image_path).convert('RGB')
            img_tensor = self.transforms(image).unsqueeze(0).to(self.device)

            self.model.zero_grad(set_to_none=True)
            outputs = self.model.detector.forward_with_features(img_tensor)
            logit = outputs.get('logit')
            rgb_feature_map = outputs.get('rgb_feature_map')

            if logit is None:
                raise RuntimeError("forward_with_features did not return logit")
            if rgb_feature_map is None:
                raise RuntimeError("forward_with_features did not return rgb_feature_map")

            debug_info["activation_shape"] = list(rgb_feature_map.shape)

            logit_value = logit.view(-1)[0]
            prob_aigc = torch.sigmoid(logit_value).detach().item()
            prediction_index = 1 if prob_aigc >= 0.5 else 0
            target_score = logit_value if prediction_index == 1 else -logit_value
            debug_info["prediction_index"] = prediction_index

            gradients = torch.autograd.grad(
                target_score,
                rgb_feature_map,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            if gradients is None:
                raise RuntimeError("gradient is None for rgb_feature_map")

            debug_info["gradient_shape"] = list(gradients.shape)

            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * rgb_feature_map, dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = F.interpolate(
                cam,
                size=(image.height, image.width),
                mode='bilinear',
                align_corners=False
            )
            cam_np = cam.squeeze().detach().cpu().numpy().astype(np.float32)
            cam_min = float(np.min(cam_np))
            cam_max = float(np.max(cam_np))
            debug_info["cam_min"] = cam_min
            debug_info["cam_max"] = cam_max

            if cam_max - cam_min <= 1e-8:
                cam_norm = np.zeros_like(cam_np, dtype=np.float32)
            else:
                cam_norm = (cam_np - cam_min) / (cam_max - cam_min)

            heatmap_bgr = cv2.applyColorMap(np.uint8(np.clip(cam_norm, 0.0, 1.0) * 255), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            heatmap = Image.fromarray(heatmap_rgb)

            base_rgb = np.array(image, dtype=np.uint8)
            overlay_rgb = cv2.addWeighted(base_rgb, 0.5, heatmap_rgb, 0.5, 0.0)
            overlay = Image.fromarray(overlay_rgb)

            debug_info["grad_cam_generated"] = True
            debug_info["overlay_generated"] = True
            if debug or self.enable_debug:
                self.logger.info(
                    "grad_cam_generated target_layer=%s prediction_index=%s activation_shape=%s gradient_shape=%s cam_min=%.6f cam_max=%.6f",
                    debug_info["target_layer"],
                    debug_info["prediction_index"],
                    debug_info["activation_shape"],
                    debug_info["gradient_shape"],
                    debug_info["cam_min"],
                    debug_info["cam_max"],
                )
            return heatmap, overlay, debug_info
        except Exception as e:
            debug_info["error"] = str(e)
            self.logger.exception("grad_cam_generation_failed image_path=%s error=%s", image_path, e)
            return None, None, debug_info
    
    def predict(self, image_path, debug=False):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)

        logit_tensor = outputs.get('logit')
        if logit_tensor is not None:
            calibrated_logit = logit_tensor / self.temperature
            probability = float(torch.sigmoid(calibrated_logit).item())
        else:
            probability_tensor = outputs.get('probability')
            probability = float(probability_tensor.item()) if probability_tensor is not None else 0.5
        probability = max(0.0, min(1.0, probability))
        prediction = "AIGC" if probability > 0.5 else "REAL"
        confidence = probability if prediction == "AIGC" else 1 - probability

        branch_contribution = {
            "rgb": None,
            "noise": None,
            "frequency": None
        }

        rgb_feat = outputs.get('rgb_feat')
        noise_feat = outputs.get('noise_feat')
        freq_feat = outputs.get('freq_feat')

        if rgb_feat is not None and noise_feat is not None and freq_feat is not None:
            rgb_norm = torch.norm(rgb_feat).item()
            noise_norm = torch.norm(noise_feat).item()
            freq_norm = torch.norm(freq_feat).item()
            total_norm = rgb_norm + noise_norm + freq_norm
            if total_norm > 0:
                branch_contribution = {
                    "rgb": float(rgb_norm / total_norm),
                    "noise": float(noise_norm / total_norm),
                    "frequency": float(freq_norm / total_norm)
                }

        probabilities = {
            "real": float(1 - probability),
            "aigc": float(probability)
        }

        heatmap, overlay, gradcam_debug = self.generate_grad_cam(image_path, debug=debug)
        grad_cam_b64 = self._to_base64(heatmap)
        grad_cam_overlay_b64 = self._to_base64(overlay)
        if grad_cam_b64 is not None:
            gradcam_debug["base64_length"] = len(grad_cam_b64)
        else:
            gradcam_debug["base64_length"] = 0
        if grad_cam_overlay_b64 is not None:
            gradcam_debug["overlay_base64_length"] = len(grad_cam_overlay_b64)
        else:
            gradcam_debug["overlay_base64_length"] = 0
        gradcam_debug["device"] = str(self.device)
        gradcam_debug["grad_cam_status"] = "success" if grad_cam_b64 else "failed"
        if gradcam_debug.get("grad_cam_status") == "failed" and gradcam_debug.get("error") is None:
            gradcam_debug["error"] = "Grad-CAM returned empty image"
        if debug or self.enable_debug:
            self.logger.info(
                "predict_done prediction=%s confidence=%.6f grad_cam_status=%s grad_cam_len=%s overlay_len=%s",
                prediction,
                confidence,
                gradcam_debug["grad_cam_status"],
                gradcam_debug["base64_length"],
                gradcam_debug["overlay_base64_length"],
            )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "branch_contribution": branch_contribution,
            "artifacts": {
                "noise_residual": None,
                "frequency_spectrum": None,
                "grad_cam": grad_cam_b64,
                "grad_cam_overlay": grad_cam_overlay_b64
            },
            "probability": probability,
            "branch_scores": branch_contribution,
            "debug": gradcam_debug if (debug or self.enable_debug) else None
        }
