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
from PIL import Image, ImageDraw

from ai_image_detector.models.model import AIGCImageDetector
from ai_image_detector.ntire.model import HybridAIGCDetector
from src.data.transforms import get_transforms
from ai_image_detector.utils.config import load_config

class ForensicDetector:
    def __init__(self, model_path, device='cuda', enable_debug=False, config_name="default"):
        self.logger = logging.getLogger("inference.detector")
        self.enable_debug = bool(enable_debug)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger.info("loading_model path=%s device=%s", model_path, self.device)
        
        try:
            if isinstance(config_name, str) and "/" in config_name:
                config_type, config_file = config_name.split("/", 1)
                cfg = load_config(config_type, config_file)
            else:
                cfg = load_config("infer", config_name)
            model_cfg = cfg.get("model", {})
            infer_cfg = cfg.get("infer", {})
            self.logger.info("loaded_config config_name=%s", config_name)
        except Exception as e:
            self.logger.warning("config_load_failed using_defaults error=%s", e)
            infer_cfg = {}
            model_cfg = {
                "backbone": "resnet18",
                "rgb_pretrained": True,
                "noise_pretrained": False,
                "freq_pretrained": False,
                "fused_dim": 512,
                "classifier_hidden_dim": 256,
                "dropout": 0.3
            }

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)
        self.model_family = self._detect_model_family(state_dict)
        image_size = int(infer_cfg.get("image_size", 224))

        if self.model_family == "ntire":
            self.model = HybridAIGCDetector(
                backbone_name=str(infer_cfg.get("backbone_name", "vit_base_patch16_clip_224.openai")),
                pretrained_backbone=False,
                image_size=image_size,
            )
        else:
            self.model = AIGCImageDetector(model_cfg)

        self.temperature = float(checkpoint.get("temperature", 1.0)) if isinstance(checkpoint, dict) else 1.0
        self.temperature = max(self.temperature, 1e-3)
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        load_result = self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        self.transforms = get_transforms(image_size=image_size)
        self.target_layer_name = self._resolve_target_layer_name()
        self.logger.info(
            "model_loaded_successfully family=%s class=%s target_layer=%s missing=%d unexpected=%d",
            self.model_family,
            self.model.__class__.__name__,
            self.target_layer_name,
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            if isinstance(checkpoint.get("model_state_dict"), dict):
                return checkpoint["model_state_dict"]
            if isinstance(checkpoint.get("model"), dict):
                return checkpoint["model"]
            if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
                return checkpoint
        raise ValueError("invalid checkpoint format: missing state_dict payload")

    def _detect_model_family(self, state_dict):
        keys = state_dict.keys()
        if any(k.startswith("semantic_branch.") for k in keys):
            return "ntire"
        return "legacy"

    def _resolve_target_layer_name(self):
        if self.model_family == "ntire":
            return "semantic_branch.encoder"
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

    def generate_fusion_evidence_plot(self, semantic, freq, noise, prediction):
        weights = np.array([float(semantic), float(freq), float(noise)], dtype=np.float32)
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if total <= 1e-8:
            weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
        else:
            weights = weights / total

        v_sem = np.array([0.5, 0.866], dtype=np.float32)
        v_noise = np.array([0.0, 0.0], dtype=np.float32)
        v_freq = np.array([1.0, 0.0], dtype=np.float32)
        point = weights[0] * v_sem + weights[2] * v_noise + weights[1] * v_freq
        width = 480
        height = 480
        margin_x = 48
        margin_y = 62
        draw_h = height - 2 * margin_y
        draw_w = width - 2 * margin_x

        def to_canvas(v):
            x = margin_x + float(v[0]) * draw_w
            y = height - margin_y - float(v[1]) * draw_h
            return (x, y)

        sem_xy = to_canvas(v_sem)
        noise_xy = to_canvas(v_noise)
        freq_xy = to_canvas(v_freq)
        point_xy = to_canvas(point)

        img = Image.new("RGB", (width, height), "#0B1020")
        draw = ImageDraw.Draw(img)

        draw.line([noise_xy, freq_xy, sem_xy, noise_xy], fill="#9CA3AF", width=3)
        r = 10
        draw.ellipse((point_xy[0] - r, point_xy[1] - r, point_xy[0] + r, point_xy[1] + r), fill="#DA205A", outline="#FFFFFF", width=2)

        draw.text((sem_xy[0] - 35, sem_xy[1] - 26), "Semantic", fill="#E5E7EB")
        draw.text((noise_xy[0] - 18, noise_xy[1] + 8), "Noise", fill="#E5E7EB")
        draw.text((freq_xy[0] - 34, freq_xy[1] + 8), "Frequency", fill="#E5E7EB")
        draw.text((126, 12), "Fusion Evidence Triangle", fill="#FFFFFF")
        draw.text((160, height - 30), f"Prediction: {prediction}", fill="#E5E7EB")

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
        
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
        if self.model_family == "ntire":
            debug_info["error"] = "grad_cam_not_supported_for_ntire_model"
            return None, None, debug_info
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

        fusion_weights = outputs.get("fusion_weights")
        rgb_feat = outputs.get('rgb_feat') or outputs.get("semantic_feat")
        noise_feat = outputs.get('noise_feat')
        freq_feat = outputs.get('freq_feat')

        if fusion_weights is not None and torch.is_tensor(fusion_weights):
            weights = fusion_weights[0] if fusion_weights.dim() == 2 else fusion_weights
            if weights.numel() >= 3:
                branch_contribution = {
                    "rgb": float(weights[0].item()),
                    "noise": float(weights[2].item()),
                    "frequency": float(weights[1].item())
                }
        elif rgb_feat is not None and noise_feat is not None and freq_feat is not None:
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

        semantic_weight = branch_contribution.get("rgb") if branch_contribution.get("rgb") is not None else 0.0
        frequency_weight = branch_contribution.get("frequency") if branch_contribution.get("frequency") is not None else 0.0
        noise_weight = branch_contribution.get("noise") if branch_contribution.get("noise") is not None else 0.0
        fusion_evidence_b64 = self.generate_fusion_evidence_plot(
            semantic=semantic_weight,
            freq=frequency_weight,
            noise=noise_weight,
            prediction=prediction,
        )

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
                "grad_cam_overlay": grad_cam_overlay_b64,
                "fusion_evidence": fusion_evidence_b64
            },
            "probability": probability,
            "branch_scores": branch_contribution,
            "debug": gradcam_debug if (debug or self.enable_debug) else None
        }
