from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import cv2


class VisualizationGenerator(ABC):
    """可视化生成器基类"""
    
    @abstractmethod
    def generate(self, **kwargs) -> Optional[str]:
        """生成可视化图像"""
        pass
    
    @abstractmethod
    def is_supported(self, model_family: str) -> bool:
        """检查是否支持指定的模型族"""
        pass
    
    def _to_base64(self, img: Image.Image) -> str:
        """转换为 base64"""
        if img is None:
            return None
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class GradCAMGenerator(VisualizationGenerator):
    """Grad-CAM 生成器"""
    
    def __init__(self, model, device, image_size: int = 224):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.target_layer_name = self._resolve_target_layer()
    
    def _resolve_target_layer(self) -> str:
        """解析目标层名称"""
        if hasattr(self.model, 'detector') and hasattr(self.model.detector, 'rgb_branch'):
            backbone_name = getattr(self.model.detector.rgb_branch, "backbone_name", "")
            if backbone_name == "resnet18":
                return "detector.rgb_branch.encoder.layer4"
            if backbone_name == "efficientnet_b0":
                return "detector.rgb_branch.encoder.conv_head"
            if backbone_name == "convnext_tiny":
                return "detector.rgb_branch.encoder.stages.3"
        return "detector.rgb_branch.encoder"
    
    def is_supported(self, model_family: str) -> bool:
        """检查是否支持 Grad-CAM"""
        return model_family not in {"ntire", "v10"}
    
    def generate(self, image_path: str, model_family: str, **kwargs) -> Optional[str]:
        """生成 Grad-CAM 热力图"""
        if not self.is_supported(model_family):
            return None
        
        try:
            image = Image.open(image_path).convert("RGB")
            from src.data.transforms import get_transforms

            legacy_transform = get_transforms(image_size=self.image_size)
            img_tensor = legacy_transform(image).unsqueeze(0).to(self.device)

            self.model.zero_grad(set_to_none=True)
            outputs = self.model.detector.forward_with_features(img_tensor)
            logit = outputs.get("logit")
            rgb_feature_map = outputs.get("rgb_feature_map")

            if logit is None or rgb_feature_map is None:
                return None

            logit_value = logit.view(-1)[0]
            prob_aigc = torch.sigmoid(logit_value).detach().item()
            prediction_index = 1 if prob_aigc >= 0.5 else 0
            target_score = logit_value if prediction_index == 1 else -logit_value

            gradients = torch.autograd.grad(
                target_score,
                rgb_feature_map,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            if gradients is None:
                return None

            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * rgb_feature_map, dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = F.interpolate(
                cam,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            cam_np = cam.squeeze().detach().cpu().numpy().astype(np.float32)
            cam_min = float(np.min(cam_np))
            cam_max = float(np.max(cam_np))

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

            return self._to_base64(overlay)
        except Exception:
            return None


class FusionTriangleGenerator(VisualizationGenerator):
    """融合证据三角图生成器"""
    
    def is_supported(self, model_family: str) -> bool:
        """所有模型族都支持"""
        return True
    
    def generate(self, semantic: float, freq: float, noise: float, 
                 prediction: str, **kwargs) -> str:
        """生成融合证据三角图"""
        weights = self._normalize_weights(semantic, freq, noise)
        return self._draw_triangle(weights, prediction)
    
    def _normalize_weights(self, semantic: float, freq: float, noise: float) -> np.ndarray:
        """归一化权重"""
        weights = np.array([float(semantic), float(freq), float(noise)], dtype=np.float32)
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if total <= 1e-8:
            return np.array([1.0/3, 1.0/3, 1.0/3], dtype=np.float32)
        return weights / total
    
    def _draw_triangle(self, weights: np.ndarray, prediction: str) -> str:
        """绘制三角图"""
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
        radius = 10
        draw.ellipse(
            (point_xy[0] - radius, point_xy[1] - radius, point_xy[0] + radius, point_xy[1] + radius),
            fill="#DA205A",
            outline="#FFFFFF",
            width=2,
        )

        draw.text((sem_xy[0] - 35, sem_xy[1] - 26), "Semantic", fill="#E5E7EB")
        draw.text((noise_xy[0] - 18, noise_xy[1] + 8), "Noise", fill="#E5E7EB")
        draw.text((freq_xy[0] - 34, freq_xy[1] + 8), "Frequency", fill="#E5E7EB")
        draw.text((126, 12), "Fusion Evidence Triangle", fill="#FFFFFF")
        draw.text((160, height - 30), "Prediction: " + prediction, fill="#E5E7EB")

        return self._to_base64(img)


class VisualizationFactory:
    """可视化生成器工厂"""
    
    @staticmethod
    def create_generator(viz_type: str, **kwargs) -> VisualizationGenerator:
        """创建可视化生成器"""
        generators = {
            "grad_cam": lambda: GradCAMGenerator(**kwargs),
            "fusion_triangle": lambda: FusionTriangleGenerator(),
        }
        
        if viz_type not in generators:
            raise ValueError(f"Unknown visualization type: {viz_type}")
        
        return generators[viz_type]()
