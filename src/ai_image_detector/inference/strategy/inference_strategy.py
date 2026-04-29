from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import numpy as np
from PIL import Image

from ai_image_detector.ntire.augmentations import build_eval_transform


class InferenceStrategy(ABC):
    """推理策略基类"""
    
    @abstractmethod
    def execute(self, image: Image.Image) -> Dict[str, Any]:
        """执行推理"""
        pass


class TriFusionStrategy(InferenceStrategy):
    """三分支门控融合推理策略"""
    
    def __init__(self, model, device, image_size: int = 224):
        self.model = model
        self.device = device
        self.image_size = image_size
    
    def execute(self, image: Image.Image) -> Dict[str, Any]:
        """执行基础推理"""
        x = self._preprocess_image(image)
        with torch.inference_mode():
            outputs = self.model(x)
        return self._process_outputs(outputs)
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """预处理图像"""
        transform = build_eval_transform(image_size=self.image_size)
        arr = np.array(image.convert("RGB"))
        transformed = transform(image=arr)
        return transformed["image"].unsqueeze(0).to(self.device)
    
    def _process_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理输出"""
        result = {
            "logit": outputs.get("logit"),
            "decision_logit": outputs.get("decision_logit", outputs.get("logit")),
            "stable_sf_logit": outputs.get("stable_sf_logit"),
            "tri_fusion_logit": outputs.get("tri_fusion_logit", outputs.get("logit")),
            "fused_logit": outputs.get("fused_logit", outputs.get("logit")),
            "semantic_logit": outputs.get("semantic_logit"),
            "freq_logit": outputs.get("freq_logit"),
            "noise_logit": outputs.get("noise_logit"),
            "fusion_weights": outputs.get("fusion_weights"),
            "decision_weights": outputs.get("decision_weights"),
            "evidence_weights": outputs.get("evidence_weights"),
            "base_logit": outputs.get("base_logit", outputs.get("logit")),
            "legacy_base_logit": outputs.get("legacy_base_logit"),
            "noise_enabled_for_decision": outputs.get("noise_enabled_for_decision"),
            "tri_fusion_enabled_for_decision": outputs.get("tri_fusion_enabled_for_decision"),
            "active_inference_mode": outputs.get("active_inference_mode", "deploy_safe_tri_branch"),
        }
        return result


BaseOnlyStrategy = TriFusionStrategy


class MultiScaleStrategy(InferenceStrategy):
    """多尺度推理策略"""
    
    def __init__(self, model, device, scales: list = None, tta_flip: bool = False):
        self.model = model
        self.device = device
        self.scales = scales or [224]
        self.tta_flip = tta_flip
    
    def execute(self, image: Image.Image) -> Dict[str, Any]:
        """执行多尺度推理"""
        all_outputs = []
        
        for scale in self.scales:
            # 单尺度推理
            strategy = TriFusionStrategy(self.model, self.device, scale)
            outputs = strategy.execute(image)
            all_outputs.append(outputs)
            
            # 测试时增强（水平翻转）
            if self.tta_flip:
                flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_outputs = strategy.execute(flipped_image)
                # 翻转回来的处理
                all_outputs.append(flipped_outputs)
        
        # 聚合结果
        aggregated = self._aggregate_outputs(all_outputs)
        return aggregated
    
    def _aggregate_outputs(self, outputs_list: list) -> Dict[str, Any]:
        """聚合多尺度输出"""
        aggregated = {}
        tensor_keys = set()
        
        # 收集所有张量键
        for outputs in outputs_list:
            tensor_keys.update(key for key, value in outputs.items() if torch.is_tensor(value))
        
        # 聚合张量
        for key in tensor_keys:
            values = [outputs[key].detach().cpu() for outputs in outputs_list if key in outputs and torch.is_tensor(outputs[key])]
            if not values:
                continue
            first = values[0]
            if first.dim() == 0:
                aggregated[key] = torch.stack(values, dim=0).mean()
            else:
                aggregated[key] = torch.cat(values, dim=0).mean(dim=0, keepdim=True)
        
        # 聚合模式
        modes = [str(outputs.get("active_inference_mode")) for outputs in outputs_list if outputs.get("active_inference_mode")]
        if modes:
            aggregated["active_inference_mode"] = modes[0]
        
        return aggregated


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model, device: str, scales: list = None, tta_flip: bool = False):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.scales = scales or [224]
        self.tta_flip = tta_flip
        self.strategies = self._init_strategies()
    
    def _init_strategies(self) -> Dict[str, InferenceStrategy]:
        """初始化推理策略"""
        strategies = {
            "deploy_safe_tri_branch": TriFusionStrategy(self.model, self.device, self.scales[0]),
            "trained_tri_fusion": TriFusionStrategy(self.model, self.device, self.scales[0]),
            "tri_fusion": TriFusionStrategy(self.model, self.device, self.scales[0]),
            "base_only": TriFusionStrategy(self.model, self.device, self.scales[0]),
        }

        # 添加多尺度策略
        if len(self.scales) > 1 or self.tta_flip:
            strategies["multi_scale"] = MultiScaleStrategy(self.model, self.device, self.scales, self.tta_flip)
        
        return strategies
    
    def predict(self, image: Image.Image, mode: str = "deploy_safe_tri_branch") -> Dict[str, Any]:
        """执行推理"""
        # 如果指定了多尺度模式或配置了多尺度
        if mode == "multi_scale" or (len(self.scales) > 1 and mode in {"base_only", "tri_fusion", "deploy_safe_tri_branch"}):
            strategy = self.strategies.get("multi_scale", self.strategies["deploy_safe_tri_branch"])
        else:
            strategy = self.strategies.get(mode, self.strategies["deploy_safe_tri_branch"])
        
        return strategy.execute(image)
    
    def get_available_modes(self) -> list:
        """获取可用的推理模式"""
        return list(self.strategies.keys())
