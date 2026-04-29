import logging
from typing import Dict, Any, Optional, Union
import torch
from PIL import Image

from ai_image_detector.inference.config import ConfigManager, InferenceConfig
from ai_image_detector.inference.strategy import InferenceEngine
from ai_image_detector.models.model import AIGCImageDetector
from ai_image_detector.ntire.model import HybridAIGCDetector
from ai_image_detector.ntire.model_v10 import V10CompetitionResetModel


class DetectorInterface:
    """检测器统一接口"""
    
    def __init__(self, model_path: str, device: str = "cuda", enable_debug: bool = False, config_name: str = "default"):
        self.logger = logging.getLogger("inference.interface")
        self.enable_debug = bool(enable_debug)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger.info("loading_model path=%s device=%s", model_path, self.device)
        
        # 初始化配置管理器
        self.config_manager = ConfigManager(config_name)
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = self.config_manager.get_inference_config(checkpoint)
        self.model = self._load_model(checkpoint, self.config)
        
        # 初始化推理引擎
        self.inference_engine = InferenceEngine(
            self.model, 
            str(self.device),
            scales=self.config.scales,
            tta_flip=self.config.tta_flip
        )
        
        self.logger.info(
            "model_loaded_successfully family=%s class=%s device=%s scales=%s tta=%s mode=%s",
            self.config.model_family,
            self.model.__class__.__name__,
            self.device,
            self.config.scales,
            self.config.tta_flip,
            self.inference_engine.get_available_modes(),
        )
    
    def _load_model(self, checkpoint: Dict, config: InferenceConfig) -> torch.nn.Module:
        """加载模型"""
        state_dict = self._extract_state_dict(checkpoint)
        
        # 提取权重
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        
        # 根据模型族加载对应模型
        if self.config.model_family == "v10":
            model = V10CompetitionResetModel(
                backbone_name="vit_base_patch16_clip_224.openai",
                pretrained_backbone=False,
                semantic_trainable_layers=0,
                image_size=config.image_size,
                frequency_dim=256,
                noise_dim=256,
                fused_dim=512,
                head_hidden_dim=256,
                dropout=0.3,
                fusion_gate_input_dropout=0.1,
                fusion_feature_dropout=0.1,
                alpha_max=0.35,
                enable_noise_expert=True,
            )
            tri_required = (
                "tri_fusion.semantic_proj.weight",
                "tri_fusion.frequency_proj.weight",
                "tri_fusion.noise_proj.weight",
                "tri_fusion.gate.0.weight",
                "tri_fusion.gate.3.weight",
                "tri_classifier.0.weight",
                "tri_classifier.3.weight",
            )
            has_trained_tri_fusion = all(key in new_state_dict for key in tri_required)
            model.set_tri_fusion_decision_enabled(has_trained_tri_fusion)
            if not has_trained_tri_fusion:
                self.logger.warning("Tri-fusion weights missing; using deploy_safe_tri_branch fallback.")
        elif self.config.model_family == "ntire":
            model = HybridAIGCDetector(
                backbone_name="vit_base_patch16_clip_224.openai",
                pretrained_backbone=False,
                image_size=config.image_size,
                use_aux_heads=True,
                fused_dim=512,
                head_hidden_dim=256,
                dropout=0.3,
            )
        else:
            model_cfg = self.config_manager.get_model_config()
            model = AIGCImageDetector(model_cfg)
        
        # 加载权重。三分支融合新增参数允许旧 checkpoint 非严格加载；新增层需要重新训练后才有可靠性能。
        try:
            load_result = model.load_state_dict(new_state_dict, strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                self.logger.warning(
                    "checkpoint_loaded_non_strict missing=%s unexpected=%s",
                    load_result.missing_keys,
                    load_result.unexpected_keys,
                )
        except RuntimeError as exc:
            raise RuntimeError(f"checkpoint state_dict does not match {model.__class__.__name__}: {exc}") from exc
        model.to(self.device)
        model.eval()
        
        return model
    
    def _extract_state_dict(self, checkpoint):
        """提取状态字典"""
        if isinstance(checkpoint, dict):
            if "ema_shadow" in checkpoint and checkpoint["ema_shadow"]:
                return checkpoint["ema_shadow"]
            if isinstance(checkpoint.get("model_state_dict"), dict):
                return checkpoint["model_state_dict"]
            if isinstance(checkpoint.get("model"), dict):
                return checkpoint["model"]
            if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
                return checkpoint
        raise ValueError("invalid checkpoint format: missing state_dict payload")
    
    def predict(
        self, 
        image_input: Union[str, Image.Image],
        debug: bool = False,
        threshold: Optional[float] = None,
        threshold_profile: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行推理"""
        # 处理图像输入
        if isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")
        
        try:
            # 执行三分支门控融合推理。
            requested_mode = mode or "deploy_safe_tri_branch"
            if self.config.model_family == "v10" and hasattr(self.model, "set_inference_mode"):
                self.model.set_inference_mode(requested_mode)
            outputs = self.inference_engine.predict(image, requested_mode)
            
            # 处理推理结果
            result = self._process_inference_result(outputs, threshold, threshold_profile)
            
            if debug or self.enable_debug:
                self.logger.info(
                    "predict_done prediction=%s confidence=%.6f mode=%s threshold=%.3f",
                    result.get("prediction"),
                    result.get("confidence", 0.0),
                    result.get("mode"),
                    result.get("threshold", 0.5),
                )
            
            return result
        except Exception as e:
            self.logger.exception("prediction_failed error=%s", e)
            raise
    
    def _process_inference_result(self, outputs: Dict[str, Any], 
                                threshold: Optional[float] = None, 
                                threshold_profile: Optional[str] = None) -> Dict[str, Any]:
        """处理推理结果"""
        logit_tensor = outputs.get("logit")
        if logit_tensor is None:
            raise ValueError("No logit found in outputs")
        
        # 计算概率
        calibrated_logit = logit_tensor / self.config.temperature
        probability = float(torch.sigmoid(calibrated_logit).item())
        probability = max(0.0, min(1.0, probability))
        
        # 解析阈值
        used_threshold, used_profile = self._resolve_threshold(threshold, threshold_profile)
        
        # 生成预测结果
        prediction = "AIGC" if probability >= used_threshold else "REAL"
        confidence = probability if prediction == "AIGC" else 1.0 - probability
        threshold_percent = float(used_threshold) * 100.0
        probability_percent = probability * 100.0
        decision_rule_text = (
            f"AIGC probability {probability_percent:.1f}% >= threshold {threshold_percent:.1f}%, prediction={prediction}"
            if prediction == "AIGC"
            else f"AIGC probability {probability_percent:.1f}% < threshold {threshold_percent:.1f}%, prediction={prediction}"
        )
        
        # 处理分支贡献
        branch_contribution = self._process_branch_contribution(outputs)
        
        # 计算分支证据
        branch_evidence, branch_triangle, branch_support = self._compute_branch_evidence(
            prediction, branch_contribution, outputs
        )
        
        return {
            "prediction": prediction,
            "label": prediction,
            "label_id": 1 if prediction == "AIGC" else 0,
            "confidence": confidence,
            "probabilities": {
                "real": float(1.0 - probability),
                "aigc": float(probability),
            },
            "branch_contribution": branch_contribution,
            "branch_usage": branch_contribution,
            "branch_evidence": branch_evidence,
            "branch_triangle": branch_triangle,
            "branch_support": branch_support,
            "branch_analysis_mode": "support_weighted_usage",
            "probability": probability,
            "branch_scores": self._extract_branch_scores(outputs),
            "fusion_weights": self._extract_fusion_weights(outputs),
            "decision_weights": self._extract_named_weights(outputs.get("decision_weights")),
            "evidence_weights": self._extract_named_weights(outputs.get("evidence_weights")),
            "temperature": self.config.temperature,
            "scales": self.config.scales,
            "tta_flip": self.config.tta_flip,
            "threshold": float(used_threshold),
            "threshold_used": float(used_threshold),
            "threshold_percent": threshold_percent,
            "decision_rule_text": decision_rule_text,
            "threshold_profile": used_profile,
            "analysis_thresholds": self.config_manager.get_analysis_thresholds(),
            "mode": outputs.get("active_inference_mode", "tri_fusion"),
            "raw_logit": self._tensor_item(logit_tensor),
            "logit": self._tensor_item(logit_tensor),
            "decision_logit": self._tensor_item(outputs.get("decision_logit", logit_tensor)),
            "tri_fusion_logit": self._tensor_item(outputs.get("tri_fusion_logit", logit_tensor)),
            "fused_logit": self._tensor_item(outputs.get("fused_logit", logit_tensor)),
            "stable_sf_logit": self._tensor_item(outputs.get("stable_sf_logit")),
            "semantic_score": self._prob_from_logit(outputs.get("semantic_logit")),
            "frequency_score": self._prob_from_logit(outputs.get("freq_logit")),
            "noise_score": self._prob_from_logit(outputs.get("noise_logit")),
            "semantic_logit": self._tensor_item(outputs.get("semantic_logit")),
            "frequency_logit": self._tensor_item(outputs.get("freq_logit")),
            "noise_logit": self._tensor_item(outputs.get("noise_logit")),
            "base_logit": self._tensor_item(outputs.get("base_logit")),
            "noise_delta_logit": self._tensor_item(outputs.get("noise_delta_logit")),
            "alpha": self._tensor_item(outputs.get("alpha")),
            "noise_enabled_for_decision": self._tensor_bool(outputs.get("noise_enabled_for_decision")),
            "tri_fusion_enabled_for_decision": self._tensor_bool(outputs.get("tri_fusion_enabled_for_decision")),
            "inference_mode": outputs.get("active_inference_mode", "deploy_safe_tri_branch"),
        }
    
    def _process_branch_contribution(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """处理分支贡献"""
        branch_contribution = {"rgb": None, "noise": None, "frequency": None}
        fusion_weights = outputs.get("fusion_weights")
        
        if fusion_weights is not None and torch.is_tensor(fusion_weights):
            weights = fusion_weights[0] if fusion_weights.dim() == 2 else fusion_weights
            if weights.numel() >= 3:
                branch_contribution = {
                    "rgb": float(weights[0].item()),
                    "frequency": float(weights[1].item()),
                    "noise": float(weights[2].item()),
                }
        
        return branch_contribution
    
    def _compute_branch_evidence(self, prediction: str, branch_usage: Dict[str, float], 
                               outputs: Dict[str, Any]) -> tuple:
        """计算分支证据"""
        semantic_prob = self._prob_from_logit(outputs.get("semantic_logit"))
        frequency_prob = self._prob_from_logit(outputs.get("freq_logit"))
        noise_prob = self._prob_from_logit(outputs.get("noise_logit"))
        
        branch_support = {
            "rgb": float(semantic_prob if prediction == "AIGC" else 1.0 - semantic_prob) if semantic_prob is not None else 0.5,
            "frequency": float(frequency_prob if prediction == "AIGC" else 1.0 - frequency_prob) if frequency_prob is not None else 0.5,
            "noise": float(noise_prob if prediction == "AIGC" else 1.0 - noise_prob) if noise_prob is not None else 0.5,
        }
        
        usage_profile = {
            "rgb": float(branch_usage.get("rgb") or 0.0),
            "frequency": float(branch_usage.get("frequency") or 0.0),
            "noise": float(branch_usage.get("noise") or 0.0),
        }
        
        normalized_usage = self._normalize_branch_profile(usage_profile)
        active_evidence = {
            key: branch_support[key] * (0.5 + 0.5 * normalized_usage[key]) if usage_profile[key] > 1e-8 else 0.0
            for key in normalized_usage
        }
        
        if sum(active_evidence.values()) <= 1e-8:
            active_evidence = branch_support
        
        triangle_profile = self._normalize_branch_profile(active_evidence)
        return active_evidence, triangle_profile, branch_support
    
    def _resolve_threshold(self, threshold: Optional[float], threshold_profile: Optional[str]) -> tuple:
        """解析阈值"""
        if threshold is not None:
            profile = self.config_manager._normalize_threshold_profile(threshold_profile or self.config.threshold_profile)
            return float(threshold), profile
        
        profile = self.config.threshold_profile if threshold_profile is None else \
            self.config_manager._normalize_threshold_profile(threshold_profile)
        if self.config.threshold is not None:
            return float(self.config.threshold), profile
        
        threshold_config = self.config_manager.get_threshold_config()
        return threshold_config.get_threshold(profile), profile
    
    def _tensor_item(self, value: Optional[torch.Tensor]) -> Optional[float]:
        """获取张量值"""
        if value is None or not torch.is_tensor(value):
            return None
        if value.numel() == 0:
            return None
        return float(value.reshape(-1)[0].item())

    def _tensor_bool(self, value: Any) -> bool:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return False
            return bool(value.reshape(-1)[0].item())
        return bool(value)
    
    def _prob_from_logit(self, value: Optional[torch.Tensor]) -> Optional[float]:
        """从 logit 计算概率"""
        if value is None or not torch.is_tensor(value):
            return None
        return float(torch.sigmoid(value.reshape(-1)[0]).item())
    
    def _normalize_branch_profile(self, profile: Dict[str, float]) -> Dict[str, float]:
        """归一化分支配置"""
        clipped = {key: max(float(value), 0.0) for key, value in profile.items()}
        total = sum(clipped.values())
        if total <= 1e-8:
            return {"rgb": 1.0 / 3.0, "frequency": 1.0 / 3.0, "noise": 1.0 / 3.0}
        return {key: value / total for key, value in clipped.items()}
    
    def _extract_fusion_weights(self, outputs: Dict[str, Any]) -> Optional[list]:
        """提取融合权重"""
        fusion_weights = outputs.get("fusion_weights")
        if fusion_weights is not None and torch.is_tensor(fusion_weights):
            weights = fusion_weights[0] if fusion_weights.dim() == 2 else fusion_weights
            if weights.numel() >= 3:
                return [float(weights[index].item()) for index in range(3)]
        return None

    def _extract_named_weights(self, value: Any) -> Optional[Dict[str, float]]:
        if value is not None and torch.is_tensor(value):
            weights = value[0] if value.dim() == 2 else value
            if weights.numel() >= 3:
                return {
                    "semantic": float(weights[0].item()),
                    "frequency": float(weights[1].item()),
                    "noise": float(weights[2].item()),
                }
        return None

    def _extract_branch_scores(self, outputs: Dict[str, Any]) -> Dict[str, Optional[float]]:
        return {
            "semantic": self._prob_from_logit(outputs.get("semantic_logit")),
            "frequency": self._prob_from_logit(outputs.get("freq_logit")),
            "noise": self._prob_from_logit(outputs.get("noise_logit")),
        }
    
    def get_available_inference_modes(self) -> list:
        """获取可用的推理模式"""
        return self.inference_engine.get_available_modes()
