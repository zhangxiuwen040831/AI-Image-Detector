from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import os
import logging
import torch

from ai_image_detector.utils.config import load_config


@dataclass
class InferenceConfig:
    """推理配置"""
    model_family: str
    image_size: int = 224
    threshold_profile: str = "balanced"
    threshold: Optional[float] = None
    scales: List[int] = None
    tta_flip: bool = False
    temperature: float = 1.0
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [self.image_size]


@dataclass
class ThresholdConfig:
    """阈值配置"""
    recall_first: float = 0.20
    balanced: float = 0.35
    precision_first: float = 0.55
    
    def get_threshold(self, profile: str) -> float:
        """获取指定配置的阈值"""
        profile_map = {
            "recall-first": self.recall_first,
            "balanced": self.balanced,
            "precision-first": self.precision_first,
        }
        return profile_map.get(profile, self.balanced)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_name: str = "default"):
        self.logger = logging.getLogger("inference.config")
        self.config_name = config_name
        self.infer_cfg = {}
        self.model_cfg = {}
        self._load_configs()
    
    def _load_configs(self):
        """加载配置"""
        # 从配置文件加载
        try:
            if isinstance(self.config_name, str) and "/" in self.config_name:
                config_type, config_file = self.config_name.split("/", 1)
                cfg = load_config(config_type, config_file)
            else:
                cfg = load_config("infer", self.config_name)
            self.model_cfg = cfg.get("model", {})
            self.infer_cfg = cfg.get("infer", {})
            self.logger.info("loaded_config config_name=%s", self.config_name)
        except Exception as exc:
            self.logger.warning("config_load_failed using_defaults error=%s", exc)
            self.infer_cfg = {}
            self.model_cfg = {
                "backbone": "resnet18",
                "rgb_pretrained": True,
                "noise_pretrained": False,
                "freq_pretrained": False,
                "fused_dim": 512,
                "classifier_hidden_dim": 256,
                "dropout": 0.3,
            }
    
    def get_inference_config(self, checkpoint: Dict) -> InferenceConfig:
        """获取推理配置"""
        # 从检查点读取温度
        temperature = float(checkpoint.get("temperature", 1.0)) if isinstance(checkpoint, dict) else 1.0
        
        # 解析阈值配置
        threshold_profile = self._resolve_threshold_profile()
        threshold = self._resolve_threshold(threshold_profile)
        
        return InferenceConfig(
            model_family=self._detect_model_family(checkpoint),
            image_size=int(self.infer_cfg.get("image_size", 224)),
            threshold_profile=threshold_profile,
            threshold=threshold,
            scales=self.infer_cfg.get("scales", [224]),
            tta_flip=self.infer_cfg.get("tta_flip", False),
            temperature=temperature,
        )
    
    def get_model_config(self) -> Dict:
        """获取模型配置"""
        return self.model_cfg
    
    def get_threshold_config(self) -> ThresholdConfig:
        """获取阈值配置"""
        thresholds = self.infer_cfg.get("thresholds", {})
        return ThresholdConfig(
            recall_first=float(thresholds.get("recall-first", 0.20)),
            balanced=float(thresholds.get("balanced", 0.35)),
            precision_first=float(thresholds.get("precision-first", 0.55)),
        )
    
    def get_analysis_thresholds(self) -> Dict[str, float]:
        """获取分析阈值"""
        analysis_thresholds = {
            "photos-test-precision-first": 0.55,
        }
        cfg_analysis_thresholds = self.infer_cfg.get("analysis_thresholds")
        if isinstance(cfg_analysis_thresholds, dict):
            for key, value in cfg_analysis_thresholds.items():
                try:
                    normalized_key = str(key).strip().lower().replace("_", "-")
                    analysis_thresholds[normalized_key] = float(value)
                except Exception:
                    continue
        return analysis_thresholds
    
    def _resolve_threshold_profile(self) -> str:
        """解析阈值配置"""
        env_profile = os.getenv("AIGC_THRESHOLD_PROFILE")
        cfg_profile = self.infer_cfg.get("threshold_profile")
        profile = cfg_profile or env_profile or "balanced"
        return self._normalize_threshold_profile(profile)
    
    def _resolve_threshold(self, profile: str) -> Optional[float]:
        """解析阈值"""
        env_threshold = os.getenv("AIGC_THRESHOLD")
        cfg_threshold = self.infer_cfg.get("threshold")
        
        if cfg_threshold is not None:
            return float(cfg_threshold)
        if env_threshold is not None:
            return float(env_threshold)
        
        # 使用默认阈值
        threshold_config = self.get_threshold_config()
        return threshold_config.get_threshold(profile)
    
    @staticmethod
    def _normalize_threshold_profile(profile: str) -> str:
        """标准化阈值配置名称"""
        profile = str(profile).strip().lower().replace("_", "-")
        aliases = {
            "default": "balanced",
            "best-f1": "balanced",
            "bestf1": "balanced",
            "recall-first": "recall-first",
            "precision-first": "precision-first",
        }
        return aliases.get(profile, profile)
    
    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Dict:
        """从 checkpoint 中提取实际模型权重。"""
        if isinstance(checkpoint, dict):
            if isinstance(checkpoint.get("ema_shadow"), dict) and checkpoint["ema_shadow"]:
                return checkpoint["ema_shadow"]
            if isinstance(checkpoint.get("model_state_dict"), dict):
                return checkpoint["model_state_dict"]
            if isinstance(checkpoint.get("model"), dict):
                return checkpoint["model"]
            if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
                return checkpoint
        raise ValueError("invalid checkpoint format: missing state_dict payload")
    
    @classmethod
    def _detect_model_family(cls, checkpoint: Dict) -> str:
        """检测模型族"""
        state_dict = cls._extract_state_dict(checkpoint)
        keys = state_dict.keys()
        if any("semantic_branch" in key for key in keys):
            return "v10"
        return "legacy"
