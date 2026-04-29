import base64
import logging
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from ai_image_detector.models.model import AIGCImageDetector
from ai_image_detector.ntire.augmentations import build_eval_transform
from ai_image_detector.ntire.model import HybridAIGCDetector
from ai_image_detector.ntire.model_v10 import V10CompetitionResetModel
from ai_image_detector.utils.config import load_config


class ForensicDetector:
    def __init__(self, model_path, device="cuda", enable_debug=False, config_name="default"):
        self.logger = logging.getLogger("inference.detector")
        self.enable_debug = bool(enable_debug)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
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
        except Exception as exc:
            self.logger.warning("config_load_failed using_defaults error=%s", exc)
            infer_cfg = {}
            model_cfg = {
                "backbone": "resnet18",
                "rgb_pretrained": True,
                "noise_pretrained": False,
                "freq_pretrained": False,
                "fused_dim": 512,
                "classifier_hidden_dim": 256,
                "dropout": 0.3,
            }

        checkpoint = torch.load(model_path, map_location=self.device)

        ema_used = False
        if isinstance(checkpoint, dict) and "ema_shadow" in checkpoint and checkpoint["ema_shadow"]:
            state_dict = checkpoint["ema_shadow"]
            ema_used = True
        else:
            state_dict = self._extract_state_dict(checkpoint)

        self.logger.info("checkpoint_weights_loaded ema_used=%s", ema_used)

        self.model_family = self._detect_model_family(state_dict)
        image_size = int(infer_cfg.get("image_size", 224))
        requested_mode = str(
            infer_cfg.get("mode") or os.getenv("AIGC_INFERENCE_MODE") or self._default_mode_for_family(self.model_family)
        )

        if self.model_family == "v10":
            self.model = V10CompetitionResetModel(
                backbone_name=str(infer_cfg.get("backbone_name", "vit_base_patch16_clip_224.openai")),
                pretrained_backbone=False,
                semantic_trainable_layers=int(infer_cfg.get("semantic_trainable_layers", 0)),
                image_size=image_size,
                frequency_dim=int(infer_cfg.get("frequency_dim", 256)),
                noise_dim=int(infer_cfg.get("noise_dim", 256)),
                fused_dim=int(infer_cfg.get("fused_dim", 512)),
                head_hidden_dim=int(infer_cfg.get("head_hidden_dim", 256)),
                dropout=float(infer_cfg.get("dropout", 0.3)),
                fusion_gate_input_dropout=float(infer_cfg.get("fusion_gate_input_dropout", 0.1)),
                fusion_feature_dropout=float(infer_cfg.get("fusion_feature_dropout", 0.1)),
                alpha_max=float(infer_cfg.get("alpha_max", 0.35)),
                enable_noise_expert=bool(infer_cfg.get("enable_noise_expert", True)),
            )
            self.model.set_inference_mode(requested_mode)
            self.inference_mode = requested_mode
        elif self.model_family == "ntire":
            self.model = HybridAIGCDetector(
                backbone_name=str(infer_cfg.get("backbone_name", "vit_base_patch16_clip_224.openai")),
                pretrained_backbone=False,
                image_size=image_size,
                use_aux_heads=infer_cfg.get("use_aux_heads", True),
                fused_dim=infer_cfg.get("fused_dim", 512),
                head_hidden_dim=infer_cfg.get("head_hidden_dim", 256),
                dropout=infer_cfg.get("dropout", 0.3),
            )
            self.inference_mode = "hybrid"
        else:
            self.model = AIGCImageDetector(model_cfg)
            self.inference_mode = "legacy"

        raw_temp = float(checkpoint.get("temperature", 1.0)) if isinstance(checkpoint, dict) else 1.0
        self.temperature = max(raw_temp, 1e-6)
        self.logger.info("temperature_loaded raw=%.6f used=%.6f", raw_temp, self.temperature)

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value

        load_result = self.model.load_state_dict(new_state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

        self.image_size = image_size
        self.eval_transform = build_eval_transform(image_size=image_size)

        self.scales = infer_cfg.get("scales", [224])
        self.tta_flip = infer_cfg.get("tta_flip", False)

        self.thresholds = {
            "recall-first": 0.20,
            "balanced": 0.35,
            "precision-first": 0.35,
            "recall": 0.20,
            "precision": 0.35,
            "f1": 0.35,
        }
        cfg_thresholds = infer_cfg.get("thresholds")
        if isinstance(cfg_thresholds, dict):
            for key, value in cfg_thresholds.items():
                try:
                    normalized_key = self._normalize_threshold_profile(str(key))
                    self.thresholds[normalized_key] = float(value)
                except Exception:
                    continue

        self.analysis_thresholds = {
            "photos-test-precision-first": 0.55,
        }
        cfg_analysis_thresholds = infer_cfg.get("analysis_thresholds")
        if isinstance(cfg_analysis_thresholds, dict):
            for key, value in cfg_analysis_thresholds.items():
                try:
                    self.analysis_thresholds[str(key).strip().lower().replace("_", "-")] = float(value)
                except Exception:
                    continue

        env_profile = os.getenv("AIGC_THRESHOLD_PROFILE")
        self.threshold_profile = self._normalize_threshold_profile(
            str(infer_cfg.get("threshold_profile") or env_profile or "balanced")
        )
        env_threshold = os.getenv("AIGC_THRESHOLD")
        cfg_threshold = infer_cfg.get("threshold")
        raw_threshold = cfg_threshold if cfg_threshold is not None else env_threshold
        self.threshold = (
            float(raw_threshold) if raw_threshold is not None else float(self.thresholds.get(self.threshold_profile, 0.5))
        )

        self.target_layer_name = self._resolve_target_layer_name()
        self.logger.info(
            "model_loaded_successfully family=%s class=%s target_layer=%s missing=%d unexpected=%d temperature=%.4f scales=%s tta=%s ema_used=%s mode=%s threshold_profile=%s threshold=%.3f",
            self.model_family,
            self.model.__class__.__name__,
            self.target_layer_name,
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
            self.temperature,
            self.scales,
            self.tta_flip,
            ema_used,
            self.inference_mode,
            self.threshold_profile,
            self.threshold,
        )

    @staticmethod
    def _default_mode_for_family(model_family: str) -> str:
        if model_family == "v10":
            return "tri_fusion"
        if model_family == "ntire":
            return "hybrid"
        return "legacy"

    def get_available_inference_modes(self) -> List[str]:
        if self.model_family == "v10":
            return ["tri_fusion"]
        if self.model_family == "ntire":
            return ["hybrid"]
        return ["legacy"]

    @staticmethod
    def _normalize_threshold_profile(profile: str) -> str:
        profile = str(profile).strip().lower().replace("_", "-")
        aliases = {
            "default": "balanced",
            "best-f1": "balanced",
            "bestf1": "balanced",
            "recall-first": "recall-first",
            "precision-first": "precision-first",
        }
        return aliases.get(profile, profile)

    def _resolve_threshold(self, threshold: Optional[float], threshold_profile: Optional[str]) -> tuple[float, str]:
        if threshold is not None:
            profile = self._normalize_threshold_profile(threshold_profile or self.threshold_profile)
            return float(threshold), profile
        profile = self.threshold_profile if threshold_profile is None else self._normalize_threshold_profile(threshold_profile)
        return float(self.thresholds.get(profile, self.threshold)), profile

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            if isinstance(checkpoint.get("model_state_dict"), dict):
                return checkpoint["model_state_dict"]
            if isinstance(checkpoint.get("model"), dict):
                return checkpoint["model"]
            if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
                return checkpoint
        raise ValueError("invalid checkpoint format: missing state_dict payload")

    def _detect_model_family(self, state_dict):
        keys = state_dict.keys()
        if any(key.startswith("primary_fusion.") or key.startswith("base_classifier.") for key in keys):
            return "v10"
        if any(key.startswith("semantic_branch.") for key in keys):
            return "ntire"
        return "legacy"

    def _resolve_target_layer_name(self):
        if self.model_family in {"ntire", "v10"}:
            return "not_supported"
        backbone_name = getattr(self.model.detector.rgb_branch, "backbone_name", "")
        if backbone_name == "resnet18":
            return "detector.rgb_branch.encoder.layer4"
        if backbone_name == "efficientnet_b0":
            return "detector.rgb_branch.encoder.conv_head"
        if backbone_name == "convnext_tiny":
            return "detector.rgb_branch.encoder.stages.3"
        return "detector.rgb_branch.encoder"

    @staticmethod
    def _tensor_item(value: Optional[torch.Tensor]) -> Optional[float]:
        if value is None or not torch.is_tensor(value):
            return None
        if value.numel() == 0:
            return None
        return float(value.reshape(-1)[0].item())

    @staticmethod
    def _prob_from_logit(value: Optional[torch.Tensor]) -> Optional[float]:
        if value is None or not torch.is_tensor(value):
            return None
        return float(torch.sigmoid(value.reshape(-1)[0]).item())

    @staticmethod
    def _normalize_branch_profile(profile: Dict[str, float]) -> Dict[str, float]:
        clipped = {key: max(float(value), 0.0) for key, value in profile.items()}
        total = sum(clipped.values())
        if total <= 1e-8:
            return {"rgb": 1.0 / 3.0, "frequency": 1.0 / 3.0, "noise": 1.0 / 3.0}
        return {key: value / total for key, value in clipped.items()}

    def _compute_branch_evidence(
        self,
        prediction: str,
        branch_usage: Dict[str, Optional[float]],
        semantic_prob: Optional[float],
        frequency_prob: Optional[float],
        noise_prob: Optional[float],
    ) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
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

    def _to_base64(self, img):
        if img is None:
            return None
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _preprocess_image(self, image: Image.Image, scale: int) -> torch.Tensor:
        transform = build_eval_transform(image_size=scale)
        arr = np.array(image.convert("RGB"))
        transformed = transform(image=arr)
        return transformed["image"].unsqueeze(0)

    def _inference_single_scale(self, image: Image.Image, scale: int) -> Dict[str, Any]:
        x = self._preprocess_image(image, scale).to(self.device)
        with torch.inference_mode():
            outputs = self.model(x)
        return outputs

    def _aggregate_tensor_outputs(self, outputs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        aggregated: Dict[str, Any] = {}
        tensor_keys = set()
        for outputs in outputs_list:
            tensor_keys.update(key for key, value in outputs.items() if torch.is_tensor(value))

        for key in tensor_keys:
            values = [outputs[key].detach().cpu() for outputs in outputs_list if key in outputs and torch.is_tensor(outputs[key])]
            if not values:
                continue
            first = values[0]
            if first.dim() == 0:
                aggregated[key] = torch.stack(values, dim=0).mean()
            else:
                aggregated[key] = torch.cat(values, dim=0).mean(dim=0, keepdim=True)

        modes = [str(outputs.get("active_inference_mode")) for outputs in outputs_list if outputs.get("active_inference_mode")]
        if modes:
            aggregated["active_inference_mode"] = modes[0]
        return aggregated

    def _inference_multi_scale(self, image: Image.Image) -> Dict[str, Any]:
        all_outputs: List[Dict[str, Any]] = []

        for scale in self.scales:
            outputs = self._inference_single_scale(image, scale)
            all_outputs.append(outputs)

            if self.tta_flip:
                arr = np.array(image.convert("RGB"))
                arr_flip = np.ascontiguousarray(np.fliplr(arr))
                transform = build_eval_transform(image_size=scale)
                xf = transform(image=arr_flip)["image"].unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    out_flip = self.model(xf)
                all_outputs.append(out_flip)

        return self._aggregate_tensor_outputs(all_outputs)

    def predict(
        self,
        image_input: Union[str, Image.Image],
        debug: bool = False,
        threshold: Optional[float] = None,
        threshold_profile: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        requested_mode = None
        previous_mode = None
        if mode is not None and self.model_family == "v10":
          requested_mode = str(mode).strip()
          previous_mode = self.inference_mode
          self.model.set_inference_mode(requested_mode)
          self.inference_mode = requested_mode

        if isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
            temp_path = None
        else:
            image = Image.open(image_input).convert("RGB")
            temp_path = image_input

        try:
            if len(self.scales) == 1 and not self.tta_flip:
                outputs = self._inference_single_scale(image, self.scales[0])
            else:
                outputs = self._inference_multi_scale(image)

            logit_tensor = outputs["logit"]
            fusion_weights = outputs.get("fusion_weights")
            calibrated_logit = logit_tensor / self.temperature
            probability = float(torch.sigmoid(calibrated_logit).item())
            probability = max(0.0, min(1.0, probability))

            used_threshold, used_profile = self._resolve_threshold(threshold=threshold, threshold_profile=threshold_profile)

            prediction = "AIGC" if probability >= used_threshold else "REAL"
            confidence = probability if prediction == "AIGC" else 1.0 - probability

            branch_contribution = {"rgb": None, "noise": None, "frequency": None}
            fusion_weights_list = None
            if fusion_weights is not None and torch.is_tensor(fusion_weights):
                weights = fusion_weights[0] if fusion_weights.dim() == 2 else fusion_weights
                if weights.numel() >= 3:
                    fusion_weights_list = [float(weights[index].item()) for index in range(3)]
                    branch_contribution = {
                        "rgb": fusion_weights_list[0],
                        "frequency": fusion_weights_list[1],
                        "noise": fusion_weights_list[2],
                    }

            semantic_logit = outputs.get("semantic_logit")
            frequency_logit = outputs.get("freq_logit")
            noise_logit = outputs.get("noise_logit")
            base_logit = outputs.get("base_logit")
            noise_delta_logit = outputs.get("noise_delta_logit")
            alpha_used = outputs.get("alpha_used")
            if alpha_used is None:
                alpha_used = outputs.get("alpha")
            active_mode = str(outputs.get("active_inference_mode") or self.inference_mode)

            probabilities = {
                "real": float(1.0 - probability),
                "aigc": float(probability),
            }

            semantic_prob = self._prob_from_logit(semantic_logit)
            frequency_prob = self._prob_from_logit(frequency_logit)
            noise_prob = self._prob_from_logit(noise_logit)
            branch_evidence, branch_triangle, branch_support = self._compute_branch_evidence(
                prediction=prediction,
                branch_usage=branch_contribution,
                semantic_prob=semantic_prob,
                frequency_prob=frequency_prob,
                noise_prob=noise_prob,
            )

            fusion_evidence_b64 = None
            grad_cam_b64 = None
            grad_cam_overlay_b64 = None
            gradcam_debug = {
                "grad_cam_status": "not_generated",
                "error": "ForensicDetector visualization generation is disabled; API artifacts are generated in services.api.main.",
                "base64_length": 0,
                "overlay_base64_length": 0,
                "device": str(self.device),
                "model_family": self.model_family,
                "mode": active_mode,
            }

            if debug or self.enable_debug:
                self.logger.info(
                    "predict_done prediction=%s confidence=%.6f mode=%s threshold=%.3f grad_cam_status=%s",
                    prediction,
                    confidence,
                    active_mode,
                    used_threshold,
                    gradcam_debug["grad_cam_status"],
                )

            return {
                "prediction": prediction,
                "label": prediction,
                "label_id": 1 if prediction == "AIGC" else 0,
                "confidence": confidence,
                "probabilities": probabilities,
                "branch_contribution": branch_contribution,
                "branch_usage": branch_contribution,
                "branch_evidence": branch_evidence,
                "branch_triangle": branch_triangle,
                "branch_support": branch_support,
                "branch_analysis_mode": "support_weighted_usage",
                "artifacts": {
                    "noise_residual": None,
                    "frequency_spectrum": None,
                    "grad_cam": grad_cam_b64,
                    "grad_cam_overlay": grad_cam_overlay_b64,
                    "fusion_evidence": fusion_evidence_b64,
                },
                "probability": probability,
                "branch_scores": branch_evidence,
                "debug": gradcam_debug if (debug or self.enable_debug) else None,
                "fusion_weights": fusion_weights_list,
                "temperature": self.temperature,
                "scales": self.scales,
                "tta_flip": self.tta_flip,
                "threshold": float(used_threshold),
                "threshold_used": float(used_threshold),
                "threshold_profile": used_profile,
                "analysis_thresholds": dict(self.analysis_thresholds),
                "mode": active_mode,
                "raw_logit": self._tensor_item(logit_tensor),
                "tri_fusion_logit": self._tensor_item(outputs.get("tri_fusion_logit", logit_tensor)),
                "fused_logit": self._tensor_item(outputs.get("fused_logit", logit_tensor)),
                "semantic_score": semantic_prob,
                "frequency_score": frequency_prob,
                "noise_score": noise_prob,
                "semantic_logit": self._tensor_item(semantic_logit),
                "frequency_logit": self._tensor_item(frequency_logit),
                "noise_logit": self._tensor_item(noise_logit),
                "base_logit": self._tensor_item(base_logit),
                "noise_delta_logit": self._tensor_item(noise_delta_logit),
                "alpha": self._tensor_item(alpha_used),
            }
        finally:
            if requested_mode is not None and previous_mode is not None:
                self.model.set_inference_mode(previous_mode)
                self.inference_mode = previous_mode
