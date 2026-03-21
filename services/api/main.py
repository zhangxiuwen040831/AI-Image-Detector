import os
import sys
import base64
import io
import logging
import uvicorn
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, List
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root and src directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from ai_image_detector.inference.detector import ForensicDetector
from src.data.transforms import apply_srm_filter, get_spectrum_heatmap
from ai_image_detector.utils import load_config, BEST_MODEL_PATH, get_model_path

app = FastAPI(title="AI Image Detector API", description="Forensic detection backend")
logger = logging.getLogger("services.api.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)


def _env_flag(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


ENABLE_GRADCAM_DEBUG = _env_flag("ENABLE_GRADCAM_DEBUG", "false")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model state
detector: Optional[ForensicDetector] = None
MODEL_PATH: str = get_model_path()


def initialize_detector():
    """Initialize detector - single model file mode"""
    global detector
    
    logger.info("[Model] path=%s", MODEL_PATH)
    
    if not os.path.exists(MODEL_PATH):
        logger.error("[Model] loaded=False")
        logger.error("[Model] missing=%s", MODEL_PATH)
        detector = None
        return
    
    try:
        detector = ForensicDetector(MODEL_PATH, enable_debug=ENABLE_GRADCAM_DEBUG)
        logger.info("[Model] loaded=True")
        logger.info("[Model] family=%s", detector.model_family)
        logger.info("[Model] device=%s", detector.device)
    except Exception as e:
        logger.exception("[Model] loaded=False error=%s", e)
        detector = None


# Initialize on startup
initialize_detector()


class DetectionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    branch_contribution: Dict[str, Optional[float]]
    branch_evidence: Optional[Dict[str, Optional[float]]] = None
    branch_triangle: Optional[Dict[str, Optional[float]]] = None
    branch_usage: Optional[Dict[str, Optional[float]]] = None
    branch_support: Optional[Dict[str, Optional[float]]] = None
    branch_analysis_mode: Optional[str] = None
    artifacts: Dict[str, Optional[str]]
    metadata: Dict[str, Any]
    explanation: Dict[str, Any]
    probability: Optional[float] = None
    label: Optional[str] = None
    label_id: Optional[int] = None
    threshold_used: Optional[float] = None
    mode: Optional[str] = None
    semantic_score: Optional[float] = None
    frequency_score: Optional[float] = None
    raw_logit: Optional[float] = None
    branch_scores: Optional[Dict[str, Optional[float]]] = None
    srm_image: Optional[str] = None
    spectrum_image: Optional[str] = None
    fusion_evidence_image: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None
    fusion_weights: Optional[Dict[str, Optional[float]]] = None
    temperature: Optional[float] = None
    scales: Optional[List[int]] = None
    tta_flip: Optional[bool] = None
    threshold: Optional[float] = None
    threshold_profile: Optional[str] = None


def _to_base64(img_pil):
    if img_pil is None:
        return None
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def generate_explanation_report(result):
    """Generate explanation report based on detection result"""
    prediction = result.get("prediction", "REAL")
    confidence = result.get("confidence", 0.0)
    branch_contribution = result.get("branch_contribution", {})
    branch_evidence = result.get("branch_evidence") or branch_contribution
    probabilities = result.get("probabilities", {})
    mode = str(result.get("mode", "base_only"))
    semantic_value = float(branch_evidence.get("rgb") or 0.0)
    noise_value = float(branch_evidence.get("noise") or 0.0)
    frequency_value = float(branch_evidence.get("frequency") or 0.0)
    real_prob = probabilities.get("real", 0.0)
    aigc_prob = probabilities.get("aigc", 0.0)
    
    summary = {
        "summary_text": f"当前模型判断结果为{prediction}，且{real_prob:.2f}的真实概率{'>' if prediction == 'REAL' else '<'}AIGC概率{aigc_prob:.2f}。置信度约为{confidence:.2f}，说明模型在本次样本上更倾向于{prediction}判定。",
        "final_result": prediction,
        "confidence_score": confidence
    }
    
    risk_level = "低" if confidence > 0.7 else "中" if confidence > 0.5 else "高"
    authenticity_score = real_prob if prediction == "REAL" else aigc_prob
    overall_assessment = {
        "assessment_text": f"模型更倾向该图像为{prediction == 'REAL' and '真实拍摄' or 'AI生成'}，当前风险等级为{risk_level}。但仍不应将单次模型结果视为绝对结论。",
        "risk_level": risk_level,
        "authenticity_score": authenticity_score
    }
    
    forensic_analysis = {
        "rgb_analysis": {
            "title": "全局语义分析",
            "description": f"语义主分支证据强度约为{semantic_value:.2f}，用于提供图像内容、高层语义一致性与主体结构证据。"
        },
        "noise_analysis": {
            "title": "噪声辅助分析",
            "description": (
                f"噪声分支证据强度约为{noise_value:.2f}。"
                + ("当前部署模式为 base_only，噪声分支仅作为诊断参考，不参与最终决策。" if mode == "base_only" else "在可选 hybrid 模式下，噪声分支仅作为辅助残差证据。")
            )
        },
        "frequency_analysis": {
            "title": "频域伪迹分析",
            "description": f"频域主分支证据强度约为{frequency_value:.2f}，用于提供频域分布、压缩痕迹与谱结构证据。"
        },
        "cross_branch_conclusion": {
            "title": "主路径融合决策",
            "description": (
                "当前最终模型以 semantic + frequency 为主判别路径。"
                + ("最终结果直接由 base_only 路径给出，重点降低真实图误报。" if mode == "base_only" else "noise 分支仅在 hybrid_optional 模式下提供有限辅助。")
            )
        }
    }
    
    visual_evidence = [
        {
            "title": "噪声残差证据",
            "description": "已提供噪声残差证据图，可用于辅助观察噪声一致性与残差结构，但默认不直接主导最终分类。",
            "available": branch_contribution.get("noise") is not None
        },
        {
            "title": "频谱证据",
            "description": "已提供频谱证据图，可用于辅助观察频域分布与潜在异常模式。",
            "available": branch_contribution.get("frequency") is not None
        },
        {
            "title": "融合证据三角图",
            "description": "已提供融合证据三角图，用于展示语义、频域、噪声三路证据在融合决策中的相对权重。",
            "available": True
        }
    ]
    
    key_indicators = [
        f"最终预测为{prediction}",
        f"概率分布：REAL={real_prob:.4f}, AIGC={aigc_prob:.4f}",
        f"分支贡献用于定位关键证据来源",
        "缺失证据图已按兼容模式降级展示"
    ]
    
    confidence_explanation = {
        "title": "置信度解释",
        "description": f"当前置信度处于{'高' if confidence > 0.7 else '中' if confidence > 0.5 else '低'}水平，说明结果有一定支持{'但仍存在不确定性' if confidence < 0.7 else ''}。"
    }
    
    final_conclusion = {
        "title": "最终解读",
        "description": f"综合当前可用证据，系统将该图像判定为{prediction}。建议在高风险场景中结合图像来源、上下文与人工复核共同决策。"
    }
    
    explanation = {
        "report_title": "AI图像取证报告",
        "prediction": prediction,
        "confidence": confidence,
        "summary": summary,
        "overall_assessment": overall_assessment,
        "forensic_analysis": forensic_analysis,
        "visual_evidence": visual_evidence,
        "key_indicators": key_indicators,
        "confidence_explanation": confidence_explanation,
        "user_friendly_conclusion": final_conclusion
    }
    
    return explanation


def create_default_response() -> Dict[str, Any]:
    """Create a default response with all required fields"""
    return {
        "prediction": "REAL",
        "confidence": 0.0,
        "probabilities": {"real": 1.0, "aigc": 0.0},
        "branch_contribution": {"rgb": None, "noise": None, "frequency": None},
        "branch_evidence": {"rgb": None, "noise": None, "frequency": None},
        "branch_triangle": {"rgb": None, "noise": None, "frequency": None},
        "branch_usage": {"rgb": None, "noise": None, "frequency": None},
        "branch_support": {"rgb": None, "noise": None, "frequency": None},
        "branch_analysis_mode": "support_weighted_usage",
        "artifacts": {
            "noise_residual": None,
            "frequency_spectrum": None,
            "grad_cam": None,
            "grad_cam_overlay": None,
            "fusion_evidence": None
        },
        "metadata": {},
        "explanation": {},
        "probability": 0.0,
        "label": "REAL",
        "label_id": 0,
        "threshold_used": None,
        "mode": "base_only",
        "semantic_score": None,
        "frequency_score": None,
        "raw_logit": None,
        "branch_scores": {"rgb": None, "noise": None, "frequency": None},
        "srm_image": None,
        "spectrum_image": None,
        "fusion_evidence_image": None,
        "debug": None,
        "fusion_weights": None,
        "temperature": None,
        "scales": None,
        "tta_flip": None
    }


async def _run_detection(
    file: UploadFile,
    debug_enabled: bool,
    threshold: Optional[float] = None,
    threshold_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Run detection with comprehensive error handling"""
    if detector is None:
        error_msg = "Model not loaded. Expected file at checkpoints/best.pth"
        logger.error("detect_rejected model_not_loaded")
        raise HTTPException(status_code=503, detail=error_msg)

    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    temp_path = None
    
    try:
        try:
            contents = await file.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            
            image = Image.open(io.BytesIO(contents))
            image = image.convert('RGB')
        except Image.UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid JPG, PNG, or WEBP image.")
        except Exception as e:
            logger.exception("invalid_image_upload filename=%s error=%s", file.filename, e)
            raise HTTPException(status_code=400, detail=f"Cannot process image: {str(e)}")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix='detect_')
        temp_path = temp_file.name
        temp_file.close()
        
        image.save(temp_path)
        
        debug_payload: Dict[str, Any] = {
            "device": str(detector.device),
            "model_loaded": detector is not None,
            "filename": file.filename or "uploaded_image",
            "debug_enabled": bool(debug_enabled),
            "grad_cam_status": "not_started",
            "grad_cam_error": None,
            "overlay_status": "not_started",
            "overlay_error": None,
            "artifacts_null": None,
        }

        try:
            logger.info(
                "detect_started filename=%s device=%s",
                file.filename,
                detector.device,
            )
            
            try:
                model_result = detector.predict(
                    temp_path,
                    debug=debug_enabled,
                    threshold=threshold,
                    threshold_profile=threshold_profile,
                )
            except Exception as e:
                logger.exception("model_inference_failed filename=%s error=%s", file.filename, e)
                raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
            
            if not isinstance(model_result, dict):
                logger.error("model_result_not_dict type=%s", type(model_result))
                model_result = create_default_response()
            
            model_artifacts = model_result.get("artifacts") or {}
            grad_cam_b64 = model_artifacts.get("grad_cam")
            grad_cam_overlay_b64 = model_artifacts.get("grad_cam_overlay")
            fusion_evidence_b64 = model_artifacts.get("fusion_evidence")

            noise_residual_b64 = None
            frequency_spectrum_b64 = None

            try:
                import numpy as np
                from ai_image_detector.ntire.augmentations import build_eval_transform
                transform = build_eval_transform(image_size=detector.image_size)
                arr = np.array(image.convert("RGB"))
                transformed = transform(image=arr)
                img_tensor = transformed["image"].unsqueeze(0)
                srm_map = apply_srm_filter(img_tensor)
                srm_pil = Image.fromarray((srm_map * 255).astype('uint8'))
                noise_residual_b64 = _to_base64(srm_pil)
            except Exception as e:
                debug_payload["noise_residual_error"] = str(e)
                logger.exception("noise_residual_generation_failed error=%s", e)

            try:
                spectrum_pil = get_spectrum_heatmap(temp_path)
                frequency_spectrum_b64 = _to_base64(spectrum_pil)
            except Exception as e:
                debug_payload["frequency_spectrum_error"] = str(e)
                logger.exception("frequency_spectrum_generation_failed error=%s", e)

            prediction = str(model_result.get("prediction", "REAL")).upper()
            confidence = float(model_result.get("confidence", 0.0))
            probabilities = model_result.get("probabilities") or {}
            mode = str(model_result.get("mode") or "base_only")
            real_prob = probabilities.get("real")
            aigc_prob = probabilities.get("aigc")
            
            if real_prob is None or aigc_prob is None:
                legacy_probability = float(model_result.get("probability", 0.5))
                if prediction == "AIGC":
                    aigc_prob = legacy_probability
                    real_prob = 1 - legacy_probability
                else:
                    real_prob = legacy_probability
                    aigc_prob = 1 - legacy_probability

            branch_contribution = model_result.get("branch_contribution") or model_result.get("branch_scores") or {
                "rgb": None,
                "noise": None,
                "frequency": None
            }
            branch_contribution = {
                "rgb": branch_contribution.get("rgb") if branch_contribution.get("rgb") is not None else None,
                "noise": branch_contribution.get("noise") if branch_contribution.get("noise") is not None else None,
                "frequency": branch_contribution.get("frequency") if branch_contribution.get("frequency") is not None else None
            }
            branch_evidence = model_result.get("branch_evidence") or model_result.get("branch_scores") or branch_contribution
            branch_evidence = {
                "rgb": branch_evidence.get("rgb") if branch_evidence.get("rgb") is not None else None,
                "noise": branch_evidence.get("noise") if branch_evidence.get("noise") is not None else None,
                "frequency": branch_evidence.get("frequency") if branch_evidence.get("frequency") is not None else None
            }
            branch_triangle = model_result.get("branch_triangle") or branch_evidence
            branch_triangle = {
                "rgb": branch_triangle.get("rgb") if branch_triangle.get("rgb") is not None else None,
                "noise": branch_triangle.get("noise") if branch_triangle.get("noise") is not None else None,
                "frequency": branch_triangle.get("frequency") if branch_triangle.get("frequency") is not None else None
            }
            branch_usage = model_result.get("branch_usage") or branch_contribution
            branch_usage = {
                "rgb": branch_usage.get("rgb") if branch_usage.get("rgb") is not None else None,
                "noise": branch_usage.get("noise") if branch_usage.get("noise") is not None else None,
                "frequency": branch_usage.get("frequency") if branch_usage.get("frequency") is not None else None
            }
            branch_support = model_result.get("branch_support") or {
                "rgb": None,
                "noise": None,
                "frequency": None
            }
            branch_support = {
                "rgb": branch_support.get("rgb") if branch_support.get("rgb") is not None else None,
                "noise": branch_support.get("noise") if branch_support.get("noise") is not None else None,
                "frequency": branch_support.get("frequency") if branch_support.get("frequency") is not None else None
            }
            branch_analysis_mode = model_result.get("branch_analysis_mode") or "support_weighted_usage"

            enabled_branches = [name for name, value in branch_contribution.items() if value is not None]
            if not enabled_branches:
                enabled_branches = ["rgb", "noise", "frequency"]

            inference_result = {
                "prediction": prediction,
                "label": model_result.get("label", prediction),
                "label_id": model_result.get("label_id", 1 if prediction == "AIGC" else 0),
                "confidence": confidence,
                "probabilities": {
                    "real": float(real_prob),
                    "aigc": float(aigc_prob)
                },
                "branch_contribution": branch_contribution,
                "branch_evidence": branch_evidence,
                "branch_triangle": branch_triangle,
                "branch_usage": branch_usage,
                "branch_support": branch_support,
                "branch_analysis_mode": branch_analysis_mode,
                "mode": mode,
                "artifacts": {
                    "noise_residual": noise_residual_b64,
                    "frequency_spectrum": frequency_spectrum_b64,
                    "grad_cam": grad_cam_b64,
                    "grad_cam_overlay": grad_cam_overlay_b64,
                    "fusion_evidence": fusion_evidence_b64,
                },
                "metadata": {
                    "enabled_branches": enabled_branches,
                    "image_width": image.width,
                    "image_height": image.height,
                    "filename": file.filename or "uploaded_image"
                }
            }

            explanation = generate_explanation_report(inference_result)
            
            model_debug = model_result.get("debug") if isinstance(model_result.get("debug"), dict) else {}
            debug_payload.update(model_debug)
            debug_payload["grad_cam_status"] = "success" if grad_cam_b64 else ("not_supported" if detector.model_family == "ntire" else "failed")
            debug_payload["overlay_status"] = "success" if grad_cam_overlay_b64 else ("not_supported" if detector.model_family == "ntire" else "failed")
            
            artifacts_values = inference_result["artifacts"].values()
            debug_payload["artifacts_null"] = all(v is None for v in artifacts_values)
            debug_payload["grad_cam_length"] = len(grad_cam_b64) if grad_cam_b64 else 0
            debug_payload["grad_cam_overlay_length"] = len(grad_cam_overlay_b64) if grad_cam_overlay_b64 else 0
            
            logger.info(
                "detect_finished prediction=%s confidence=%.6f",
                prediction,
                confidence,
            )

            fusion_weights_list = model_result.get("fusion_weights")
            temperature = model_result.get("temperature")
            scales = model_result.get("scales")
            tta_flip = model_result.get("tta_flip")
            
            fusion_weights_dict = None
            if fusion_weights_list and isinstance(fusion_weights_list, list) and len(fusion_weights_list) >= 3:
                fusion_weights_dict = {
                    "semantic": fusion_weights_list[0],
                    "frequency": fusion_weights_list[1],
                    "noise": fusion_weights_list[2],
                }
            
            response = {
                **inference_result,
                "explanation": explanation,
                "probability": inference_result["probabilities"]["aigc"],
                "threshold_used": model_result.get("threshold_used", model_result.get("threshold")),
                "branch_scores": inference_result["branch_evidence"],
                "srm_image": inference_result["artifacts"]["noise_residual"],
                "spectrum_image": inference_result["artifacts"]["frequency_spectrum"],
                "fusion_evidence_image": inference_result["artifacts"]["fusion_evidence"],
                "debug": debug_payload if (debug_enabled or not grad_cam_b64 or not grad_cam_overlay_b64) else None,
                "fusion_weights": fusion_weights_dict,
                "temperature": temperature,
                "scales": scales,
                "tta_flip": tta_flip,
                "threshold": model_result.get("threshold"),
                "threshold_profile": model_result.get("threshold_profile"),
                "mode": mode,
                "semantic_score": model_result.get("semantic_score"),
                "frequency_score": model_result.get("frequency_score"),
                "raw_logit": model_result.get("raw_logit"),
            }
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("detect_failed filename=%s error=%s", file.filename, e)
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
            
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning("failed_to_remove_temp_file path=%s error=%s", temp_path, e)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok" if detector is not None else "error",
        "model_loaded": detector is not None,
        "model_path": "checkpoints/best.pth",
        "model_family": getattr(detector, "model_family", None) if detector else None,
        "mode": getattr(detector, "inference_mode", None) if detector else None,
        "device": str(detector.device) if detector else None,
    }


@app.get("/model/info")
async def model_info():
    """Model information endpoint"""
    if detector is None:
        return {
            "loaded": False,
            "model_path": "checkpoints/best.pth",
            "error": "Model file not found"
        }
    
    return {
        "loaded": True,
        "model_path": "checkpoints/best.pth",
        "model_family": getattr(detector, "model_family", "unknown"),
        "mode": getattr(detector, "inference_mode", None),
        "device": str(detector.device),
        "input_size": getattr(detector, "image_size", 224),
        "supports_gradcam": getattr(detector, "model_family", "") == "legacy",
        "scales": getattr(detector, "scales", [224]),
        "temperature": getattr(detector, "temperature", 1.0),
        "threshold": getattr(detector, "threshold", None),
        "threshold_profile": getattr(detector, "threshold_profile", None),
        "thresholds": getattr(detector, "thresholds", None),
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    debug: int = Query(default=0),
    threshold: Optional[float] = Query(default=None),
    threshold_profile: Optional[str] = Query(default=None),
):
    debug_enabled = bool(debug) or ENABLE_GRADCAM_DEBUG
    return await _run_detection(
        file=file,
        debug_enabled=debug_enabled,
        threshold=threshold,
        threshold_profile=threshold_profile,
    )


@app.post("/detect/debug", response_model=DetectionResponse)
async def detect_debug(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(default=None),
    threshold_profile: Optional[str] = Query(default=None),
):
    return await _run_detection(
        file=file,
        debug_enabled=True,
        threshold=threshold,
        threshold_profile=threshold_profile,
    )


if __name__ == "__main__":
    uvicorn.run("services.api.main:app", host="0.0.0.0", port=8000, reload=True)
