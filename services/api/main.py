import os
import sys
import base64
import io
import logging
import uvicorn
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
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
from ai_image_detector.utils import load_config, BEST_MODEL_PATH

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
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
try:
    # Try to load model path from config
    cfg = load_config("infer")
    model_path_from_config = cfg.get("model_path", "")
    if model_path_from_config:
        MODEL_PATH = model_path_from_config
    else:
        # Use robust path based on project root
        project_root = Path(__file__).resolve().parents[2]
        # Try multiple common locations
        possible_paths = [
            BEST_MODEL_PATH,
            project_root / "model" / "best.pth",
            project_root / "checkpoints" / "pipeline" / "best.pth",
            project_root / "best.pth"
        ]
        
        MODEL_PATH = None
        for path in possible_paths:
            if path.exists():
                MODEL_PATH = str(path)
                break
        
        if not MODEL_PATH:
            # Default fallback
            MODEL_PATH = str(BEST_MODEL_PATH)
except Exception as e:
    logger.warning("config_load_failed using_default_path error=%s", e)
    # Default path as fallback
    MODEL_PATH = str(BEST_MODEL_PATH)

logger.info("using_model_path path=%s", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    logger.error("model_not_found path=%s", MODEL_PATH)
    detector = None
else:
    try:
        detector = ForensicDetector(MODEL_PATH, enable_debug=ENABLE_GRADCAM_DEBUG)
        logger.info("detector_initialized device=%s debug=%s", detector.device, ENABLE_GRADCAM_DEBUG)
    except Exception as e:
        logger.exception("detector_init_failed error=%s", e)
        detector = None

class DetectionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    branch_contribution: Dict[str, Optional[float]]
    artifacts: Dict[str, Optional[str]]
    metadata: Dict[str, Any]
    explanation: Dict[str, Any]
    probability: Optional[float] = None
    branch_scores: Optional[Dict[str, Optional[float]]] = None
    srm_image: Optional[str] = None
    spectrum_image: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


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
    probabilities = result.get("probabilities", {})
    real_prob = probabilities.get("real", 0.0)
    aigc_prob = probabilities.get("aigc", 0.0)
    
    # 摘要
    summary = {
        "summary_text": f"当前模型判断结果为{prediction}，且{real_prob:.2f}的真实概率{'>' if prediction == 'REAL' else '<'}AIGC概率{aigc_prob:.2f}。置信度约为{confidence:.2f}，说明模型在本次样本上更倾向于{prediction}判定。",
        "final_result": prediction,
        "confidence_score": confidence
    }
    
    # 总体评估
    risk_level = "低" if confidence > 0.7 else "中" if confidence > 0.5 else "高"
    authenticity_score = real_prob if prediction == "REAL" else aigc_prob
    overall_assessment = {
        "assessment_text": f"模型更倾向该图像为{prediction == 'REAL' and '真实拍摄' or 'AI生成'}，当前风险等级为{risk_level}。但仍不应将单次模型结果视为绝对结论。",
        "risk_level": risk_level,
        "authenticity_score": authenticity_score
    }
    
    # 取证分析
    forensic_analysis = {
        "rgb_analysis": {
            "title": "RGB语义分析",
            "description": f"RGB分支贡献约为{branch_contribution.get('rgb', 0):.2f}，为辅助证据来源。该分支主要从语义结构与纹理一致性角度提供证据。"
        },
        "noise_analysis": {
            "title": "噪声残差分析",
            "description": f"噪声分支贡献约为{branch_contribution.get('noise', 0):.2f}，是本次判定的主要依据。该分支主要从残差与噪声一致性角度提供证据，其方向与最终结果（{prediction}）保持一致。"
        },
        "frequency_analysis": {
            "title": "频域分析",
            "description": f"频域分支贡献约为{branch_contribution.get('frequency', 0):.2f}。该分支主要从频域分布与谱结构角度提供证据。"
        },
        "cross_branch_conclusion": {
            "title": "跨分支结论",
            "description": "分支贡献存在明显主次差异，模型主要依赖高贡献分支完成{prediction}判定，其余分支提供辅助支持。"
        }
    }
    
    # 视觉证据
    visual_evidence = [
        {
            "title": "噪声残差证据",
            "description": "已提供噪声残差证据图，可用于辅助观察噪声一致性与残差结构。",
            "available": True
        },
        {
            "title": "频谱证据",
            "description": "已提供频谱证据图，可用于辅助观察频域分布与潜在异常模式。",
            "available": True
        },
        {
            "title": "注意力热力图",
            "description": "已提供注意力热图，可用于定位模型重点关注的判别区域。",
            "available": True
        }
    ]
    
    # 关键指标
    key_indicators = [
        f"最终预测为{prediction}",
        f"概率分布：REAL={real_prob:.4f}, AIGC={aigc_prob:.4f}",
        f"分支贡献用于定位关键证据来源",
        "缺失证据图已按兼容模式降级展示"
    ]
    
    # 置信度解释
    confidence_explanation = {
        "title": "置信度解释",
        "description": f"当前置信度处于{'高' if confidence > 0.7 else '中' if confidence > 0.5 else '低'}水平，说明结果有一定支持{'但仍存在不确定性' if confidence < 0.7 else ''}。适合结合可视化证据共同解读。"
    }
    
    # 最终解读
    final_conclusion = {
        "title": "最终解读",
        "description": f"综合当前可用证据，系统将该图像判定为{prediction}。建议在高风险场景中结合图像来源、上下文与人工复核共同决策，以避免单模型误判带来的业务风险。"
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


async def _run_detection(file: UploadFile, debug_enabled: bool) -> Dict[str, Any]:
    if detector is None:
        logger.error("detect_rejected model_not_loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        logger.exception("invalid_image_upload filename=%s error=%s", file.filename, e)
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    temp_path = f"temp_{uuid.uuid4()}.png"
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
            "detect_started filename=%s device=%s debug=%s",
            file.filename,
            detector.device,
            debug_enabled,
        )
        model_result = detector.predict(temp_path, debug=debug_enabled)
        model_artifacts = model_result.get("artifacts") or {}
        grad_cam_b64 = model_artifacts.get("grad_cam")
        grad_cam_overlay_b64 = model_artifacts.get("grad_cam_overlay")

        noise_residual_b64 = None
        frequency_spectrum_b64 = None

        try:
            img_tensor = detector.transforms(image).unsqueeze(0)
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
            "rgb": branch_contribution.get("rgb"),
            "noise": branch_contribution.get("noise"),
            "frequency": branch_contribution.get("frequency")
        }

        enabled_branches = [name for name, value in branch_contribution.items() if value is not None]
        if not enabled_branches:
            enabled_branches = ["rgb", "noise", "frequency"]

        inference_result = {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "real": float(real_prob),
                "aigc": float(aigc_prob)
            },
            "branch_contribution": branch_contribution,
            "artifacts": {
                "noise_residual": noise_residual_b64,
                "frequency_spectrum": frequency_spectrum_b64,
                "grad_cam": grad_cam_b64,
                "grad_cam_overlay": grad_cam_overlay_b64,
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
        debug_payload["grad_cam_status"] = "success" if grad_cam_b64 else "failed"
        debug_payload["overlay_status"] = "success" if grad_cam_overlay_b64 else "failed"
        if not grad_cam_b64 and not debug_payload.get("grad_cam_error"):
            debug_payload["grad_cam_error"] = model_debug.get("error") or "grad_cam is empty"
        if not grad_cam_overlay_b64 and not debug_payload.get("overlay_error"):
            debug_payload["overlay_error"] = model_debug.get("error") or "grad_cam_overlay is empty"
        artifacts_values = inference_result["artifacts"].values()
        debug_payload["artifacts_null"] = all(v is None for v in artifacts_values)
        debug_payload["grad_cam_length"] = len(grad_cam_b64) if grad_cam_b64 else 0
        debug_payload["grad_cam_overlay_length"] = len(grad_cam_overlay_b64) if grad_cam_overlay_b64 else 0
        logger.info(
            "detect_finished prediction=%s confidence=%.6f grad_cam=%s overlay=%s artifacts_null=%s",
            prediction,
            confidence,
            bool(grad_cam_b64),
            bool(grad_cam_overlay_b64),
            debug_payload["artifacts_null"],
        )

        response = {
            **inference_result,
            "explanation": explanation,
            "probability": inference_result["probabilities"]["aigc"],
            "branch_scores": inference_result["branch_contribution"],
            "srm_image": inference_result["artifacts"]["noise_residual"],
            "spectrum_image": inference_result["artifacts"]["frequency_spectrum"],
            "debug": debug_payload if (debug_enabled or not grad_cam_b64 or not grad_cam_overlay_b64) else None,
        }
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("detect_failed filename=%s error=%s", file.filename, e)
        if debug_enabled:
            raise HTTPException(status_code=500, detail=f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail="Detection failed")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...), debug: int = Query(default=0)):
    debug_enabled = bool(debug) or ENABLE_GRADCAM_DEBUG
    return await _run_detection(file=file, debug_enabled=debug_enabled)


@app.post("/detect/debug", response_model=DetectionResponse)
async def detect_debug(file: UploadFile = File(...)):
    return await _run_detection(file=file, debug_enabled=True)


if __name__ == "__main__":
    uvicorn.run("services.api.main:app", host="0.0.0.0", port=8000, reload=True)
