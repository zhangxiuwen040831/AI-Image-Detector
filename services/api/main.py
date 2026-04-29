import os
import sys
import base64
import io
import logging
import uvicorn
from threading import Lock
from typing import Any, Dict, Optional, List
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root and src directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from ai_image_detector.inference import DetectorInterface
from src.data.transforms import apply_srm_filter
from ai_image_detector.utils import get_model_path

# 导入认证模块
from services.api.auth import (
    router as auth_router,
    LoginRequest,
    RegisterRequest,
    login as auth_login,
    logout as auth_logout,
    register as auth_register,
    get_current_user,
    get_current_admin_user,
    get_user_list as auth_get_user_list,
    get_user_logs as auth_get_user_logs,
    ensure_auth_tables,
)

app = FastAPI(title="AI Image Detector API", description="Forensic detection backend")
logger = logging.getLogger("services.api.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

# 注册认证路由
app.include_router(auth_router)
ensure_auth_tables()


@app.post("/login")
async def login(request: LoginRequest, http_request: Request):
    """Compatibility login endpoint backed by MySQL authentication."""
    return await auth_login(request, http_request)


@app.post("/register")
async def register(request: RegisterRequest):
    """Compatibility register endpoint backed by MySQL authentication."""
    return await auth_register(request)


@app.post("/logout")
async def logout(current_user: dict = Depends(get_current_user), request: Request = None):
    """Compatibility logout endpoint that updates last_logout_time."""
    return await auth_logout(current_user, request)


@app.get("/admin/users")
async def admin_users(current_admin: dict = Depends(get_current_admin_user)):
    """Compatibility admin users endpoint."""
    return await auth_get_user_list(current_admin)


@app.get("/admin/logs")
async def admin_logs(
    current_admin: dict = Depends(get_current_admin_user),
    user_id: Optional[int] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    """Compatibility admin logs endpoint."""
    return await auth_get_user_logs(current_admin, user_id=user_id, limit=limit)


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
detector: Optional[DetectorInterface] = None
MODEL_PATH: str = get_model_path()
detector_lock = Lock()


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
        detector = DetectorInterface(MODEL_PATH, enable_debug=ENABLE_GRADCAM_DEBUG)
        logger.info("[Model] loaded=True")
        logger.info("[Model] family=%s", detector.config.model_family)
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
    probability: Optional[float] = None
    label: Optional[str] = None
    label_id: Optional[int] = None
    threshold_used: Optional[float] = None
    threshold_percent: Optional[float] = None
    decision_rule_text: Optional[str] = None
    mode: Optional[str] = None
    semantic_score: Optional[float] = None
    frequency_score: Optional[float] = None
    noise_score: Optional[float] = None
    raw_logit: Optional[float] = None
    logit: Optional[float] = None
    decision_logit: Optional[float] = None
    stable_sf_logit: Optional[float] = None
    tri_fusion_logit: Optional[float] = None
    fused_logit: Optional[float] = None
    semantic_logit: Optional[float] = None
    frequency_logit: Optional[float] = None
    noise_logit: Optional[float] = None
    base_logit: Optional[float] = None
    branch_scores: Optional[Dict[str, Optional[float]]] = None
    srm_image: Optional[str] = None
    spectrum_image: Optional[str] = None
    fusion_evidence_image: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None
    fusion_weights: Optional[Dict[str, Optional[float]]] = None
    decision_weights: Optional[Dict[str, Optional[float]]] = None
    evidence_weights: Optional[Dict[str, Optional[float]]] = None
    noise_enabled_for_decision: Optional[bool] = None
    tri_fusion_enabled_for_decision: Optional[bool] = None
    inference_mode: Optional[str] = None
    temperature: Optional[float] = None
    scales: Optional[List[int]] = None
    tta_flip: Optional[bool] = None
    threshold: Optional[float] = None
    threshold_profile: Optional[str] = None
    analysis_thresholds: Optional[Dict[str, float]] = None


def _to_base64(img_pil):
    if img_pil is None:
        return None
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def _get_spectrum_heatmap_from_image(image: Image.Image):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (224, 224))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    min_val = magnitude_spectrum.min()
    max_val = magnitude_spectrum.max()
    if max_val - min_val > 1e-8:
        magnitude_spectrum = (magnitude_spectrum - min_val) / (max_val - min_val)
    magnitude_spectrum = (magnitude_spectrum * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(magnitude_spectrum, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap)


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _normalize_triplet(value: Any, default: Optional[Dict[str, Optional[float]]] = None) -> Dict[str, Optional[float]]:
    default = default or {"semantic": None, "frequency": None, "noise": None}
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return {
            "semantic": _finite_float(value[0]),
            "frequency": _finite_float(value[1]),
            "noise": _finite_float(value[2]),
        }
    if not isinstance(value, dict):
        return dict(default)
    return {
        "semantic": _finite_float(value.get("semantic", value.get("rgb", default.get("semantic")))),
        "frequency": _finite_float(value.get("frequency", value.get("freq", default.get("frequency")))),
        "noise": _finite_float(value.get("noise", default.get("noise"))),
    }


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
        "probability": 0.0,
        "label": "REAL",
        "label_id": 0,
        "threshold_used": None,
        "threshold_percent": None,
        "decision_rule_text": None,
        "mode": "deploy_safe_tri_branch",
        "semantic_score": None,
        "frequency_score": None,
        "raw_logit": None,
        "branch_scores": {"semantic": None, "noise": None, "frequency": None},
        "srm_image": None,
        "spectrum_image": None,
        "fusion_evidence_image": None,
        "debug": None,
        "fusion_weights": None,
        "decision_weights": None,
        "evidence_weights": None,
        "noise_enabled_for_decision": None,
        "tri_fusion_enabled_for_decision": None,
        "inference_mode": None,
        "temperature": None,
        "scales": None,
        "tta_flip": None,
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
                with detector_lock:
                    model_result = detector.predict(
                        image,
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
            fusion_evidence_b64 = None

            noise_residual_b64 = None
            frequency_spectrum_b64 = None

            try:
                import numpy as np
                from ai_image_detector.ntire.augmentations import build_eval_transform
                transform = build_eval_transform(image_size=detector.config.image_size)
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
                spectrum_pil = _get_spectrum_heatmap_from_image(image)
                frequency_spectrum_b64 = _to_base64(spectrum_pil)
            except Exception as e:
                debug_payload["frequency_spectrum_error"] = str(e)
                logger.exception("frequency_spectrum_generation_failed error=%s", e)

            prediction = str(model_result.get("prediction", "REAL")).upper()
            confidence = float(model_result.get("confidence", 0.0))
            probabilities = model_result.get("probabilities") or {}
            mode = str(model_result.get("mode") or "deploy_safe_tri_branch")
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

            branch_contribution = _normalize_triplet(model_result.get("branch_contribution") or model_result.get("decision_weights"))
            branch_evidence = model_result.get("branch_evidence") or model_result.get("branch_scores") or branch_contribution
            branch_evidence = _normalize_triplet(branch_evidence)
            branch_triangle = model_result.get("branch_triangle") or branch_evidence
            branch_triangle = _normalize_triplet(branch_triangle)
            branch_usage = model_result.get("branch_usage") or branch_contribution
            branch_usage = _normalize_triplet(branch_usage)
            branch_support = model_result.get("branch_support") or {
                "semantic": None,
                "noise": None,
                "frequency": None
            }
            branch_support = _normalize_triplet(branch_support)
            branch_analysis_mode = model_result.get("branch_analysis_mode") or "support_weighted_usage"

            enabled_branches = [name for name, value in branch_contribution.items() if value is not None]
            if not enabled_branches:
                enabled_branches = ["semantic", "noise", "frequency"]

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
            elif isinstance(fusion_weights_list, dict):
                fusion_weights_dict = _normalize_triplet(fusion_weights_list)

            decision_weights = _normalize_triplet(model_result.get("decision_weights"))
            evidence_weights = _normalize_triplet(model_result.get("evidence_weights") or fusion_weights_dict)

            model_debug = model_result.get("debug") if isinstance(model_result.get("debug"), dict) else {}
            debug_payload.update(model_debug)
            debug_payload["grad_cam_status"] = "success" if grad_cam_b64 else ("not_supported" if detector.config.model_family in {"ntire", "v10"} else "failed")
            debug_payload["overlay_status"] = "success" if grad_cam_overlay_b64 else ("not_supported" if detector.config.model_family in {"ntire", "v10"} else "failed")

            artifacts_values = inference_result["artifacts"].values()
            debug_payload["artifacts_null"] = all(v is None for v in artifacts_values)
            debug_payload["grad_cam_length"] = len(grad_cam_b64) if grad_cam_b64 else 0
            debug_payload["grad_cam_overlay_length"] = len(grad_cam_overlay_b64) if grad_cam_overlay_b64 else 0

            logger.info(
                "detect_finished prediction=%s confidence=%.6f",
                prediction,
                confidence,
            )

            response = {
                **inference_result,
                "probability": inference_result["probabilities"]["aigc"],
                "threshold_used": model_result.get("threshold_used", model_result.get("threshold")),
                "threshold_percent": model_result.get("threshold_percent"),
                "decision_rule_text": model_result.get("decision_rule_text"),
                "branch_scores": _normalize_triplet(model_result.get("branch_scores") or inference_result["branch_evidence"]),
                "srm_image": inference_result["artifacts"]["noise_residual"],
                "spectrum_image": inference_result["artifacts"]["frequency_spectrum"],
                "fusion_evidence_image": inference_result["artifacts"]["fusion_evidence"],
                "debug": debug_payload if (debug_enabled or not grad_cam_b64 or not grad_cam_overlay_b64) else None,
                "fusion_weights": fusion_weights_dict,
                "decision_weights": decision_weights,
                "evidence_weights": evidence_weights,
                "temperature": temperature,
                "scales": scales,
                "tta_flip": tta_flip,
                "threshold": model_result.get("threshold"),
                "threshold_profile": model_result.get("threshold_profile"),
                "analysis_thresholds": model_result.get("analysis_thresholds"),
                "mode": mode,
                "semantic_score": model_result.get("semantic_score"),
                "frequency_score": model_result.get("frequency_score"),
                "noise_score": model_result.get("noise_score"),
                "raw_logit": model_result.get("raw_logit"),
                "logit": model_result.get("raw_logit"),
                "decision_logit": model_result.get("decision_logit", model_result.get("raw_logit")),
                "stable_sf_logit": model_result.get("stable_sf_logit"),
                "tri_fusion_logit": model_result.get("tri_fusion_logit", model_result.get("raw_logit")),
                "fused_logit": model_result.get("fused_logit", model_result.get("raw_logit")),
                "semantic_logit": model_result.get("semantic_logit"),
                "frequency_logit": model_result.get("frequency_logit"),
                "noise_logit": model_result.get("noise_logit"),
                "base_logit": model_result.get("base_logit"),
                "noise_enabled_for_decision": model_result.get("noise_enabled_for_decision"),
                "tri_fusion_enabled_for_decision": model_result.get("tri_fusion_enabled_for_decision"),
                "inference_mode": model_result.get("inference_mode", mode),
            }

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("detect_failed filename=%s error=%s", file.filename, e)
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    finally:
        pass


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok" if detector is not None else "error",
        "model_loaded": detector is not None,
        "model_path": "checkpoints/best.pth",
        "model_family": detector.config.model_family if detector else None,
        "mode": "deploy_safe_tri_branch" if detector else None,
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
        "model_family": detector.config.model_family,
        "mode": "deploy_safe_tri_branch",
        "available_modes": detector.get_available_inference_modes(),
        "device": str(detector.device),
        "input_size": detector.config.image_size,
        "supports_gradcam": detector.config.model_family == "legacy",
        "scales": detector.config.scales,
        "temperature": detector.config.temperature,
        "threshold": detector.config.threshold,
        "threshold_profile": detector.config.threshold_profile,
        "thresholds": detector.config_manager.get_threshold_config(),
        "analysis_thresholds": detector.config_manager.get_analysis_thresholds(),
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
