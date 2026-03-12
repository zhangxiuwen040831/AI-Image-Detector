
import os
import sys
import base64
import io
import logging
import uvicorn
import uuid
from typing import Any, Dict, Optional
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.detector import ForensicDetector
from backend.transforms import apply_srm_filter, get_spectrum_heatmap
from backend.explanation_service import generate_explanation_report

app = FastAPI(title="AI Image Detector API", description="Forensic detection backend")
logger = logging.getLogger("backend.main")
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
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/best.pth')
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
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
