
import os
import sys
import base64
import io
import uvicorn
import uuid
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.detector import ForensicDetector
from backend.transforms import apply_srm_filter, get_spectrum_heatmap

app = FastAPI(title="AI Image Detector API", description="Forensic detection backend")

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
    print(f"Warning: Model not found at {MODEL_PATH}")
    # Fallback for dev without model
    detector = None
else:
    try:
        detector = ForensicDetector(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        detector = None

class DetectionResponse(BaseModel):
    prediction: str
    probability: float
    confidence: float
    branch_scores: dict
    srm_image: str  # Base64
    spectrum_image: str  # Base64

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Save temporary file for transforms that need path or just process in memory
    # detector.predict expects path or PIL image? 
    # Let's check backend/detector.py. It takes image_path. 
    # I should modify it to take PIL Image or handle bytes.
    # Or save to temp file.
    
    temp_path = f"temp_{uuid.uuid4()}.png"
    image.save(temp_path)
    
    try:
        # Run inference
        result = detector.predict(temp_path)
        
        # Generate forensic visualizations
        # 1. SRM Filter
        # srm_map is numpy array [H, W]
        img_tensor = detector.transforms(image).unsqueeze(0)
        srm_map = apply_srm_filter(img_tensor)
        srm_pil = Image.fromarray((srm_map * 255).astype('uint8'))
        
        # 2. Spectrum
        spectrum_pil = get_spectrum_heatmap(temp_path)
        
        # Convert to Base64
        def to_base64(img_pil):
            if img_pil is None: return ""
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
        result['srm_image'] = to_base64(srm_pil)
        result['spectrum_image'] = to_base64(spectrum_pil)
        
        return result
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
