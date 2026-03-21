import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
import hashlib
import os

MODEL_NAME = "facebook/convnextv2-base-22k-384"
SAVE_DIR = "/root/lanyun-tmp/ai-image-detector/models/convnextv2-base"

def download_and_verify():
    print(f"Downloading model {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)
    
    print(f"Model saved to {SAVE_DIR}")
    
    # Simple verification by listing files
    for f in os.listdir(SAVE_DIR):
        fpath = os.path.join(SAVE_DIR, f)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as file:
                h = hashlib.sha256(file.read()).hexdigest()
                print(f"File: {f}, SHA256: {h}")

if __name__ == "__main__":
    download_and_verify()
