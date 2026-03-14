import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_image_detector.inference.detector import ForensicDetector
from ai_image_detector.utils import RESNET18_WEIGHTS_PATH


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    device = build_device(args.device)
    print(f"Using device: {device}")

    # Initialize detector
    detector = ForensicDetector(
        model_path=args.model,
        device=device,
        enable_debug=args.debug
    )

    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found at {args.image}")
        return

    # Run inference
    result = detector.predict(str(image_path), debug=args.debug)

    # Print results
    print("===== Detection Result =====")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: REAL={result['probabilities']['real']:.4f}, AIGC={result['probabilities']['aigc']:.4f}")
    
    if result['branch_contribution']:
        print("\nBranch Contributions:")
        for branch, score in result['branch_contribution'].items():
            if score is not None:
                print(f"  {branch}: {score:.4f}")
    
    if args.debug and result.get('debug'):
        print("\nDebug Information:")
        print(f"  Grad-CAM Status: {result['debug'].get('grad_cam_status')}")
        print(f"  Device: {result['debug'].get('device')}")


if __name__ == "__main__":
    main()
