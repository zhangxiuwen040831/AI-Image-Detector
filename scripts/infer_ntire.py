from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.inference.detector import ForensicDetector


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def predict_single(detector: ForensicDetector, image_path: Path, threshold: float | None, threshold_profile: str) -> Dict[str, object]:
    result = detector.predict(
        str(image_path),
        threshold=threshold,
        threshold_profile=threshold_profile,
    )
    return {
        "path": str(image_path),
        "prediction": result.get("prediction"),
        "label": result.get("label"),
        "label_id": result.get("label_id"),
        "probability": result.get("probability"),
        "threshold_used": result.get("threshold_used"),
        "threshold_profile": result.get("threshold_profile"),
        "mode": result.get("mode"),
        "semantic_score": result.get("semantic_score"),
        "frequency_score": result.get("frequency_score"),
        "raw_logit": result.get("raw_logit"),
    }


def infer_folder(detector: ForensicDetector, folder: Path, threshold: float | None, threshold_profile: str) -> List[Dict[str, object]]:
    image_paths: List[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        image_paths.extend(folder.rglob(pattern))
    rows = []
    for image_path in sorted(image_paths):
        rows.append(predict_single(detector, image_path=image_path, threshold=threshold, threshold_profile=threshold_profile))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference entrypoint for the final V10 detector.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--threshold-profile",
        type=str,
        default="balanced",
        choices=["recall-first", "balanced", "precision-first", "recall", "precision", "f1"],
    )
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image is None and args.folder is None:
        raise ValueError("Provide either --image or --folder")

    detector = ForensicDetector(
        model_path=args.checkpoint,
        device=args.device,
        config_name="default",
    )

    if args.image is not None:
        row = predict_single(
            detector,
            image_path=Path(args.image),
            threshold=args.threshold,
            threshold_profile=args.threshold_profile,
        )
        print(json.dumps(row, ensure_ascii=False, indent=2))

    if args.folder is not None:
        rows = infer_folder(
            detector,
            folder=Path(args.folder),
            threshold=args.threshold,
            threshold_profile=args.threshold_profile,
        )
        out_csv = Path(args.out_csv) if args.out_csv else Path(args.folder) / "inference_results.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved inference CSV: {out_csv}")
        print(f"Processed images: {len(rows)}")


if __name__ == "__main__":
    main()
