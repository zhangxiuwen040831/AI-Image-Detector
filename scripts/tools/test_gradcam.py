import argparse
import base64
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from ai_image_detector.inference.detector import ForensicDetector


def _save_base64_png(base64_str: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode(base64_str)
    output_path.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "checkpoints" / "best.pth"), help="模型路径")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "gradcam_outputs"), help="输出目录")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"[1/5] 模型路径: {model_path}")
    if not model_path.exists():
        print("[FAIL] 模型文件不存在")
        return 1

    print(f"[2/5] 测试图片: {image_path}")
    if not image_path.exists():
        print("[FAIL] 测试图片不存在")
        return 1

    try:
        detector = ForensicDetector(str(model_path), enable_debug=True)
        print(f"[3/5] 模型加载成功, device={detector.device}")
    except Exception as e:
        print(f"[FAIL] 模型加载失败: {e}")
        return 1

    try:
        result = detector.predict(str(image_path), debug=True)
        print("[4/5] 推理成功")
    except Exception as e:
        print(f"[FAIL] 推理失败: {e}")
        return 1

    artifacts = result.get("artifacts") or {}
    debug = result.get("debug") or {}
    grad_cam = artifacts.get("grad_cam")
    grad_cam_overlay = artifacts.get("grad_cam_overlay")

    print(f"    - 预测结果: {result.get('prediction')} (confidence={result.get('confidence')})")
    print(f"    - grad_cam 状态: {'OK' if grad_cam else 'FAIL'}")
    print(f"    - overlay 状态: {'OK' if grad_cam_overlay else 'FAIL'}")
    print(f"    - debug.grad_cam_status: {debug.get('grad_cam_status')}")
    print(f"    - debug.error: {debug.get('error')}")

    if grad_cam:
        grad_cam_path = output_dir / "grad_cam.png"
        _save_base64_png(grad_cam, grad_cam_path)
        print(f"[5/5] grad_cam 已保存: {grad_cam_path}")
    else:
        print("[5/5] grad_cam 未生成")

    if grad_cam_overlay:
        overlay_path = output_dir / "grad_cam_overlay.png"
        _save_base64_png(grad_cam_overlay, overlay_path)
        print(f"      overlay 已保存: {overlay_path}")
    else:
        print("      overlay 未生成")

    if not grad_cam or not grad_cam_overlay:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
