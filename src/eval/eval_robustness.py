import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import DataLoader

from src.datasets.genimage import GenImageDataset
from src.eval.metrics import compute_binary_metrics
from src.models.detector_model import DetectorModel, build_detector_from_config


def load_model(ckpt_path: str, device: torch.device) -> (DetectorModel, Dict[str, Any]):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg: Dict[str, Any] = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {})

    model = build_detector_from_config(
        device=str(device),
        train_backbone=False,
        head_hidden_dim=model_cfg.get("head_hidden_dim"),
        head_dropout=model_cfg.get("head_dropout", 0.0),
        backbone_name="ViT-L-14",
        backbone_pretrained="openai",
        use_lora=model_cfg.get("use_lora", False),
        lora_rank=model_cfg.get("lora_rank", 8),
        lora_alpha=model_cfg.get("lora_alpha", 16.0),
        lora_dropout=model_cfg.get("lora_dropout", 0.0),
        use_osd=model_cfg.get("use_osd", False),
        osd_proj_dim=model_cfg.get("osd_proj_dim", 128),
        osd_lambda_orth=model_cfg.get("osd_lambda_orth", 1e-3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, cfg


def jpeg_compress(pil_img: Image.Image, quality: int = 60) -> Image.Image:
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def center_crop_fraction(pil_img: Image.Image, fraction: float = 0.5) -> Image.Image:
    w, h = pil_img.size
    new_w, new_h = int(w * fraction), int(h * fraction)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    return pil_img.crop((left, top, right, bottom)).resize((w, h), Image.BICUBIC)


def add_gaussian_noise(pil_img: Image.Image, sigma: float = 10.0) -> Image.Image:
    arr = np.array(pil_img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_perturbation(pil_img: Image.Image, kind: str) -> Image.Image:
    if kind == "jpeg60":
        return jpeg_compress(pil_img, quality=60)
    if kind == "center_crop_50":
        return center_crop_fraction(pil_img, fraction=0.5)
    if kind == "gaussian_10":
        return add_gaussian_noise(pil_img, sigma=10.0)
    if kind == "gaussian_25":
        return add_gaussian_noise(pil_img, sigma=25.0)
    return pil_img


def evaluate_under_perturbations(
    model: DetectorModel,
    dataset: GenImageDataset,
    device: torch.device,
    kinds: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    在原始与不同扰动下分别评估模型。
    注意：这里为了简单起见，重新构造每个样本的扰动图像并使用 dataset 的 transform。
    """
    results: Dict[str, Dict[str, float]] = {}

    # 先评估原始图像
    from src.eval.metrics import compute_binary_metrics

    base_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    y_true: List[int] = []
    y_prob: List[float] = []
    with torch.no_grad():
        for images, labels, _ in base_loader:
            images = images.to(device)
            outputs = model(images)
            probs = outputs["probs"]
            y_true.extend(labels.tolist())
            y_prob.extend(probs.cpu().tolist())
    results["clean"] = compute_binary_metrics(y_true, y_prob)

    # 对每种扰动单独评估
    for kind in kinds:
        print(f"[INFO] Evaluating perturbation: {kind}")
        y_true = []
        y_prob = []
        with torch.no_grad():
            for idx in range(len(dataset)):
                img, label, _ = dataset[idx]
                # 反向到 PIL，再施加扰动，再应用相同 transform
                # 注意：dataset.transform 可能为空，实际项目中可定制更优实现
                pil_img = dataset.samples[idx][0].open().convert("RGB")  # type: ignore[attr-defined]
                perturbed = apply_perturbation(pil_img, kind=kind)
                if dataset.transform is not None:
                    import torchvision.transforms as T

                    tensor_img = dataset.transform(perturbed)
                else:
                    tensor_img = T.ToTensor()(perturbed)

                tensor_img = tensor_img.unsqueeze(0).to(device)
                outputs = model(tensor_img)
                probs = outputs["probs"]
                y_true.append(int(label))
                y_prob.extend(probs.cpu().tolist())

        results[kind] = compute_binary_metrics(y_true, y_prob)

    return results


def main(ckpt_path: str, output_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(ckpt_path, device)

    image_size = cfg.get("data", {}).get("image_size", 224)
    dataset = GenImageDataset(split="val", image_size=image_size)

    kinds = ["jpeg60", "center_crop_50", "gaussian_10", "gaussian_25"]
    results = evaluate_under_perturbations(model, dataset, device, kinds=kinds)

    print("[INFO] Robustness results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    out = {
        "checkpoint": ckpt_path,
        "robustness": results,
    }
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved robustness report to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness evaluation")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/robustness_eval.json",
        help="Path to output JSON report",
    )
    args = parser.parse_args()
    main(args.ckpt, args.output)

