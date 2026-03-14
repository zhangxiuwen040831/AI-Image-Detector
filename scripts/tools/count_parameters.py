import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.detector_model import build_detector_from_config
from src.models.hybrid_detector import build_hybrid_detector_from_config

# 计算模型参数量的函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 构建detector_model
detector = build_detector_from_config(
    device="cpu",
    train_backbone=False,
    use_lora=False,
    use_osd=False
)

# 构建hybrid_detector
hybrid_config = {
    "backbone_name": "ViT-L-14",
    "backbone_pretrained": "openai",
    "device": "cpu",
    "train_backbone": False,
    "use_lora": False,
    "use_osd": False
}
hybrid_detector = build_hybrid_detector_from_config(hybrid_config)

# 计算参数量
detector_params = count_parameters(detector)
hybrid_params = count_parameters(hybrid_detector)

print(f"Detector model parameters: {detector_params:,}")
print(f"Hybrid detector parameters: {hybrid_params:,}")
