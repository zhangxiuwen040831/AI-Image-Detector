import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.detector_model import build_detector_from_config
from src.models.hybrid_detector import build_hybrid_detector_from_config

# 尝试使用fvcore.nn计算FLOPs
try:
    from fvcore.nn import FlopCountAnalysis
    has_fvcore = True
except ImportError:
    has_fvcore = False

def calculate_flops(model, input_size=(3, 224, 224)):
    """计算模型的FLOPs"""
    if not has_fvcore:
        return "fvcore not installed, cannot calculate FLOPs"
    
    # 创建一个输入张量
    input_tensor = torch.randn(1, *input_size)
    
    # 计算FLOPs
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()
    
    # 格式化输出
    if total_flops >= 1e12:
        return f"{total_flops / 1e12:.2f} TFLOPs"
    elif total_flops >= 1e9:
        return f"{total_flops / 1e9:.2f} GFLOPs"
    elif total_flops >= 1e6:
        return f"{total_flops / 1e6:.2f} MFLOPs"
    else:
        return f"{total_flops:.2f} FLOPs"

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

# 计算FLOPs
detector_flops = calculate_flops(detector)
hybrid_flops = calculate_flops(hybrid_detector)

print(f"Detector model FLOPs: {detector_flops}")
print(f"Hybrid detector FLOPs: {hybrid_flops}")
