"""
few-shot 微调模板（不执行）
用于加载模型并准备 few-shot 微调环境
"""
import torch
from pathlib import Path

# 添加 src 到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_image_detector.ntire.model import HybridAIGCDetector

CKPT_PATH = Path("checkpoints/best.pth")
OUTPUT_DIR = Path("photos_test_outputs/fewshot_finetune")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    model = HybridAIGCDetector(backbone_name="vit_base_patch16_clip_224.openai", pretrained_backbone=False)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state_dict = ckpt["model"]
    
    # 处理 DataParallel 前缀
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    # 保存模型信息
    readme_content = """# Few-shot Fine-tune Template

## 说明
此模板用于 few-shot 微调，尚未执行。

## 使用前请检查
1. 确认训练数据加载器 (dataloader) 配置正确
2. 确认训练器 (trainer) 配置正确
3. 确认优化器和学习率调度器设置

## 模型信息
- Checkpoint: checkpoints/best.pth
- Backbone: vit_base_patch16_clip_224.openai
- Device: {device}

## 执行步骤
1. 准备 few-shot 数据集
2. 配置训练参数
3. 运行微调脚本
""".format(device=device)

    (OUTPUT_DIR / "README.txt").write_text(readme_content, encoding="utf-8")
    print(f"Few-shot fine-tune template created at: {OUTPUT_DIR}")
    print("Note: This is a template only. Check trainer/dataloader before running actual fine-tuning.")

if __name__ == "__main__":
    main()
