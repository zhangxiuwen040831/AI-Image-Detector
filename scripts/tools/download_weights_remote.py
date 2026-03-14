import os
import timm
import torch

# 配置环境变量以使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用 SSL 验证以避免证书错误
os.environ["CURL_CA_BUNDLE"] = ""

print("Downloading ResNet18 weights from hf-mirror.com...")
try:
    # 这会自动下载权重并缓存到 ~/.cache/huggingface/hub
    model = timm.create_model('resnet18', pretrained=True)
    print("Successfully downloaded ResNet18 weights!")
except Exception as e:
    print(f"Failed to download ResNet18: {e}")

print("\nDownloading CLIP (ViT-L-14) weights...")
try:
    # 如果代码中用到了 CLIP，这里也预先下载
    import open_clip
    model = open_clip.create_model('ViT-L-14', pretrained='openai')
    print("Successfully downloaded CLIP weights!")
except ImportError:
    print("open_clip not installed, skipping.")
except Exception as e:
    print(f"Failed to download CLIP: {e}")
