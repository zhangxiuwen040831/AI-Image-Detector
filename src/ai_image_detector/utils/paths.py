from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 配置文件目录
CONFIG_DIR = PROJECT_ROOT / "configs"

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"

# 检查点目录
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 源码目录
SRC_DIR = PROJECT_ROOT / "src"

# 服务目录
SERVICES_DIR = PROJECT_ROOT / "services"

# 脚本目录
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# 测试目录
TESTS_DIR = PROJECT_ROOT / "tests"

# 预训练权重路径
PRETRAINED_DIR = CHECKPOINT_DIR / "pretrained"

# 最佳模型路径 - 唯一模型入口
BEST_MODEL_PATH = CHECKPOINT_DIR / "best.pth"

# ResNet18预训练权重路径
RESNET18_WEIGHTS_PATH = PRETRAINED_DIR / "resnet18.safetensors"


def get_model_path() -> str:
    """
    获取模型文件路径
    单模型文件模式：只返回 checkpoints/best.pth
    """
    return str(BEST_MODEL_PATH)
