from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class ImageFolderWithMeta(Dataset):
    """
    简单版 ImageFolder，约定目录结构为：

    root/
      real/
      fake/

    其中 real 映射为标签 0，fake 映射为标签 1。

    额外返回 meta 字典，包含：
    - path: 图像绝对路径
    - dataset: 数据集名称（如 genimage / aigibench / chameleon）
    - split: 数据划分（train / val / test）
    - label_str: 'real' or 'fake'
    """

    _CLASS_TO_LABEL = {"real": 0, "fake": 1}

    def __init__(
        self,
        root: str,
        transform=None,
        dataset_name: str = "generic",
        split: str = "train",
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.dataset_name = dataset_name
        self.split = split

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.samples = self._gather_samples()

    def _gather_samples(self) -> Tuple[Tuple[Path, int, str], ...]:
        items = []
        # 同时检查小写和大写目录
        for class_name, label in self._CLASS_TO_LABEL.items():
            class_dir = self.root / class_name
            # 如果小写目录不存在，尝试大写
            if not class_dir.exists():
                class_dir = self.root / class_name.upper()
            
            if not class_dir.exists():
                continue

            # 收集图片，匹配大小写后缀
            extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP")
            for ext in extensions:
                for img_path in class_dir.rglob(ext):
                    items.append((img_path, label, class_name))

        if not items:
            raise RuntimeError(f"No images found under {self.root}. \n"
                             f"Please ensure class directories are named 'real'/'fake' or 'REAL'/'FAKE'.")

        return tuple(items)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, int, Dict[str, Any]]:  # type: ignore[override]
        img_path, label, label_str = self.samples[index]
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            if isinstance(self.transform, dict):
                # 根据 label 选择 transform
                # 0: real, 1: fake
                tf = self.transform.get(label, self.transform.get("default"))
                if tf:
                    img = tf(img)
            else:
                img = self.transform(img)

        meta: Dict[str, Any] = {
            "path": str(img_path),
            "dataset": self.dataset_name,
            "split": self.split,
            "label_str": label_str,
            "index": index,
        }
        return img, label, meta


def _build_root(root: Optional[str], default_root: str) -> str:
    if root is not None:
        return root
    return default_root

