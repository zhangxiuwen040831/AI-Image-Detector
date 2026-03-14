from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from src.data.transforms import get_transforms


class FusionDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int = 224) -> None:
        self.root_dir = Path(root)
        self.split = split
        self.image_size = image_size
        
        # Map 'val' to 'test' if 'val' does not exist but 'test' does
        if split == 'val' and not (self.root_dir / 'val').exists() and (self.root_dir / 'test').exists():
            self.dataset_split = 'test'
        else:
            self.dataset_split = split
            
        self.transform = get_transforms(image_size)
            
        self.samples: List[Tuple[Path, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        # 记录已扫描的文件，避免重复
        seen_files = set()
        
        # 1. 扫描 CIFAKE (标准结构: train/real, train/fake)
        # 假设 cifake 目录下有 train/test，或者直接是类别
        cifake_root = self.root_dir / "cifake"
        if cifake_root.exists():
            print(f"Scanning CIFAKE at {cifake_root}")
            # 优先找 split
            split_path = cifake_root / self.dataset_split
            if split_path.exists():
                self._scan_dir(split_path, seen_files, source="cifake")
            elif self.dataset_split == "train":
                self._scan_dir(cifake_root, seen_files, source="cifake")

        # 2. 扫描 Artifact Dataset (复杂结构)
        artifact_root = self.root_dir / "artifact-dataset"
        if artifact_root.exists():
            print(f"Scanning Artifact Dataset at {artifact_root}")
            self._scan_artifact_dir(artifact_root, seen_files)

        print(f"Dataset({self.dataset_split}): Found {len(self.samples)} samples.")

    def _scan_dir(self, directory: Path, seen_files: set, source: str = "unknown") -> None:
        if not directory.exists():
            return
        
        # 使用确定性排序确保 hash 稳定
        for img_path in sorted(directory.rglob("*")):
            if not img_path.is_file(): continue
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}: continue
            if str(img_path) in seen_files: continue

            # 动态划分验证集 (5%)
            # 使用文件路径的 hash 来决定该样本属于 train 还是 val
            # 这种方式无需物理移动文件，且跨进程/跨机器一致
            import hashlib
            # 使用相对路径进行 hash，避免绝对路径差异影响
            rel_path = str(img_path.relative_to(self.root_dir)).replace("\\", "/")
            path_hash = int(hashlib.md5(rel_path.encode('utf-8')).hexdigest(), 16)
            is_val_sample = (path_hash % 100) < 5  # 5% 概率
            
            if self.dataset_split == "train":
                if is_val_sample: continue # 如果是训练集模式，跳过验证样本
            elif self.dataset_split == "val":
                if not is_val_sample: continue # 如果是验证集模式，跳过训练样本
            
            # 之前的 split 目录过滤逻辑 (如果目录本身叫 val/test，优先尊重目录名)
            # 但既然现在要强制从全量数据里抽 5%，我们可以覆盖这个逻辑，或者仅对未分 split 的数据应用动态划分
            # 为了简单且符合用户需求，我们对所有数据应用动态划分，除非它明确位于 test 目录（通常 test 不参与 train/val 划分）
            if "test" in str(img_path).lower():
                continue # 忽略测试集，只在 train/val 间划分

            # 通用规则：根据路径关键字判断
            path_str = str(img_path).lower().replace("\\", "/")
            label = -1
            
            parts = path_str.split("/")
            if "real" in parts or "0_real" in parts:
                label = 0
            elif "fake" in parts or "1_fake" in parts:
                label = 1
            
            if label != -1:
                self.samples.append((img_path, label))
                seen_files.add(str(img_path))

    def _scan_artifact_dir(self, root: Path, seen_files: set) -> None:
        # Artifact Dataset 规则：
        # Real Categories:
        real_categories = {"afhq", "celebahq", "coco", "ffhq", "imagenet", "landscape", "metfaces"}
        # 其他都是 Fake
        
        # 遍历 root 下的一级子目录
        for category_dir in root.iterdir():
            if not category_dir.is_dir(): continue
            
            category_name = category_dir.name.lower()
            is_real = category_name in real_categories
            target_label = 0 if is_real else 1
            
            # print(f"  - Scanning category: {category_name} (Label: {target_label})")
            
            # 递归扫描该类别下的所有图片
            # 使用 rglob("*") 遍历所有文件
            for img_path in sorted(category_dir.rglob("*")):
                # 基本过滤
                if not img_path.is_file(): continue
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}: continue
                if str(img_path) in seen_files: continue
                
                # 动态划分验证集 (5%)
                import hashlib
                rel_path = str(img_path.relative_to(self.root_dir)).replace("\\", "/")
                path_hash = int(hashlib.md5(rel_path.encode('utf-8')).hexdigest(), 16)
                is_val_sample = (path_hash % 100) < 5  # 5%
                
                if self.dataset_split == "train":
                    if is_val_sample: continue
                elif self.dataset_split == "val":
                    if not is_val_sample: continue
                
                self.samples.append((img_path, target_label))
                seen_files.add(str(img_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image or handle error? 
            # For simplicity, let's return a zero tensor and label -1 (or handle in collate)
            # But standard is to fail or skip. 
            # Let's try to reload a random sample?
            # For this audit, let's just fail loudly or return zeros.
            # Returning zeros might break batch processing if shape is wrong.
            # Let's re-raise.
            raise e
