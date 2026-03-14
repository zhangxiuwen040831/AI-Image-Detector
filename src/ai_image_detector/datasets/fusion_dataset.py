import os
import shutil
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from .transforms import get_transforms


class FusionDataset(Dataset):
    """
    CIFAKE 与 Artifact-dataset 融合数据集
    """

    def __init__(self, split: str, image_size: int = 224, root: str = "data/fusion"):
        """
        Args:
            split: "train", "val", or "test"
            image_size: 图像大小
            root: 融合数据集根目录
        """
        self.split = split
        self.image_size = image_size
        self.root = root
        self.transform = get_transforms(split=split, image_size=image_size)
        
        # 加载图像路径和标签
        self.image_paths = []
        self.labels = []
        self._load_data()
        print(f"Loaded {len(self.image_paths)} images for split {split}")

    def _load_data(self):
        """
        加载融合数据集的图像路径和标签
        """
        print(f"Loading data from {self.root} for split {self.split}")
        
        # 加载CIFAKE数据集
        cifake_dir = os.path.join(self.root, "cifake", self.split)
        print(f"Checking CIFAKE directory: {cifake_dir}")
        
        # 如果是val目录且不存在，使用test目录作为验证集
        if self.split == "val" and not os.path.exists(cifake_dir):
            print(f"Val directory not found, using test directory instead")
            cifake_dir = os.path.join(self.root, "cifake", "test")
            print(f"Checking CIFAKE test directory: {cifake_dir}")
        
        if os.path.exists(cifake_dir):
            # CIFAKE的目录结构是 train/REAL, train/FAKE
            real_dir = os.path.join(cifake_dir, "REAL")
            fake_dir = os.path.join(cifake_dir, "FAKE")
            
            # 加载真实图像
            if os.path.exists(real_dir):
                print(f"Loading real images from {real_dir}")
                # 加载所有真实图像
                count = 0
                for img_name in os.listdir(real_dir):
                    img_path = os.path.join(real_dir, img_name)
                    if self._is_valid_image(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(0)  # 真实图像标签为 0
                        count += 1
                print(f"Loaded {count} real images from CIFAKE")
            
            # 加载生成图像
            if os.path.exists(fake_dir):
                print(f"Loading fake images from {fake_dir}")
                # 加载所有生成图像
                count = 0
                for img_name in os.listdir(fake_dir):
                    img_path = os.path.join(fake_dir, img_name)
                    if self._is_valid_image(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(1)  # 生成图像标签为 1
                        count += 1
                print(f"Loaded {count} fake images from CIFAKE")
        else:
            print(f"CIFAKE directory {cifake_dir} does not exist")
        
        # 加载Artifact-dataset数据集
        artifact_dir = os.path.join(self.root, "artifact-dataset")
        print(f"Checking Artifact-dataset directory: {artifact_dir}")
        if os.path.exists(artifact_dir):
            # Artifact-dataset的目录结构比较复杂，需要递归搜索
            print(f"Searching for {self.split} directory in Artifact-dataset")
            for root, dirs, files in os.walk(artifact_dir):
                # 检查是否是train或val目录
                if os.path.basename(root) == self.split:
                    print(f"Found {self.split} directory: {root}")
                    # 查找real和fake子目录
                    for subdir in dirs:
                        if subdir.lower() == "real":
                            real_dir = os.path.join(root, subdir)
                            print(f"Loading real images from {real_dir}")
                            # 加载所有真实图像
                            count = 0
                            for img_name in os.listdir(real_dir):
                                img_path = os.path.join(real_dir, img_name)
                                if self._is_valid_image(img_path):
                                    self.image_paths.append(img_path)
                                    self.labels.append(0)  # 真实图像标签为 0
                                    count += 1
                            print(f"Loaded {count} real images from Artifact-dataset")
                        elif subdir.lower() == "fake":
                            fake_dir = os.path.join(root, subdir)
                            print(f"Loading fake images from {fake_dir}")
                            # 加载所有生成图像
                            count = 0
                            for img_name in os.listdir(fake_dir):
                                img_path = os.path.join(fake_dir, img_name)
                                if self._is_valid_image(img_path):
                                    self.image_paths.append(img_path)
                                    self.labels.append(1)  # 生成图像标签为 1
                                    count += 1
                            print(f"Loaded {count} fake images from Artifact-dataset")
        else:
            print(f"Artifact-dataset directory {artifact_dir} does not exist")

    def _is_valid_image(self, path: str) -> bool:
        """
        检查文件是否为有效的图像文件
        """
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception as e:
            print(f"Invalid image: {path}, error: {e}")
            return False

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载图像
        with Image.open(img_path).convert("RGB") as img:
            img = self.transform(img)

        return img, label, img_path


def fuse_datasets(
    cifake_root: str, 
    artifact_root: str, 
    output_root: str, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1,
    image_size: int = 224
):
    """
    融合 CIFAKE 和 Artifact-dataset 数据集
    
    Args:
        cifake_root: CIFAKE 数据集根目录
        artifact_root: Artifact-dataset 数据集根目录
        output_root: 融合后数据集输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        image_size: 图像大小
    """
    # 创建输出目录结构
    output_dirs = {
        "train": {
            "real": os.path.join(output_root, "train", "real"),
            "fake": os.path.join(output_root, "train", "fake")
        },
        "val": {
            "real": os.path.join(output_root, "val", "real"),
            "fake": os.path.join(output_root, "val", "fake")
        },
        "test": {
            "real": os.path.join(output_root, "test", "real"),
            "fake": os.path.join(output_root, "test", "fake")
        }
    }

    for split in output_dirs:
        for label in output_dirs[split]:
            os.makedirs(output_dirs[split][label], exist_ok=True)

    # 收集所有图像路径
    all_real_images = []
    all_fake_images = []

    # 处理 CIFAKE 数据集
    if os.path.exists(cifake_root):
        # CIFAKE 的目录结构是 train/REAL, train/FAKE, test/REAL, test/FAKE
        for split in ["train", "test"]:
            real_dir = os.path.join(cifake_root, split, "REAL")
            fake_dir = os.path.join(cifake_root, split, "FAKE")

            if os.path.exists(real_dir):
                for img_name in os.listdir(real_dir):
                    img_path = os.path.join(real_dir, img_name)
                    if _is_valid_image(img_path):
                        all_real_images.append(img_path)

            if os.path.exists(fake_dir):
                for img_name in os.listdir(fake_dir):
                    img_path = os.path.join(fake_dir, img_name)
                    if _is_valid_image(img_path):
                        all_fake_images.append(img_path)

    # 处理 Artifact-dataset 数据集
    if os.path.exists(artifact_root):
        # Artifact-dataset 的目录结构是 train/real, train/fake, val/real, val/fake
        for split in ["train", "val"]:
            real_dir = os.path.join(artifact_root, split, "real")
            fake_dir = os.path.join(artifact_root, split, "fake")

            if os.path.exists(real_dir):
                for img_name in os.listdir(real_dir):
                    img_path = os.path.join(real_dir, img_name)
                    if _is_valid_image(img_path):
                        all_real_images.append(img_path)

            if os.path.exists(fake_dir):
                for img_name in os.listdir(fake_dir):
                    img_path = os.path.join(fake_dir, img_name)
                    if _is_valid_image(img_path):
                        all_fake_images.append(img_path)

    # 数据平衡
    min_count = min(len(all_real_images), len(all_fake_images))
    all_real_images = all_real_images[:min_count]
    all_fake_images = all_fake_images[:min_count]

    # 随机打乱
    random.shuffle(all_real_images)
    random.shuffle(all_fake_images)

    # 计算划分数量
    total_count = len(all_real_images)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count

    # 划分数据集
    splits = {
        "train": slice(0, train_count),
        "val": slice(train_count, train_count + val_count),
        "test": slice(train_count + val_count, total_count)
    }

    # 复制图像到输出目录
    for split in splits:
        # 复制真实图像
        real_images = all_real_images[splits[split]]
        for i, img_path in enumerate(real_images):
            img_name = f"real_{i}.jpg"
            output_path = os.path.join(output_dirs[split]["real"], img_name)
            _resize_and_save(img_path, output_path, image_size)

        # 复制生成图像
        fake_images = all_fake_images[splits[split]]
        for i, img_path in enumerate(fake_images):
            img_name = f"fake_{i}.jpg"
            output_path = os.path.join(output_dirs[split]["fake"], img_name)
            _resize_and_save(img_path, output_path, image_size)

    print(f"融合完成！")
    print(f"训练集：{train_count} 对图像")
    print(f"验证集：{val_count} 对图像")
    print(f"测试集：{test_count} 对图像")


def _is_valid_image(path: str) -> bool:
    """
    检查文件是否为有效的图像文件
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False


def _resize_and_save(input_path: str, output_path: str, size: int):
    """
    调整图像大小并保存
    """
    try:
        with Image.open(input_path).convert("RGB") as img:
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            img.save(output_path, "JPEG", quality=95)
    except Exception as e:
        print(f"处理图像 {input_path} 时出错: {e}")
