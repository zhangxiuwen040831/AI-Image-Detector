import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import sys
import platform

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import FusionDataset
from src.data.sampler import SourceBalancedSampler

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def patch_runtime_dataset_root(cfg: dict) -> dict:
    data_cfg = cfg["data"]
    if "runtime_dataset_root" in data_cfg and data_cfg["runtime_dataset_root"]:
        return cfg
    if platform.system().lower().startswith("win"):
        data_cfg["runtime_dataset_root"] = data_cfg["local_dataset_root"]
    else:
        data_cfg["runtime_dataset_root"] = data_cfg["server_dataset_root"]
    return cfg

def build_train_loader(config_path, batch_size=None):
    cfg = load_config(config_path)
    cfg = patch_runtime_dataset_root(cfg)
    data_cfg = cfg['data']
    loader_cfg = cfg['loader']
    
    print(f"Using dataset root: {data_cfg['runtime_dataset_root']}")
    
    if batch_size is None:
        batch_size = loader_cfg['batch_size']
    
    # 当num_workers为0时，prefetch_factor必须为None
    prefetch_factor = loader_cfg['prefetch_factor'] if loader_cfg['num_workers'] > 0 else None
    persistent_workers = loader_cfg['persistent_workers'] if loader_cfg['num_workers'] > 0 else False
    
    dataset = FusionDataset(
        split='train',
        image_size=data_cfg['image_size'],
        root=data_cfg['runtime_dataset_root']
    )
    sampler = None
    if hasattr(dataset, "records") or hasattr(dataset, "source_to_indices"):
        sampler = SourceBalancedSampler(
            dataset=dataset,
            artifact_cifake_ratio=tuple(loader_cfg.get("artifact_cifake_ratio", [3, 1])),
            real_fake_ratio=tuple(loader_cfg.get("real_fake_ratio", [1, 1])),
            num_samples=len(dataset),
            seed=int(cfg.get("train", {}).get("seed", 42)),
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=loader_cfg['num_workers'],
        pin_memory=loader_cfg['pin_memory'],
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

def build_val_loader(config_path, batch_size=None):
    cfg = load_config(config_path)
    cfg = patch_runtime_dataset_root(cfg)
    data_cfg = cfg['data']
    loader_cfg = cfg['loader']
    
    print(f"Using dataset root: {data_cfg['runtime_dataset_root']}")
    
    if batch_size is None:
        batch_size = loader_cfg['batch_size']
    
    # 当num_workers为0时，prefetch_factor必须为None
    prefetch_factor = loader_cfg['prefetch_factor'] if loader_cfg['num_workers'] > 0 else None
    persistent_workers = loader_cfg['persistent_workers'] if loader_cfg['num_workers'] > 0 else False
    
    dataset = FusionDataset(
        split='val',
        image_size=data_cfg['image_size'],
        root=data_cfg['runtime_dataset_root']
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_cfg['num_workers'],
        pin_memory=loader_cfg['pin_memory'],
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
