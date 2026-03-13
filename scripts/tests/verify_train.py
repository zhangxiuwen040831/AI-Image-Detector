
import sys
import torch
import shutil
from pathlib import Path
import logging

# Add root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model import AIGCImageDetector
from src.training.trainer import Trainer
from src.data.dataset import FusionDataset
from torch.utils.data import DataLoader

def setup_logger():
    logger = logging.getLogger("verify_train")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def verify_train():
    print("="*40)
    print("Step 4, 5 & 6: Training Verification")
    print("="*40)

    # 1. Config
    print("[1] Setting up Configuration...")
    device = torch.device("cpu")
    dataset_root = ROOT / "data" / "fusion_test"
    
    if not dataset_root.exists():
        print(f"ERROR: Dataset root {dataset_root} does not exist!")
        return

    cfg = {
        "train": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "epochs": 1,
            "label_smoothing": 0.1,
            "grad_clip": 1.0,
            "save_dir": str(ROOT / "checkpoints" / "verify"),
            "device": "cpu"
        },
        "model": {
            "backbone": "resnet18",
            "rgb_pretrained": False,
            "noise_pretrained": False,
            "freq_pretrained": False,
            "fused_dim": 512,
            "classifier_hidden_dim": 256,
            "dropout": 0.3
        },
        "data": {
            "runtime_dataset_root": str(dataset_root),
            "image_size": 224
        }
    }
    
    logger = setup_logger()
    
    # 2. Data Loading
    print("[2] verifying Data Loading...")
    try:
        train_dataset = FusionDataset(root=str(dataset_root), split="train", image_size=224)
        val_dataset = FusionDataset(root=str(dataset_root), split="test", image_size=224) # Use test as val
        
        print(f"    Train samples: {len(train_dataset)}")
        print(f"    Val samples: {len(val_dataset)}")
        
        if len(train_dataset) == 0:
            print("    ERROR: No training samples found!")
            return

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        # Test loading one batch
        img, label = next(iter(train_loader))
        print(f"    Batch shape: {img.shape}, Label shape: {label.shape}")
        
    except Exception as e:
        print(f"    ERROR: Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Model & Trainer
    print("[3] Initializing Trainer...")
    try:
        model = AIGCImageDetector(cfg["model"])
        trainer = Trainer(model=model, config=cfg, device=device, logger=logger)
        print("    Trainer initialized.")
    except Exception as e:
        print(f"    ERROR: Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Training Loop
    print("[4] Running Minimal Training...")
    try:
        trainer.fit(train_loader, val_loader)
        print("    Training loop completed successfully.")
    except Exception as e:
        print(f"    ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
        
    # Cleanup
    if Path(cfg["train"]["save_dir"]).exists():
        shutil.rmtree(cfg["train"]["save_dir"])
        print("    Cleaned up checkpoints.")

    print("\nTraining Verification Complete!")

if __name__ == "__main__":
    verify_train()
