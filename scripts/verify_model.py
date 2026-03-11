
import sys
import torch
from pathlib import Path

# Add root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model import AIGCImageDetector
from src.models.detector import MultiBranchDetector
from src.models.rgb_branch import RGBSpatialBranch
from src.models.noise_branch import NoiseResidualBranch
from src.models.freq_branch import FrequencyBranch
from src.models.fusion import ConcatFusionMLP

def verify_model():
    print("="*40)
    print("Step 2 & 3: Model Verification")
    print("="*40)

    # Config
    model_cfg = {
        "backbone": "resnet18",
        "rgb_pretrained": False, # Save time/bandwidth
        "noise_pretrained": False,
        "freq_pretrained": False,
        "fused_dim": 512,
        "classifier_hidden_dim": 256,
        "dropout": 0.3
    }

    # 1. Initialize Model
    print("[1] Initializing Model...")
    try:
        model = AIGCImageDetector(model_cfg)
        print("    Model initialized successfully.")
    except Exception as e:
        print(f"    ERROR: Model initialization failed: {e}")
        return

    # 2. Dummy Input
    batch_size = 2
    image_size = 224
    x = torch.randn(batch_size, 3, image_size, image_size)
    print(f"[2] Dummy Input: {x.shape}")

    # 3. Forward Pass
    print("[3] Running Forward Pass...")
    try:
        out = model(x)
        print("    Forward pass successful.")
        for k, v in out.items():
            print(f"    Output '{k}': {v.shape}")
            
        # Check shapes
        assert out["logit"].shape == (batch_size, 1), f"Logit shape mismatch: {out['logit'].shape}"
        assert out["probability"].shape == (batch_size, 1), f"Prob shape mismatch: {out['probability'].shape}"
        assert out["fused_feat"].shape == (batch_size, 512), f"Fused feat shape mismatch: {out['fused_feat'].shape}"
        
    except Exception as e:
        print(f"    ERROR: Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Backward Pass
    print("[4] Running Backward Pass...")
    try:
        labels = torch.randint(0, 2, (batch_size, 1)).float()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(out["logit"], labels)
        print(f"    Loss: {loss.item()}")
        
        loss.backward()
        print("    Backward pass successful.")
        
        # Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if has_grad:
            print("    Gradients computed.")
        else:
            print("    WARNING: No gradients found!")
            
    except Exception as e:
        print(f"    ERROR: Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nModel Verification Complete!")

if __name__ == "__main__":
    verify_model()
