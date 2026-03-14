import os
import tarfile
import paramiko
import sys
import time
from pathlib import Path

# Configuration
SERVER_HOST = "connect.westd.seetacloud.com"
SERVER_PORT = 41383
SERVER_USER = "root"
SERVER_PASS = "PLHtzLMLcxei"
REMOTE_BASE_DIR = "/root/autodl-tmp"
REMOTE_PROJECT_DIR = f"{REMOTE_BASE_DIR}/ai-image-detector"
LOCAL_ARCHIVE_NAME = "ai-image-detector.tar.gz"
REMOTE_ARCHIVE_PATH = f"{REMOTE_BASE_DIR}/{LOCAL_ARCHIVE_NAME}"

# Weights to upload (local path to remote path)
LOCAL_WEIGHTS = {
    "resnet18.safetensors": f"{REMOTE_PROJECT_DIR}/resnet18.safetensors"
}

# Directories/Files to Include
INCLUDE_PATHS = ["src", "configs", "backend", "scripts", "requirements.txt", "resnet18.safetensors"]
# Directories to Exclude (patterns)
# CRITICAL: Do NOT exclude 'src/data' just because it contains 'data'.
# We must allow 'src/data' but exclude top-level 'data/' folder.
EXCLUDE_PATTERNS = ["datasets", "logs", "checkpoints", "__pycache__", ".git", "build", "wandb"]

def create_default_config():
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "dataset_config.yaml"
    
    if not config_path.exists():
        print("Creating default configuration: configs/dataset_config.yaml")
        config_content = """
system:
  seed: 42
  device: cuda
  num_workers: 4
  pin_memory: true

data:
  image_size: 224
  # These paths are relative to the project root on the server
  # But since we use absolute paths or relative to working dir, we can adjust.
  # The user says data is in /root/autodl-tmp/ai-image-detector/data
  # And we run from /root/autodl-tmp/ai-image-detector
  # So "data" relative path is correct if we run from project root.
  # BUT to be safe and robust, let's use the explicit structure we check.
  local_dataset_root: "data"
  server_dataset_root: "data"
  runtime_dataset_root: "data"
  train_dir: "data"
  val_dir: "data"
  real_ratio: 0.5
  artifact_to_cifake_ratio: [3, 1]

loader:
  batch_size: 32
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: true
  real_fake_ratio: [1, 1]
  artifact_cifake_ratio: [3, 1]

model:
  backbone: resnet18
  rgb_pretrained: true
  noise_pretrained: false
  freq_pretrained: false
  fused_dim: 512
  classifier_hidden_dim: 256
  dropout: 0.3

train:
  epochs: 10
  lr: 1e-4
  weight_decay: 1e-4
  save_dir: "checkpoints"
  label_smoothing: 0.1
  grad_clip: 1.0
  use_ema: true
  ema_decay: 0.999
  temperature_init: 1.0
  bce_weight: 0.7
  focal_weight: 0.3
  focal_alpha: 0.25
  focal_gamma: 2.0
  lambda_rgb: 0.2
  lambda_freq: 0.2
  lambda_spatial: 0.2

logging:
  log_dir: "logs"
"""
        with open(config_path, "w") as f:
            f.write(config_content.strip())

def filter_func(tarinfo):
    name = tarinfo.name
    # Check if any part of the path matches exclude patterns
    parts = name.replace("\\", "/").split("/")
    
    # Special handling: 'data' is tricky because we have 'src/data' (code) and 'data/' (dataset)
    # If the path starts with 'data/', exclude it.
    # If the path is just 'data', exclude it.
    if name == "data" or name.startswith("data/") or name.startswith("data\\"):
        return None
        
    if any(p in parts for p in EXCLUDE_PATTERNS):
        return None
    # Also exclude specific files if needed, e.g. .DS_Store
    if name.endswith(".DS_Store"):
        return None
    return tarinfo

def create_archive(output_filename):
    create_default_config()
    print(f"Creating archive: {output_filename}")
    with tarfile.open(output_filename, "w:gz") as tar:
        for item in INCLUDE_PATHS:
            if os.path.exists(item):
                tar.add(item, arcname=item, filter=filter_func)
            else:
                print(f"Warning: {item} not found locally.")

def connect_ssh():
    print(f"Connecting to {SERVER_HOST}:{SERVER_PORT} as {SERVER_USER}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER_HOST, port=SERVER_PORT, username=SERVER_USER, password=SERVER_PASS)
    return client

def run_command(client, command, print_output=True):
    print(f"Executing: {command}")
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if print_output:
        if out: print(out)
        if err: print(f"Error output: {err}")
    return exit_status, out, err

def main():
    # 1. Package Project
    create_archive(LOCAL_ARCHIVE_NAME)
    
    # 2. Connect
    client = connect_ssh()
    sftp = client.open_sftp()
    
    # 3. Upload Archive
    print(f"Uploading {LOCAL_ARCHIVE_NAME} to {REMOTE_ARCHIVE_PATH}...")
    sftp.put(LOCAL_ARCHIVE_NAME, REMOTE_ARCHIVE_PATH)
    
    # 4. Deploy (Extract)
    print("Deploying on server...")
    run_command(client, f"mkdir -p {REMOTE_PROJECT_DIR}")
    # FORCE CLEAN: Remove old src directory to ensure no stale files or missing init files persist
    print("Cleaning old source files...")
    run_command(client, f"rm -rf {REMOTE_PROJECT_DIR}/src {REMOTE_PROJECT_DIR}/scripts {REMOTE_PROJECT_DIR}/backend {REMOTE_PROJECT_DIR}/configs")
    
    # Extract into the project directory
    run_command(client, f"tar -xzf {REMOTE_ARCHIVE_PATH} -C {REMOTE_PROJECT_DIR}")
    
    # 5. Install Dependencies
    print("Installing dependencies...")
    # Assuming pip is available. Check if requirements.txt exists remotely first?
    # It should be there after extraction.
    run_command(client, f"cd {REMOTE_PROJECT_DIR} && pip install -r requirements.txt")
    
    # 6. Check GPU Environment
    print("Checking GPU environment...")
    run_command(client, "nvidia-smi")
    _, out, _ = run_command(client, "python3 -c \"import torch; print(f'CUDA Available: {torch.cuda.is_available()}')\"")
    if "True" not in out:
        print("Warning: CUDA might not be available or PyTorch is not using GPU.")
    
    # 7. Check Dataset
    print("Checking dataset structure...")
    data_dir = f"{REMOTE_PROJECT_DIR}/data"
    artifact_dir = f"{data_dir}/artifact-dataset"
    cifake_dir = f"{data_dir}/cifake"
    
    # Check existence
    status, _, _ = run_command(client, f"[ -d '{artifact_dir}' ] && [ -d '{cifake_dir}' ]", print_output=False)
    
    if status != 0:
        print("\n" + "="*50)
        print("DATASET CHECK FAILED")
        print(f"Please manually upload dataset to: {data_dir}")
        print("Required structure:")
        print(f"  {data_dir}/")
        print("  ├── artifact-dataset/")
        print("  └── cifake/")
        print("="*50 + "\n")
        
        # Clean up local archive
        if os.path.exists(LOCAL_ARCHIVE_NAME):
            os.remove(LOCAL_ARCHIVE_NAME)
        client.close()
        return

    print("Dataset structure confirmed.")
    
    # 8. Start Training
    print("Starting training...")
    # Use nohup to run in background
    # Ensure logs are captured
    log_file = "train_remote.log"
    # Using python3 -m to run as module which handles pathing more robustly
    train_cmd = f"nohup python3 -u -m src.training.train > {log_file} 2>&1 & echo $!"
    
    status, pid, err = run_command(client, f"cd {REMOTE_PROJECT_DIR} && {train_cmd}")
    
    if status == 0:
        print(f"\nTraining started successfully! PID: {pid}")
        print(f"Logs are being written to: {REMOTE_PROJECT_DIR}/{log_file}")
        print(f"You can monitor training with: tail -f {REMOTE_PROJECT_DIR}/{log_file}")
    else:
        print(f"Failed to start training. Error: {err}")

    # Clean up local archive
    if os.path.exists(LOCAL_ARCHIVE_NAME):
        os.remove(LOCAL_ARCHIVE_NAME)
    
    client.close()

if __name__ == "__main__":
    main()
