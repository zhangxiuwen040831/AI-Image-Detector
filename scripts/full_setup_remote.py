import paramiko
import os
import time

HOST = "region-41.seetacloud.com"
PORT = 24888
USER = "root"
PASS = "BexyrhAT8rbN"

REMOTE_DIR = "/root/autodl-tmp/ai-image-detector"
BACKUP_DIR = f"/root/autodl-tmp/ai-image-detector_backup_{int(time.time())}"

def run_remote_command(client, cmd):
    print(f"Executing: {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out: print(out)
    if err: print(f"Errors: {err}")
    return out, err

def setup_remote():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS)
    
    # 1. Backup
    print("Backing up remote directory...")
    run_remote_command(client, f"mv {REMOTE_DIR} {BACKUP_DIR} || true")
    run_remote_command(client, f"mkdir -p {REMOTE_DIR}")
    
    # 2. Upload Code (excluding data and large artifacts)
    print("Uploading code...")
    sftp = client.open_sftp()
    
    local_root = os.getcwd()
    exclude_dirs = {'.git', '.idea', 'data', 'coverage', '__pycache__', 'node_modules', '.vscode'}
    exclude_files = {'.DS_Store', 'cifake.zip'}
    
    for root, dirs, files in os.walk(local_root):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # Create remote directories
        rel_path = os.path.relpath(root, local_root)
        if rel_path == ".":
            remote_path = REMOTE_DIR
        else:
            remote_path = os.path.join(REMOTE_DIR, rel_path).replace("\\", "/")
            try:
                sftp.mkdir(remote_path)
            except IOError:
                pass
        
        for file in files:
            if file in exclude_files: continue
            local_file = os.path.join(root, file)
            remote_file = os.path.join(remote_path, file).replace("\\", "/")
            print(f"Uploading {file}...")
            sftp.put(local_file, remote_file)
            
    sftp.close()
    
    # 3. Move data back from backup to avoid re-fusion if needed?
    # Actually, we are re-fusing into mix_data_v2, so we need the original data.
    print("Restoring data from backup...")
    run_remote_command(client, f"ln -s {BACKUP_DIR}/data {REMOTE_DIR}/data || mv {BACKUP_DIR}/data {REMOTE_DIR}/data")
    
    # 4. Install dependencies
    print("Installing dependencies...")
    run_remote_command(client, "/root/miniconda3/bin/pip install torch torchvision torchaudio")
    run_remote_command(client, "/root/miniconda3/bin/pip install deepspeed accelerate transformers peft bitsandbytes flash-attn")
    run_remote_command(client, "/root/miniconda3/bin/pip install -r " + REMOTE_DIR + "/requirements.txt")
    
    client.close()
    print("Setup complete.")

if __name__ == "__main__":
    setup_remote()
