import paramiko
import os
import sys
import time

# Configuration
HOST = "region-41.seetacloud.com"
PORT = 24888
USER = "root"
PASS = "BexyrhAT8rbN"

LOCAL_SCRIPT = "scripts/remote_fusion_worker.py"
REMOTE_SCRIPT_PATH = "/root/autodl-tmp/ai-image-detector/scripts/remote_fusion_worker.py"
REMOTE_PYTHON = "/root/miniconda3/bin/python" # Explicit python path commonly used in autodl

def run_remote_fusion():
    print(f"Connecting to {USER}@{HOST}:{PORT}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(HOST, port=PORT, username=USER, password=PASS)
        print("Connected.")
        
        # 1. Upload Worker Script
        sftp = client.open_sftp()
        try:
            # Ensure remote scripts dir exists
            try:
                sftp.mkdir("/root/autodl-tmp/ai-image-detector/scripts")
            except IOError:
                pass # Exists
                
            print(f"Uploading {LOCAL_SCRIPT} to {REMOTE_SCRIPT_PATH}...")
            # Normalize local path for Windows
            local_path = os.path.abspath(LOCAL_SCRIPT)
            sftp.put(local_path, REMOTE_SCRIPT_PATH)
            print("Upload complete.")
        finally:
            sftp.close()
            
        # 2. Execute Script
        # Check disk space and inodes
        print("Checking disk space and inodes...")
        stdin, stdout, stderr = client.exec_command("df -h /root/autodl-tmp && df -i /root/autodl-tmp")
        print(stdout.read().decode())

        cmd = f"{REMOTE_PYTHON} {REMOTE_SCRIPT_PATH}"
        print(f"Executing remote command: {cmd}")
        
        stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
        
        # Stream output
        while True:
            line = stdout.readline()
            if not line:
                break
            print(line, end="")
            
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print("Remote fusion completed successfully.")
        else:
            print(f"Remote fusion failed with exit code {exit_status}")
            print("Errors:")
            print(stderr.read().decode())
            
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    run_remote_fusion()
