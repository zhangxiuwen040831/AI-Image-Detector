import paramiko

HOST = "region-41.seetacloud.com"
PORT = 24888
USER = "root"
PASS = "BexyrhAT8rbN"

def run_checks():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS)
    
    commands = [
        "nvidia-smi",
        "nvcc --version",
        "/root/miniconda3/bin/python -c 'import torch; print(\"Torch GPU:\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\")'",
        "/root/miniconda3/bin/python -c 'import torch; print(\"Torch version:\", torch.__version__)'",
        "/root/miniconda3/bin/python -c 'import deepspeed; print(\"DeepSpeed version:\", deepspeed.__version__)'",
        "ls -R /root/lanyun-tmp/ai-image-detector | head -n 20"
    ]
    
    for cmd in commands:
        print(f"--- Executing: {cmd} ---")
        stdin, stdout, stderr = client.exec_command(cmd)
        print(stdout.read().decode())
        err = stderr.read().decode()
        if err:
            print(f"Errors: {err}")
            
    client.close()

if __name__ == "__main__":
    run_checks()
