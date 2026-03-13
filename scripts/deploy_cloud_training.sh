
#!/bin/bash
set -e

# Configuration
DATA_ROOT="/root/autodl-tmp/ai-image-detector/NTIRE-RobustAIGenDetection-train"
SAVE_DIR="/root/autodl-tmp/checkpoints/ntire_final_v1"
EPOCHS=20
BATCH_SIZE=48
IMAGE_SIZE=224
NUM_WORKERS=12
BACKBONE="vit_base_patch16_clip_224.openai"
MIXUP_PROB=0.3

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}>>> Starting NTIRE 2026 Formal Training Deployment on 5090 GPU <<<${NC}"

# 1. Environment Check
echo -e "${GREEN}[1/5] Checking GPU Environment...${NC}"
nvidia-smi
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# 2. Dependency Check
echo -e "${GREEN}[2/5] Ensuring Dependencies...${NC}"
pip install -r requirements.txt > /dev/null 2>&1 || echo "Dependencies might be already installed or requirements.txt missing"

# 3. Data Check
echo -e "${GREEN}[3/5] Verifying Data Path...${NC}"
if [ -d "$DATA_ROOT" ]; then
    echo "Data root found: $DATA_ROOT"
    ls -F "$DATA_ROOT" | head -n 5
else
    echo "ERROR: Data root not found at $DATA_ROOT"
    exit 1
fi

# 4. Launch Training
echo -e "${GREEN}[4/5] Launching Training Job...${NC}"
mkdir -p "$SAVE_DIR"

# Set HF Mirror
export HF_ENDPOINT=https://hf-mirror.com

cmd="nohup python scripts/train_ntire.py \
  --data-root $DATA_ROOT \
  --save-dir $SAVE_DIR \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --image-size $IMAGE_SIZE \
  --num-workers $NUM_WORKERS \
  --backbone-name $BACKBONE \
  --pretrained-backbone \
  --use-balanced-sampler \
  --use-ema \
  --mixup-prob $MIXUP_PROB \
  --check-val-every-n-epoch 1 \
  > training.log 2>&1 &"

echo "Command: $cmd"
eval $cmd

PID=$!
echo -e "${GREEN}Training started with PID: $PID${NC}"
echo "Logs are being written to: training.log"

# 5. Monitor Info
echo -e "${GREEN}[5/5] Deployment Complete.${NC}"
echo "To monitor logs: tail -f training.log"
echo "To monitor GPU:  watch -n 1 nvidia-smi"
echo "To stop training: kill $PID"

