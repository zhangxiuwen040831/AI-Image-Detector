# 云GPU训练指南

本指南详细说明如何在云GPU上训练模型，并将训练好的模型传回本地。

## 目录
1. [准备工作](#准备工作)
2. [云GPU环境配置](#云gpu环境配置)
3. [数据集上传](#数据集上传)
4. [开始训练](#开始训练)
5. [监控训练](#监控训练)
6. [下载模型到本地](#下载模型到本地)
7. [本地部署](#本地部署)

---

## 准备工作

### 1. 确认本地项目结构

确保您的本地项目包含以下内容：
```
ai-image-detector/
├── configs/
│   ├── cloud_gpu_config.yml  # 云GPU专用配置
│   └── demo_config.yml
├── src/
├── requirements.txt
└── ...
```

### 2. 准备云GPU服务器

确保您有云GPU服务器（如AutoDL、AWS、GCP等），并且：
- 已安装 Python 3.8+
- 有 GPU 访问权限
- 有足够的存储空间（至少 20GB）

---

## 云GPU环境配置

### 步骤1：连接到云GPU服务器

使用SSH连接到您的云GPU服务器：
```bash
# 替换为您的服务器信息
ssh -p 44495 root@region-42.seetacloud.com
# 输入密码：BexyrhAT8rbN
```

### 步骤2：在云GPU上克隆或上传项目

#### 选项A：从Git克隆（推荐）
```bash
# 如果您的代码在Git仓库
cd ~/autodl-tmp
git clone <your-repository-url> ai-image-detector
cd ai-image-detector
```

#### 选项B：从本地上传（如果没有Git）
在本地PowerShell执行：
```powershell
# 使用SCP上传项目（注意：会很慢，只适合小项目）
cd c:\Users\32902\Desktop\ai-image-detector
scp -P 44495 -r . root@region-42.seetacloud.com:~/autodl-tmp/ai-image-detector
```

### 步骤3：在云GPU上创建虚拟环境

```bash
# 进入项目目录
cd ~/autodl-tmp/ai-image-detector

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

---

## 数据集上传

### 步骤1：确认本地数据集路径

确保您的本地数据集在：
```
c:\Users\32902\Desktop\ai-image-detector\hf_datasets\data\genimage\
```

### 步骤2：上传数据集到云GPU

在本地PowerShell执行：
```powershell
cd c:\Users\32902\Desktop\ai-image-detector\hf_datasets

# 使用SCP上传数据集
scp -P 44495 -r data root@region-42.seetacloud.com:~/autodl-tmp/ai-image-detector/
```

或者使用rsync（如果可用）：
```powershell
rsync -avz -e "ssh -p 44495" data/ root@region-42.seetacloud.com:~/autodl-tmp/ai-image-detector/data/
```

### 步骤3：在云GPU上验证数据集

在云GPU服务器上执行：
```bash
cd ~/autodl-tmp/ai-image-detector

# 检查数据集结构
ls -la data/genimage/

# 应该看到：
# train/real/
# train/fake/
# val/real/
# val/fake/
```

---

## 开始训练

### 步骤1：修改配置文件（可选）

在云GPU上编辑云GPU配置：
```bash
cd ~/autodl-tmp/ai-image-detector

# 查看配置
cat configs/cloud_gpu_config.yml

# 如果需要修改，可以编辑
nano configs/cloud_gpu_config.yml
```

关键配置项：
- `device: cuda` - 使用GPU
- `batch_size: 32` - 根据GPU内存调整
- `num_epochs: 10` - 训练轮数
- `use_lora: true` - 使用LoRA加速
- `save_dir: checkpoints/cloud_gpu` - 模型保存路径

### 步骤2：启动训练

在云GPU上执行：
```bash
cd ~/autodl-tmp/ai-image-detector
source .venv/bin/activate

# 使用云GPU配置开始训练
python -m src.training.train_baseline --config configs/cloud_gpu_config.yml
```

### 步骤3：后台运行训练（推荐）

如果您需要断开SSH连接但保持训练运行：
```bash
# 使用screen
screen -S training

# 然后运行训练命令
cd ~/autodl-tmp/ai-image-detector
source .venv/bin/activate
python -m src.training.train_baseline --config configs/cloud_gpu_config.yml

# 按 Ctrl+A，然后 D 来分离screen

# 要重新连接
screen -r training
```

或者使用nohup：
```bash
cd ~/autodl-tmp/ai-image-detector
source .venv/bin/activate
nohup python -m src.training.train_baseline --config configs/cloud_gpu_config.yml > training.log 2>&1 &

# 查看日志
tail -f training.log
```

---

## 监控训练

### 1. 查看训练日志

```bash
# 如果使用nohup
tail -f training.log

# 如果在screen中，直接查看输出
```

### 2. 使用TensorBoard（可选）

如果配置了TensorBoard：
```bash
cd ~/autodl-tmp/ai-image-detector
source .venv/bin/activate
tensorboard --logdir logs/cloud_gpu --port 6006
```

然后在本地浏览器访问：
```
http://<your-server-ip>:6006
```

### 3. 监控GPU使用

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1
```

---

## 下载模型到本地

### 步骤1：确认训练完成

训练完成后，在云GPU上查看最佳模型：
```bash
cd ~/autodl-tmp/ai-image-detector

# 查看checkpoint目录
ls -la checkpoints/cloud_gpu/

# 应该看到类似：
# best_epoch_5.pth
# best_epoch_10.pth
```

### 步骤2：确定最佳模型

查看训练日志，找到验证准确率最高的epoch：
```bash
# 如果有training.log
grep "Saved new best checkpoint" training.log
```

或者查看最新的best文件：
```bash
ls -t checkpoints/cloud_gpu/best_epoch_*.pth | head -1
```

### 步骤3：下载模型到本地

在本地PowerShell执行：

#### 选项A：使用SCP直接下载
```powershell
cd c:\Users\32902\Desktop\ai-image-detector

# 创建本地checkpoints目录（如果不存在）
New-Item -ItemType Directory -Force -Path checkpoints\demo

# 下载最佳模型（替换为实际的文件名）
scp -P 44495 root@region-42.seetacloud.com:~/autodl-tmp/ai-image-detector/checkpoints/cloud_gpu/best_epoch_10.pth checkpoints/demo/best_epoch_1.pth
```

#### 选项B：使用rsync（如果可用）
```powershell
rsync -avz -e "ssh -p 44495" root@region-42.seetacloud.com:~/autodl-tmp/ai-image-detector/checkpoints/cloud_gpu/best_epoch_*.pth checkpoints/demo/
```

### 步骤4：验证本地模型

在本地检查模型是否下载成功：
```powershell
cd c:\Users\32902\Desktop\ai-image-detector

# 查看文件
ls checkpoints\demo\

# 应该看到：best_epoch_1.pth
```

---

## 本地部署

### 步骤1：更新配置（如果需要）

确认 `src/api/server.py` 中的模型路径正确：
```python
# DEFAULT_CKPT = "checkpoints/demo/best_epoch_1.pth"
```

### 步骤2：启动后端API

```powershell
cd c:\Users\32902\Desktop\ai-image-detector

# 激活虚拟环境
.\.venv\Scripts\activate

# 启动API服务器
python serve_api.py --host 0.0.0.0 --port 8000
```

### 步骤3：启动前端

```powershell
# 打开新的终端
cd c:\Users\32902\Desktop\ai-image-detector\web
python -m http.server 8080
```

### 步骤4：测试模型

在浏览器中打开 http://localhost:8080，上传图片测试模型！

---

## 常见问题

### Q1: 训练时GPU内存不足怎么办？
A: 减小 `configs/cloud_gpu_config.yml` 中的 `batch_size`（例如从32改为16或8）。

### Q2: 上传数据集太慢？
A: 先在本地压缩数据集，然后上传压缩包，在云GPU上解压：
```powershell
# 本地压缩
Compress-Archive -Path data -DestinationPath data.zip

# 上传
scp -P 44495 data.zip root@region-42.seetacloud.com:~/autodl-tmp/ai-image-detector/

# 云GPU上解压
cd ~/autodl-tmp/ai-image-detector
unzip data.zip
```

### Q3: 如何暂停和恢复训练？
A: 目前代码不支持热恢复。需要重新开始训练，或者使用checkpoint继续训练（需要修改代码）。

### Q4: 训练多久能完成？
A: 
- 10 epochs，RTX 2080 Ti，batch_size 32：约 30-60分钟
- 具体时间取决于数据集大小和GPU性能

---

## 快速命令参考

### 本地到云GPU
```powershell
# 上传项目
scp -P 44495 -r ai-image-detector root@region-42.seetacloud.com:~/autodl-tmp/

# 上传数据集
scp -P 44495 -r data root@region-42.seetacloud.com:~/autodl-tmp/ai-image-detector/
```

### 云GPU上
```bash
# 激活环境
cd ~/autodl-tmp/ai-image-detector
source .venv/bin/activate

# 开始训练
python -m src.training.train_baseline --config configs/cloud_gpu_config.yml
```

### 云GPU到本地
```powershell
# 下载模型
scp -P 44495 root@region-42.seetacloud.com:~/autodl-tmp/ai-image-detector/checkpoints/cloud_gpu/best_epoch_*.pth checkpoints/demo/best_epoch_1.pth
```

---

祝您训练顺利！🚀
