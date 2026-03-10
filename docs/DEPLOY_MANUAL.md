# AI Image Detector 远程部署与高级训练手册

本手册详细记录了如何优化训练流程，并使用高性能云服务器进行大规模数据集训练。

## 1. 准备工作 (本地环境)

### 1.1 SSH 连接信息
*   **地址**: `region-41.seetacloud.com`
*   **端口**: `23705`
*   **用户**: `root`
*   **密码**: `BexyrhAT8rbN`

### 1.2 数据集预处理与质量检查
在上传前，必须确保本地数据集格式正确且无损坏文件。

1.  **数据清洗**:
    运行以下脚本扫描并记录损坏图片：
    ```powershell
    python data_management/verify_data.py --data_dir ./data/cifake
    ```
2.  **打包压缩**:
    ```powershell
    tar -czf cifake.tar.gz data/cifake
    ```

3.  **本地代码验证 (可选但推荐)**:
    在本地 PowerShell 中运行以下命令，确保环境无误：
    *注意：本地验证请确保 configs/demo_config.yml 中 device 为 cpu*
    ```powershell
    python -m src.training.train_baseline --config configs/demo_config.yml
    ```

## 2. 远程部署与环境配置

### 2.1 上传代码与数据 (本地执行)
```powershell
# 上传代码
tar -czf project_code.tar.gz src configs data_management requirements.txt
scp -P 23705 project_code.tar.gz root@region-41.seetacloud.com:/root/autodl-tmp/ai-image-detector/

# 上传数据
scp -P 23705 cifake.tar.gz root@region-41.seetacloud.com:/root/autodl-tmp/ai-image-detector/
```

### 2.2 云端初始化 (云端执行)
```bash
cd /root/autodl-tmp/ai-image-detector/
tar -xzf project_code.tar.gz
tar -xzf cifake.tar.gz

# 安装依赖
pip install -r requirements.txt
pip install open_clip_torch
```

## 3. 高级训练策略

项目已配置支持以下优化策略：
*   **混合精度训练 (AMP)**: 减少显存占用，提升训练速度。
*   **学习率调度 (CosineAnnealing)**: 动态调整学习率以实现更好收敛。
*   **LoRA 微调**: 针对 ViT-L-14 进行高效参数微调。
*   **数据增强**: 包含 JPEG 压缩和高斯模糊，模拟真实网络环境。

**启动训练**:
```bash
# 使用优化后的配置文件
nohup python -m src.training.train_baseline --config configs/cloud_gpu_config.yml > logs/cloud_gpu.log 2>&1 &
```

## 4. 模型回传与本地部署

### 4.1 获取最优模型 (本地执行)
训练完成后，将最优权重、配置文件和标签映射表传回本地：

```powershell
# 创建本地存储目录
mkdir -p ./deploy/model

# 下载权重
scp -P 23705 root@region-41.seetacloud.com:/root/autodl-tmp/ai-image-detector/checkpoints/cloud_gpu/best_epoch_*.pth ./deploy/model/

# 下载配置
scp -P 23705 root@region-41.seetacloud.com:/root/autodl-tmp/ai-image-detector/configs/cloud_gpu_config.yml ./deploy/model/
```

### 4.2 本地验证
在本地运行推理脚本验证准确率：
```powershell
python -m src.eval.eval_robustness --model_path ./deploy/model/best_epoch_X.pth --config ./deploy/model/cloud_gpu_config.yml
```

### 4.3 启动本地 Web 服务
模型验证无误后，启动本地 Web API 服务，系统会自动加载 `./deploy/model` 目录下最新的模型文件。

```powershell
# 安装 Web 依赖 (如果尚未安装)
pip install fastapi uvicorn python-multipart

# 启动服务
python run_web.py --model_dir ./deploy/model --port 8000
```

服务启动后，访问 `http://localhost:8000/docs` 即可使用 Swagger UI 进行图片检测测试。

### 4.4 回滚方案
1.  **版本保留**: 每次部署前，将旧模型文件夹重命名为 `model_backup_YYYYMMDD`。
2.  **快速切换**: 若新模型识别率下降，修改本地 Web 项目的路径配置，指向备份文件夹即可实现秒级回滚。

---
**维护记录**:
- 2024-05-20: 升级至 region-41 服务器，启用 AMP 和 LoRA 优化。
