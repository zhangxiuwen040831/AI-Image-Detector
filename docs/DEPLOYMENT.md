# Deployment Guide

## 1. 环境要求

- Python `3.10+`
- Node.js `18+`
- npm `9+`
- 推荐使用 CUDA 环境进行训练或批量推理

安装依赖：

```bash
pip install -r requirements.txt
cd frontend && npm install
```

## 2. 模型准备

公开仓库默认不包含最终权重。请将模型文件放到：

```text
checkpoints/best.pth
```

默认相关配置：

- 推理配置：`configs/infer/default.yml`
- 部署配置：`configs/deploy/default.yml`
- API 服务入口：`services/api/main.py`
- 命令行推理入口：`scripts/infer_ntire.py`

## 3. 默认推理设置

- 版本：`V5.1`
- 模式：`base_only`
- 默认阈值档位：`balanced`
- 推荐阈值：
  - `recall-first = 0.20`
  - `balanced = 0.35`
  - `precision-first = 0.35`

如果你更关心误报控制，优先从 `balanced=0.35` 开始；如果你更关心召回，可优先使用 `0.20`。

## 4. 启动后端

```bash
python scripts/start_backend.py
```

或：

```bash
python -m uvicorn services.api.main:app --host 0.0.0.0 --port 8000
```

默认监听：`http://localhost:8000`

## 5. 启动前端

```bash
cd frontend
npm run dev
```

默认访问：`http://localhost:5173`

前端默认会请求 `http://localhost:8000/detect`。

## 6. API 使用说明

主接口：`POST /detect`

返回内容包括：

- `prediction`
- `probability`
- `threshold_used`
- `mode`
- `branch_contribution`
- `branch_evidence`
- `fusion_weights`
- `artifacts`
- `explanation`

## 7. 命令行推理示例

### 单图推理

```bash
python scripts/infer_ntire.py --image photos_test/aigc7.png --checkpoint checkpoints/best.pth
```

### 文件夹推理

```bash
python scripts/infer_ntire.py --folder photos_test --checkpoint checkpoints/best.pth --threshold-profile balanced
```

### 使用召回优先阈值

```bash
python scripts/infer_ntire.py --folder photos_test --checkpoint checkpoints/best.pth --threshold-profile recall-first
```

## 8. 训练与评估示例

```bash
python scripts/train_v10.py --data-root /path/to/NTIRE-RobustAIGenDetection-train --save-dir outputs/v10_experiment --pretrained-backbone
python scripts/evaluate_v10.py --checkpoint checkpoints/best.pth --data-root /path/to/NTIRE-RobustAIGenDetection-train --output-dir outputs/v10_eval
```

## 9. 已知限制

- 公开仓库不包含训练数据与最终模型权重。
- 当前公共说明默认围绕 `base_only` 展开；如果需要研究混合模式，请自行调整配置并重新校准阈值。
- 生产环境部署前，建议先在你自己的验证集上重扫阈值。
