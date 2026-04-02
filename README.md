# AI Image Detector

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=0b0f19)
![Release](https://img.shields.io/badge/Public%20Release-V5.1-2ea44f)
![Mode](https://img.shields.io/badge/Inference-base__only-0f766e)

一个面向真实场景的 AI 生成图像检测项目，覆盖训练、评估、推理、FastAPI 后端、React 前端，以及论文/答辩材料整理。

当前对外发布版本统一命名为 **V5.1**。需要说明的是，本地最终训练路线已经演进到 **V10**，但公开仓库不按内部迭代号逐版发布，因此本次继续以 GitHub 公共版本号 `V5.1` 对齐最终可交付工程状态。

## 当前版本定位

- 当前公开版本：`V5.1`
- 当前默认推理模式：`base_only`
- 当前推荐阈值：`recall-first=0.20`、`balanced=0.35`、`precision-first=0.35`
- 当前默认模型路径：`checkpoints/best.pth`
- 当前默认骨干：`vit_base_patch16_clip_224.openai`

## 项目亮点

- 训练、评估、推理、前后端展示和文档整理已经形成完整工程闭环。
- 当前部署默认走 `semantic + frequency` 主路径，`noise` 分支仅保留为辅助诊断证据。
- 公开仓库保留了论文材料、图表和演示资源，适合项目展示、复现和二次开发。
- V5.1 重点清理了大文件权重、旧版 v9 脚本和内部 smoke/cloud 辅助脚本，让仓库更适合公开维护。

## 性能快照

以下结果来自最终 V10 路线在 `photos_test` 上的代表性表现，用于说明当前公开版本的默认部署效果。

| 场景 | Threshold | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 召回优先 | `0.20` | `0.8182` | `1.0000` | `0.9000` | `2` | `0` |
| 默认平衡 | `0.35` | `1.0000` | `1.0000` | `1.0000` | `0` | `0` |

## 系统截图

| 主界面 | 检测分析页 |
| --- | --- |
| ![主界面](docs/assets/main_interface.png) | ![检测分析页](docs/assets/analysis_panel.png) |

| 文档引导页 |
| --- |
| ![文档引导页](docs/assets/documentation_intro.png) |

## 项目目录结构

```text
AI-Image-Detector/
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── pyproject.toml
├── checkpoints/               # 权重放置说明（默认不提交大权重）
├── configs/                   # 训练 / 评估 / 推理 / 部署配置
├── docs/                      # 发布说明、部署文档、模型卡、论文材料
├── figures/                   # 报告图表与流程图
├── frontend/                  # React + Vite 前端
├── photos/                    # 演示图片
├── photos_test/               # 轻量验证样例
├── scripts/                   # 训练、评估、推理、报告脚本
├── services/api/              # FastAPI 服务
└── src/ai_image_detector/     # 核心算法与推理逻辑
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
cd frontend && npm install
```

### 2. 准备模型权重

公开仓库默认不包含大体积权重文件。请将最终模型放到：

```text
checkpoints/best.pth
```

详细说明见 [checkpoints/README.md](checkpoints/README.md) 和 [docs/MODEL_CARD.md](docs/MODEL_CARD.md)。

### 3. 启动后端

```bash
python scripts/start_backend.py
```

或：

```bash
python -m uvicorn services.api.main:app --host 0.0.0.0 --port 8000
```

### 4. 启动前端

```bash
cd frontend
npm run dev
```

默认访问地址：

- 前端：`http://localhost:5173`
- 后端：`http://localhost:8000`

## 推理接口说明

- 主接口：`POST /detect`
- 输入：单张图像文件
- 输出：类别、概率、阈值、分支贡献、解释性图像与调试字段
- 默认模式：`base_only`
- 默认阈值档位：`balanced`

命令行推理示例：

```bash
python scripts/infer_ntire.py --image photos_test/aigc7.png --checkpoint checkpoints/best.pth
python scripts/infer_ntire.py --folder photos_test --checkpoint checkpoints/best.pth --threshold-profile recall-first
```

## 常用训练与评估命令

```bash
python scripts/train_v10.py --data-root /path/to/NTIRE-RobustAIGenDetection-train --save-dir outputs/v10_experiment --pretrained-backbone
python scripts/evaluate_v10.py --checkpoint checkpoints/best.pth --data-root /path/to/NTIRE-RobustAIGenDetection-train --output-dir outputs/v10_eval
python scripts/generate_report_artifacts.py
```

## 文档导航

- [CHANGELOG.md](CHANGELOG.md)：版本变更记录
- [docs/FINAL_UPDATE.md](docs/FINAL_UPDATE.md)：V5.1 发布总览
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)：前后端与模型部署说明
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md)：默认模型、模式、阈值与限制
- [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)：项目维护与复现指南
- [docs/releases/v5.1.md](docs/releases/v5.1.md)：V5.1 release notes
- [docs/Thesis/README.md](docs/Thesis/README.md)：论文与答辩材料入口

## 更新日志入口

- 当前发布说明：`docs/releases/v5.1.md`
- 版本历史：`CHANGELOG.md`
- 上一公开整理版本：`v5.0.0`

## 已知限制

- 公开仓库不直接提供最终模型权重与训练数据集。
- 当前默认部署模式为 `base_only`；如果需要研究噪声分支影响，请自行切换配置并复核阈值。
- 当前代表性结果主要来自 `photos_test` 与论文材料，跨数据集泛化仍建议自行复验。
- 公开版保留了 V10 主线脚本，但没有继续公开中间 v8/v9/v10 的每一次内部试验快照。

## 公开仓库策略

- 历史 release 与历史 tag 保留，不做删除。
- 默认分支仅保留当前可运行代码、轻量样例和关键文档。
- 大型训练缓存、临时输出、重复 checkpoint 和一次性云端辅助脚本不进入当前公开版。
- 如需完整训练，请自行准备数据集和权重，并参考 `docs/DEPLOYMENT.md` 与 `docs/PROJECT_GUIDE.md`。
