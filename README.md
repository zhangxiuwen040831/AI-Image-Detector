# AIGC 图像检测系统

基于 CLIP ViT-L/14 与轻量微调（LoRA、正交子空间分解）的 AI 生成图像检测系统。支持跨数据集评估、鲁棒性测试、用户管理、REST API 以及现代化的 React Web 界面。采用 Focal Loss + Metric Loss + Orthogonality Loss 混合损失函数，提高模型检测性能。

## 🚀 核心功能

- **AI 图像检测**：高精度识别图像是否为 AI 生成。
- **可解释性可视化**：生成热力图（CAM），展示模型关注区域。
- **用户管理系统**：完整的注册、登录、权限管理及检测历史记录。
- **高性能训练**：支持混合精度训练（AMP）、LoRA 微调、度量学习（Triplet Loss）及多种数据增强策略。
- **数据集融合**：支持 cifake 数据集与 artifact-dataset 数据集的有效融合。
- **现代化 UI**：基于 React + Vite + Tailwind CSS 的响应式前端界面。
- **模块化架构**：清晰的代码结构，易于维护和扩展。

## 📁 项目结构

```text
ai-image-detector/
├── adversarial/        # 对抗样本生成
├── checkpoints/        # 模型检查点目录
│   └── demo/           # 演示模型检查点
├── configs/            # 核心配置文件（数据库、模型、训练参数）
│   ├── cloud_gpu_config.yml
│   ├── cloud_gpu_fast_config.yml
│   ├── cloud_gpu_optimized.yml
│   ├── database.yml
│   └── demo_config.yml
├── data/               # 数据集存放目录
│   └── cifake/         # CIFAKE 数据集
├── data_management/    # 数据管理相关
│   ├── README.md
│   └── requirements.txt
├── datasets/           # 数据集加载器
├── docs/               # 详细文档目录
│   ├── DEPLOY_MANUAL.md
│   ├── QUICK_START_GUIDE.md
│   ├── USER_MANAGEMENT_SETUP_GUIDE.md
│   ├── cloud_gpu_huggingface_guide.md
│   ├── cloud_gpu_training_guide.md
│   └── 使用说明.md
├── evaluation/         # 评估和测试代码
├── explainability/     # 可解释性分析
├── frontend/           # 前端代码
├── models/             # 模型定义和架构
├── scripts/            # 统一脚本目录
│   └── deploy_and_run_fusion.py  # 部署和运行融合模型脚本
├── src/                # 核心源代码
├── training/           # 训练策略与优化
├── utils/              # 工具类
├── .idea/              # IDE 配置（未提交）
├── components.json     # UI 组件配置
├── docker-compose.yml  # Docker Compose 配置
├── Dockerfile          # 容器化部署配置
├── package.json        # 前端依赖与构建脚本
├── package-lock.json   # 前端依赖锁定文件
├── postcss.config.js   # PostCSS 配置
├── README.md           # 本文档
├── requirements.txt    # 后端 Python 依赖
├── tailwind.config.js  # Tailwind CSS 配置
├── tsconfig.json       # TypeScript 配置
├── tsconfig.node.json  # TypeScript Node 配置
└── vite.config.ts      # Vite 构建配置
```

## 🧠 核心算法

### 1. CLIP ViT-L/14 骨干网络
- **原理**：CLIP (Contrastive Language-Image Pre-training) 是一种多模态预训练模型，通过对比学习同时理解图像和文本。ViT-L/14 是一个大型视觉Transformer模型，使用14x14的补丁大小。
- **应用**：作为特征提取器，将输入图像转换为高维特征向量，利用其强大的图像理解能力识别AI生成图像的特征。
- **优势**：预训练模型拥有丰富的视觉知识，能够捕捉到AI生成图像与真实图像之间的细微差异。

### 2. LoRA (Low-Rank Adaptation) 微调
- **原理**：LoRA通过在原始模型的线性层中注入低秩矩阵，实现参数高效的微调。具体来说，对于每个线性层 W，LoRA引入两个低秩矩阵 A 和 B，使得 W' = W + BA。
- **实现**：在 `src/models/lora.py` 中实现，支持对CLIP模型的关键层进行LoRA微调。
- **优势**：大幅减少可训练参数数量，降低内存需求，同时保持模型性能。

### 3. 正交子空间分解 (OSD)
- **原理**：将特征空间分解为真实图像子空间和AI生成图像子空间，通过正交性约束增强特征的判别能力。
- **实现**：在 `src/models/osd.py` 中实现，包括正交性损失函数。
- **优势**：通过强制两个子空间正交，提高模型对AI生成图像的识别能力，特别是在面对新的生成模型时。

### 4. 混合损失函数
- **Focal Loss**：解决类别不平衡问题，通过降低易分类样本的权重，专注于难分类样本。
- **Metric Loss (Triplet Loss)**：通过三元组损失（锚点、正样本、负样本）增强特征的判别能力。
- **Orthogonality Loss**：确保真实图像和AI生成图像的特征子空间保持正交。
- **实现**：在 `src/training/losses.py` 中实现，通过加权组合这三种损失函数。

### 5. 数据集融合策略
- **原理**：将 CIFAKE 数据集与 artifact-dataset 数据集融合，利用不同数据集的特点提高模型的泛化能力。
- **实现**：在 `scripts/deploy_and_run_fusion.py` 中实现，支持数据集的自动下载、预处理和融合。
- **优势**：扩大训练数据规模，覆盖更多AI生成图像的类型和特征，提高模型的鲁棒性。

### 6. 可解释性分析
- **原理**：使用类激活映射 (CAM) 技术，可视化模型在预测时关注的图像区域。
- **实现**：在 `src/explain/cam.py` 中实现，生成热力图展示模型关注区域。
- **优势**：帮助理解模型的决策过程，提高模型的可信度和可解释性。

## 🛠️ 快速开始

### 1. 环境准备

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 安装后端依赖
pip install -r requirements.txt

# 安装前端依赖
npm install
```

### 2. 数据库配置

1. 初始化数据库：
   ```bash
   mysql -u root -p < data_management/complete_schema.sql
   ```
2. 编辑 `configs/database.yml` 配置实际的连接信息和 `secret_key`。

### 3. 运行服务

```bash
# 启动后端 API (默认 8000 端口)
python scripts/serve_api_with_auth.py

# 启动前端开发服务器
npm run dev
```

## 🧪 训练与评估 (云端/本地)

项目提供了一个核心脚本，用于部署和运行融合模型：

### 核心脚本说明

#### 1. deploy_and_run_fusion.py - 部署和运行融合模型脚本
**功能：**
- 数据集融合（cifake + artifact-dataset）
- 模型训练和评估
- 部署和推理

**使用方法：**
```bash
# 运行融合模型部署和训练
python scripts/deploy_and_run_fusion.py

# 仅运行推理
python scripts/deploy_and_run_fusion.py --infer

# 显示帮助
python scripts/deploy_and_run_fusion.py --help
```

## 📂 目录详细说明

### configs/
存放所有配置文件，包括数据库连接、模型配置、训练参数等。
- `cloud_gpu_config.yml`: 云端 GPU 标准训练配置
- `cloud_gpu_fast_config.yml`: 快速训练配置（较少 epoch）
- `cloud_gpu_optimized.yml`: 优化训练配置
- `database.yml`: 数据库连接配置
- `demo_config.yml`: 本地演示配置

### scripts/
脚本目录，包含核心脚本：
- `deploy_and_run_fusion.py`: 部署和运行融合模型脚本

### docs/
详细文档目录
- `DEPLOY_MANUAL.md`: 部署手册
- `QUICK_START_GUIDE.md`: 快速开始指南
- `USER_MANAGEMENT_SETUP_GUIDE.md`: 用户管理设置指南
- `cloud_gpu_training_guide.md`: 云端 GPU 训练指南
- `cloud_gpu_huggingface_guide.md`: HuggingFace 配置指南
- `使用说明.md`: 脚本使用说明

## 🔒 安全与技术特性

- **安全认证**：使用 bcrypt 进行密码哈希，JWT 风格的会话管理。
- **防御加固**：参数化查询防止 SQL 注入，支持 CSRF 防护。
- **模型优化**：采用 LoRA 技术，在保持 CLIP 强大特征提取能力的同时实现极低资源占用的微调。
- **模块化架构**：清晰的代码结构，易于维护和扩展。
- **统一配置管理**：集中式配置管理，简化配置变更。