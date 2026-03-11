# AI Image Detector (Multi-Branch ResNet18)

基于多分支架构（RGB + 噪声 + 频域）的 AI 生成图像检测系统。本项目旨在通过融合多模态特征，高精度识别由 AIGC（AI Generated Content）模型生成的伪造图像。

当前版本采用 **ResNet18** 作为核心骨干网络，并在 **CIFAKE** 与 **Artifact** 混合数据集上取得了优异性能。

## 📊 核心性能

在混合数据集（CIFAKE + Artifact）上的测试结果（Epoch 10）：

- **准确率 (Accuracy)**: **87.31%**
- **AUC (Area Under Curve)**: **0.9496**
- **训练速度**: \~930 img/s (RTX 5090)

## 🧠 算法架构

本项目采用 **Multi-Branch Detector** 架构，支持以下特征分支的灵活组合：

1. **RGB 分支 (主分支)**:
   - **Backbone**: ResNet18 (预训练)
   - **作用**: 提取图像的高层语义特征和纹理细节。
   - **配置**: 默认启用 (`rgb_pretrained=True`)。
2. **噪声分支 (Noise Branch)**:
   - **作用**: 通过隐写分析滤波器（SRM/Bayar）提取图像的噪声残差，捕捉生成模型留下的指纹痕迹。
   - **状态**: 可选开启。
3. **频域分支 (Frequency Branch)**:
   - **作用**: 利用 DCT/FFT 变换分析图像在频域上的异常分布（如棋盘格效应）。
   - **状态**: 可选开启。

**融合策略**: 各分支特征经过提取后，在全连接层前进行拼接 (Concatenation)，并通过 MLP 进行最终分类。

## 📁 项目结构

```text
ai-image-detector/
├── configs/            # 配置文件
│   ├── dataset_config.yaml  # 核心训练配置 (Batch Size, LR, Paths)
│   └── ...
├── src/                # 源代码
│   ├── data/           # 数据加载与预处理
│   ├── models/         # 模型定义
│   │   ├── detector.py # MultiBranchDetector 实现
│   │   └── model.py    # AIGCImageDetector 封装
│   ├── training/       # 训练循环与 Trainer
│   └── utils/          # 工具函数
├── tools/              # 运维与部署脚本
│   ├── deploy.py       # 自动部署到云服务器
│   ├── package_code.py # 代码打包
│   └── monitor_*.py    # 训练监控脚本
├── frontend/           # 前端可视化项目
│   ├── src/
│   │   ├── components/ # 可视化组件
│   │   ├── App.jsx     # 主应用入口
│   │   └── index.css   # 全局样式
│   └── vite.config.js  # Vite 配置
├── backend/            # 后端推理服务
│   ├── main.py         # FastAPI 入口
│   └── detector.py     # 推理逻辑封装
└── README.md           # 项目文档
```

## 💻 前端技术文档

本项目前端采用 **React + Vite** 构建，专注于高性能的 AI 取证可视化分析。

### 1. 主界面文件

| 文件路径                     | 作用           | 关键配置与依赖                                                                                                                                                                                                                    |
| :----------------------- | :----------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `frontend/index.html`    | **HTML 入口**  | 引入 Google Fonts (Inter, JetBrains Mono)，挂载 React 根节点。                                                                                                                                                                      |
| `frontend/src/main.jsx`  | **React 入口** | 负责渲染 `<App />`，引入全局样式 `index.css`。使用 `ReactDOM.createRoot` 启用并发模式。                                                                                                                                                         |
| `frontend/src/App.jsx`   | **应用主容器**    | • **布局**: 包含 Header, UploadPanel, Analysis Dashboard, Footer。• **逻辑**: 处理文件上传、调用后端 API、管理分析状态 (`isAnalyzing`) 和结果数据。• **性能**: 使用 `React.lazy` 和 `Suspense` 懒加载所有图表组件，确保首屏秒开。• **动效**: 使用 `framer-motion` 实现页面级过渡和背景动态模糊效果。 |
| `frontend/src/index.css` | **全局样式**     | 基于 **Tailwind CSS**。定义了 `.glass` (毛玻璃), `.neon-glow` (霓虹光晕), `.text-gradient` 等自定义原子类。                                                                                                                                     |

### 2. 核心逻辑与分析模块

由于前端逻辑较为轻量，核心业务逻辑主要集中在组件内部和 `App.jsx` 中。

| 模块/文件                        | 职责         | 接口与数据流                                                                                               |
| :--------------------------- | :--------- | :--------------------------------------------------------------------------------------------------- |
| `App.jsx` (`handleUpload`)   | **API 通信** | • **输入**: `File` 对象• **调用**: `POST /detect`• **输出**: 更新 `result` 状态 (JSON)• **错误处理**: 捕获网络异常并显示错误提示。 |
| `components/UploadPanel.jsx` | **文件处理**   | • **职责**: 拖拽上传、文件类型校验 (Image only)、大小限制 (10MB)、生成本地预览 (FileReader)。• **交互**: 提供上传进度动画和拖拽反馈。          |

### 3. 图表与可视化组件

所有图表组件均位于 `frontend/src/components/` 目录下，采用 **Recharts** 进行数据可视化，支持响应式布局。

| 组件文件                      | 类型              | 库版本              | 功能与配置                                                                                      |
| :------------------------ | :-------------- | :--------------- | :----------------------------------------------------------------------------------------- |
| `ProbabilityChart.jsx`    | **环形图 (Donut)** | `recharts ^2.13` | • **展示**: 真/假概率分布。• **配置**: 自定义 Tooltip 样式，中心显示主概率数值。• **动效**: 初始加载时的扇区展开动画。               |
| `BranchContribution.jsx`  | **条形图 (Bar)**   | `recharts ^2.13` | • **展示**: RGB、噪声、频域三个分支的贡献度。• **配置**: 水平条形图，自定义颜色映射。• **适配**: `ResponsiveContainer` 自适应宽度。 |
| `NoiseResidualViewer.jsx` | **图像查看器**       | N/A              | • **展示**: SRM 噪声残差图 (Base64)。• **交互**: 点击放大查看像素级细节 (`pixelated` 渲染模式)，支持平滑过渡动画。            |
| `FrequencySpectrum.jsx`   | **图像查看器**       | N/A              | • **展示**: FFT 频谱热力图 (Base64)。• **交互**: 同上，提供频谱异常的可视化检查。                                    |
| `DetectionResult.jsx`     | **结果卡片**        | N/A              | • **展示**: 最终判定结果 (REAL/AIGC) 和置信度。• **动效**: 结果图标 (Check/Alert) 的弹跳动画和霓虹光效。                 |

### 4. 动画与交互

| 技术栈                | 用途    | 关键实现                                                                                                                                  |
| :----------------- | :---- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **Framer Motion**  | 组件级动画 | • **页面进入**: `initial={{ opacity: 0 }}` -> `animate={{ opacity: 1 }}`• **列表加载**: `AnimatePresence` 实现结果面板的平滑展开。• **上传反馈**: 进度条的无限循环动画。 |
| **CSS / Tailwind** | 装饰性动画 | • **背景**: 两个巨大的模糊圆球 (`blur-[120px]`) 在背景中缓慢脉冲 (`animate-pulse`)。• **光效**: `.neon-glow` 类实现的 CSS `box-shadow` 动画。                      |

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

项目支持自动融合 **CIFAKE** 和 **Artifact** 数据集。请确保数据放置在 `data/` 目录下，或在 `configs/dataset_config.yaml` 中配置 `local_dataset_root`。

### 3. 模型训练

**本地训练**:

```bash
python src/training/train.py --config configs/dataset_config.yaml
```

**云端部署与训练 (AutoDL/SeetaCloud)**:
本项目包含完整的云端部署工具链：

1. **打包代码**: `python tools/package_code.py`
2. **一键部署**: `python tools/deploy.py` (自动上传、解压、环境配置并启动训练)
3. **监控训练**: `python tools/monitor_training.py`

### 4. 启动可视化系统

**启动后端 (API)**:

```bash
python start_backend.py
# 服务运行在 http://localhost:8000
```

**启动前端 (Dashboard)**:

```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:5173
```

## 🛠️ 开发与维护

- **代码打包**: `tools/package_code.py` 会自动忽略 `data/`, `logs/`, `checkpoints/` 等大文件，仅打包核心代码。
- **训练日志**: 训练过程日志保存在 `logs/pipeline_train.log`。
- **模型权重**: 最佳模型保存在 `checkpoints/pipeline/best.pth`。

## 📝 引用与致谢

- CIFAKE Dataset
- Artifact Dataset
- ResNet: Deep Residual Learning for Image Recognition

