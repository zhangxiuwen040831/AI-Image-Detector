# AI Image Detector 更新公告

## 🚀 V3.1 版本更新 (2026-03-14)

### 核心更新：文档质量全面提升

#### README 专业优化
- **章节美化**：为所有章节添加序号和图标，提升视觉层次
- **结构优化**：重新组织章节结构，使其更符合专业开源项目标准
- **内容准确性**：基于代码审计结果，修正不准确描述，确保与项目真实状态一致
- **数据集明确**：明确说明训练数据集为 `NTIRE-RobustAIGenDetection-train`
- **新增章节**：添加 Backend API 详细说明、架构图说明等

#### 新增架构文档
- **创建 docs/architecture.md**：包含 4 张完整的 Mermaid 架构图
  - System Architecture - 系统整体架构图
  - Training Pipeline - 训练流程图
  - Inference Pipeline - 推理流程图
  - Code Architecture - 代码模块架构图
- **docs/assets/**：创建资源目录，用于存放架构图资源

#### README 审计与重写
- **删除不准确内容**：移除 CIFAKE/Artifact 等旧数据集描述
- **移除无依据内容**：删除 Benchmark、Roadmap、Citation 等无真实数据的章节
- **明确 NTIRE 支持**：将 NTIRE 作为正式支持部分，非归档内容
- **补充 GradCAM 说明**：添加 `test_gradcam.py` 工具的使用说明
- **完善安装指南**：补充 Frontend 依赖安装步骤

---

## 🚀 V3.0 版本更新 (2026-03-13)

### 核心更新：NTIRE 2026 鲁棒性检测算法

#### 新数据集
- **数据集更换**：从 ArtiFact/CIFAKE 迁移至 **NTIRE 2026 Robust AIGC Detection** 数据集
- **数据特点**：专为野外鲁棒性设计，更好地对齐当前挑战分布
- **数据格式**：支持 shard 格式，自动扫描 `shard_*` 目录
- **数据增强**：针对社交媒体重新编码、平台调整大小、光学/后处理模糊等场景优化

#### 新算法架构
- **全局语义分支**：采用 timm backbone，默认 CLIP ViT (`vit_base_patch16_clip_224.openai`)
- **频域分支**：FFT 对数幅度 + 轻量级 CNN 编码器
- **噪声/伪影分支**：SRM 风格残差滤波 + 轻量级 CNN 编码器
- **融合策略**：门控 softmax 加权，跨三分支融合
- **可解释性**：返回每个样本的融合权重

#### 训练策略优化
- **数据增强**：
  - JPEG 压缩：提高对社交媒体重新编码的鲁棒性
  - 调整大小/重新缩放：提高对平台调整大小的鲁棒性
  - 高斯模糊：提高对光学/后处理模糊的鲁棒性
  - 高斯/ISO 噪声：提高对传感器和管道噪声的鲁棒性
  - 轻微颜色抖动：避免对精确颜色统计的过拟合
- **采样策略**：
  - 主要平衡：真实/虚假类别平衡
  - 次要平衡：失真/来源（如果可用）
  - 回退：shard 平衡近似
- **损失函数**：BCEWithLogitsLoss + 可选 focal 项 + 轻量级辅助分支监督
- **多尺度推理**：支持多尺度（如 224、336）和可选水平翻转 TTA

#### 性能指标 (NTIRE 2026)
- **准确率**: 87.31% (NTIRE 2026 训练集)
- **AUC**: 0.9496
- **训练速度**: ~930 img/s (RTX 5090)
- **推理速度**: CPU ~0.5s/张，GPU ~0.1s/张

#### 新增文件
- `src/ai_image_detector/ntire/dataset.py` - NTIRE 数据集加载器
- `src/ai_image_detector/ntire/augmentations.py` - 鲁棒性增强策略
- `src/ai_image_detector/ntire/model.py` - 升级的多分支模型架构
- `src/ai_image_detector/ntire/losses.py` - 损失函数实现
- `src/ai_image_detector/ntire/trainer.py` - NTIRE 训练器
- `src/ai_image_detector/ntire/metrics.py` - 评估指标
- `src/ai_image_detector/ntire/calibration.py` - 温度缩放校准
- `scripts/train_ntire.py` - NTIRE 训练入口
- `scripts/evaluate_ntire.py` - NTIRE 评估入口
- `scripts/infer_ntire.py` - NTIRE 推理入口
- `scripts/make_tiny_subset.py` - 创建小型子集用于本地测试
- `scripts/smoke_test_cpu.py` - CPU 快速测试

---

## 🚀 V2.0 版本更新 (之前版本)

### 前端增强
- **新增多语言支持** - 实现中文/英文切换功能
- **注意力热力图** - 新增 Grad-CAM 可视化，展示模型关注区域
- **解释报告** - 生成详细的检测结果解释
- **引导动画** - 新增启动引导页，提升用户体验
- **文档页面** - 新增项目架构说明页面
- **响应式设计** - 优化不同屏幕尺寸的显示效果
- **流畅动画** - 使用 Framer Motion 实现平滑过渡效果

### 后端改进
- **Grad-CAM 支持** - 实现模型注意力可视化
- **详细错误处理** - 完善的异常捕获和日志记录
- **多分支贡献度计算** - 精确计算各分支的贡献权重
- **调试模式** - 新增详细的调试信息输出
- **CORS 配置** - 优化跨域请求支持
- **解释服务** - 新增检测结果解释生成

### 模型优化
- **多分支融合** - RGB + 噪声 + 频域特征融合
- **多 Backbone 支持** - 兼容 ResNet18、EfficientNet、ConvNeXt
- **设备自动选择** - 智能切换 CPU/GPU 模式
- **模型权重加载** - 支持不同格式的权重文件

## 📁 新增文件

### 前端
- `frontend/src/components/Documentation.jsx` - 项目文档页面
- `frontend/src/components/ShieldAnimation.jsx` - 盾牌动画组件
- `frontend/src/components/IntroAnimation.jsx` - 引导页动画
- `frontend/src/components/ExplanationReport.jsx` - 解释报告组件
- `frontend/src/components/AttentionHeatmap.jsx` - 热力图组件

### 后端
- `backend/explanation_service.py` - 解释报告生成服务
- `backend/transforms.py` - 图像处理转换工具

## 🛠️ 技术栈更新

### 前端
- React 18.3.1
- Framer Motion 11.11.10
- Recharts 2.13.0
- Tailwind CSS 3.4.14
- Lucide React 0.454.0

### 后端
- FastAPI 0.110.0+
- PyTorch 2.0.0+
- ONNX Runtime 1.16.0+
- timm 0.9.12+ (新增)

## 📊 性能指标对比

| 版本 | 数据集 | 准确率 | AUC |
|------|--------|--------|-----|
| V1.0 | CIFAKE | 94.8% | 0.98 |
| V2.0 | Artifact | 87.3% | 0.95 |
| **V3.0** | **NTIRE 2026** | **87.31%** | **0.9496** |

## 🚦 部署说明

### 数据集下载
```bash
# 下载 NTIRE 2026 验证集和测试集
huggingface-cli download deepfakesMSU/NTIRE-RobustAIGenDetection-val --repo-type dataset --local-dir NTIRE-RobustAIGenDetection-val
huggingface-cli download deepfakesMSU/NTIRE-RobustAIGenDetection-test --repo-type dataset --local-dir NTIRE-RobustAIGenDetection-test
```

### 后端启动
```bash
python scripts/start_backend.py
# 服务运行在 http://localhost:8000
```

### 前端启动
```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:5173
```

### NTIRE 训练
```bash
# CPU 快速测试
python scripts/smoke_test_cpu.py --freeze-backbone --batch-size 4 --num-workers 0 --image-size 160

# 单 GPU 训练
python scripts/train_ntire.py --shards 0,1,2 --image-size 224 --batch-size 24 --epochs 20 --lr 3e-4

# 多 GPU 训练
python scripts/train_ntire.py --shards 0,1,2,3,4,5 --data-parallel --image-size 224 --batch-size 24 --epochs 30
```

### NTIRE 推理
```bash
# 单张图像推理
python scripts/infer_ntire.py --image example.jpg --checkpoint checkpoints/best.pth --scales 224 336

# 文件夹推理
python scripts/infer_ntire.py --input_dir input_folder --output_dir output_folder --checkpoint checkpoints/best.pth
```

## 🔧 环境要求

- **前端**: Node.js 16+, npm 7+
- **后端**: Python 3.8+, CUDA 11.8+ (推荐)
- **硬件**: 至少 8GB 内存，GPU 推荐
- **新增依赖**: timm, huggingface-hub

## 🎯 核心特性

- **多模态检测** - 融合 RGB、噪声、频域特征
- **可视化分析** - 热力图、频谱图、噪声残差图
- **用户友好** - 直观的界面和流畅的交互
- **可扩展** - 支持不同模型和配置
- **高性能** - 优化的推理速度
- **鲁棒性** - 针对真实世界场景优化
- **专业文档** - 高质量 README 和架构图

## 🔄 变更总结

### V3.1 主要变更
1. **文档质量提升**：全面审计并重写 README，确保与代码一致
2. **新增架构文档**：创建 docs/architecture.md，包含 4 张 Mermaid 架构图
3. **README 美化**：为所有章节添加序号和图标，提升可读性
4. **内容准确性**：删除不准确内容，明确 NTIRE 支持
5. **架构图集成**：在 README 中集成 Mermaid 架构图

### V3.0 主要变更
1. **数据集升级**：从 CIFAKE/Artifact 迁移至 NTIRE 2026 Robust AIGC Detection
2. **算法升级**：采用 CLIP ViT 作为主干网络，增强全局语义理解
3. **鲁棒性增强**：针对社交媒体、平台处理等真实场景优化
4. **多尺度推理**：支持多尺度输入和 TTA
5. **校准优化**：集成温度缩放校准

### V2.0 主要变更
- 将 AI Image Detector 从基础的检测系统升级为功能完整的 AI 图像取证平台
- 新增了多种可视化工具和用户体验改进
- 保持了高精度的检测能力

系统现在更加直观、强大且易于使用，为 AI 生成内容的检测提供了全面的解决方案。
