# AI Image Detector 更新公告

## 🚀 新功能与改进

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

## 📊 性能指标

- **准确率**: 87.31% (CIFAKE + Artifact 混合数据集)
- **AUC**: 0.9496
- **训练速度**: ~930 img/s (RTX 5090)
- **推理速度**: CPU ~0.5s/张，GPU ~0.1s/张

## 🚦 部署说明

### 后端启动
```bash
python start_backend.py
# 服务运行在 http://localhost:8000
```

### 前端启动
```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:5173
```

## 🔧 环境要求

- **前端**: Node.js 16+, npm 7+
- **后端**: Python 3.8+, CUDA 11.8+ (推荐)
- **硬件**: 至少 8GB 内存，GPU 推荐

## 🎯 核心特性

- **多模态检测** - 融合 RGB、噪声、频域特征
- **可视化分析** - 热力图、频谱图、噪声残差图
- **用户友好** - 直观的界面和流畅的交互
- **可扩展** - 支持不同模型和配置
- **高性能** - 优化的推理速度

## 🔄 变更总结

本次更新将 AI Image Detector 从基础的检测系统升级为功能完整的 AI 图像取证平台，新增了多种可视化工具和用户体验改进，同时保持了高精度的检测能力。系统现在更加直观、强大且易于使用，为 AI 生成内容的检测提供了全面的解决方案。