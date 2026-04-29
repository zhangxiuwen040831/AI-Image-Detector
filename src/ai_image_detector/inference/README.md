# 推理引擎架构

本文档描述了 AI 图像检测器的推理引擎架构，基于职责分离和模块化设计原则。

## 架构概览

推理引擎采用分层架构，将复杂的推理过程分解为多个职责单一的组件：

```
┌─────────────────────┐
│ DetectorInterface   │  # 统一接口
├─────────────────────┤
│                     │
├── ConfigManager     │  # 配置管理
├── InferenceEngine   │  # 推理引擎
└── VisualizationGenerator  # 可视化生成
```

## 核心组件

### 1. ConfigManager

**职责**：统一管理所有配置的读取、验证和优先级。

**功能**：
- 从配置文件、环境变量读取配置
- 提供类型安全的配置访问接口
- 支持配置验证和标准化
- 管理阈值配置和分析阈值

**使用示例**：

```python
from ai_image_detector.inference.config import ConfigManager

config_manager = ConfigManager("default")
inference_config = config_manager.get_inference_config(checkpoint)
```

### 2. VisualizationGenerator

**职责**：生成各种可视化结果。

**实现**：
- `GradCAMGenerator`：生成 Grad-CAM 热力图
- `FusionTriangleGenerator`：生成融合证据三角图

**使用示例**：

```python
from ai_image_detector.inference.visualization import VisualizationFactory

fusion_generator = VisualizationFactory.create_generator("fusion_triangle")
fusion_image = fusion_generator.generate(
    semantic=0.33, freq=0.33, noise=0.33, prediction="REAL"
)
```

### 3. InferenceStrategy

**职责**：定义不同的推理策略。

**实现**：
- `BaseOnlyStrategy`：仅基础推理
- `HybridOptionalStrategy`：混合推理（包含噪声专家）
- `MultiScaleStrategy`：多尺度推理

**使用示例**：

```python
from ai_image_detector.inference.strategy import InferenceEngine

engine = InferenceEngine(model, "cuda", scales=[224, 256])
result = engine.predict(image, mode="base_only")
```

### 4. DetectorInterface

**职责**：统一的推理接口，整合所有子组件。

**功能**：
- 模型加载和管理
- 推理执行
- 结果处理
- 可视化生成

**使用示例**：

```python
from ai_image_detector.inference import DetectorInterface

detector = DetectorInterface("checkpoints/best.pth")
result = detector.predict("test.jpg")
```

## 目录结构

```
inference/
├── __init__.py            # 导出接口
├── detector_interface.py  # 统一接口
├── config/                # 配置管理
│   ├── __init__.py
│   └── config_manager.py
├── visualization/         # 可视化生成
│   ├── __init__.py
│   └── visualization.py
└── strategy/              # 推理策略
    ├── __init__.py
    └── inference_strategy.py
```

## 配置说明

配置文件位于 `configs/infer/default.yml`，支持以下配置项：

- `image_size`：输入图像大小
- `scales`：多尺度推理的尺度列表
- `tta_flip`：是否启用测试时增强
- `thresholds`：不同配置的阈值
- `analysis_thresholds`：分析阈值

## 推理模式

支持的推理模式：

- `base_only`：仅使用基础推理路径（语义 + 频域）
- `hybrid_optional`：使用混合推理路径（语义 + 频域 + 噪声）
- `multi_scale`：多尺度推理

## 迁移指南

### 从 ForensicDetector 迁移到 DetectorInterface

**旧代码**：

```python
from ai_image_detector.inference.detector import ForensicDetector

detector = ForensicDetector("checkpoints/best.pth")
result = detector.predict("test.jpg")
```

**新代码**：

```python
from ai_image_detector.inference import DetectorInterface

detector = DetectorInterface("checkpoints/best.pth")
result = detector.predict("test.jpg")
```

### API 兼容性

DetectorInterface 保持了与 ForensicDetector 相同的 API 接口，确保向后兼容：

- `predict()` 方法的参数和返回值格式保持不变
- 支持相同的推理模式和配置选项
- 保持相同的结果格式

## 性能优化

- **异步处理**：可视化生成可以异步执行，不阻塞推理请求
- **缓存策略**：支持模型和配置的缓存
- **多线程**：支持多线程推理

## 测试

### 单元测试

每个组件都可以独立进行单元测试：

- `ConfigManager`：测试配置解析和验证
- `VisualizationGenerator`：测试可视化生成
- `InferenceStrategy`：测试不同推理策略
- `DetectorInterface`：测试端到端推理

### 集成测试

- 运行 `scripts/run_threshold_test.py` 测试阈值性能
- 启动 API 服务测试完整流程

## 维护指南

### 新增推理模式

1. 创建新的 `InferenceStrategy` 子类
2. 在 `InferenceEngine._init_strategies()` 中注册新策略
3. 更新文档和测试

### 新增可视化类型

1. 创建新的 `VisualizationGenerator` 子类
2. 在 `VisualizationFactory.create_generator()` 中注册新生成器
3. 更新文档和测试

### 新增配置项

1. 在 `InferenceConfig` dataclass 中添加新字段
2. 更新 `ConfigManager` 中的配置加载逻辑
3. 更新文档和测试
