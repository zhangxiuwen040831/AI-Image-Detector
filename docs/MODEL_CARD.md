# Model Card

## 1. 基本信息

- 公共发布版本：`V5.1`
- 本地训练路线：`V10`
- 任务：二分类 AIGC 图像检测（REAL vs AIGC）
- 默认权重路径：`checkpoints/best.pth`
- 默认推理模式：`base_only`
- 默认骨干：`vit_base_patch16_clip_224.openai`

## 2. 模型结构

当前公开版默认部署使用 `semantic + frequency` 主决策路径：

- semantic branch：负责高层语义与全局结构证据
- frequency branch：负责频域伪迹与压缩相关证据
- noise branch：保留为辅助诊断，不作为默认最终决策主路径

这也是当前 README 和后端服务默认说明中的 `base_only` 模式含义。

## 3. 推荐阈值

| 策略 | 阈值 | 适用场景 |
| --- | ---: | --- |
| `recall-first` | `0.20` | 更关注召回，允许少量误报 |
| `balanced` | `0.35` | 默认部署推荐 |
| `precision-first` | `0.35` | 更保守的公开默认值 |

阈值定义来源于：

- `configs/infer/default.yml`
- `src/ai_image_detector/inference/detector.py`
- `photos_test` 代表性结果

## 4. 公开仓库策略

公开仓库默认不附带最终模型权重文件。这样做是为了：

- 控制仓库体积
- 避免大文件拖慢克隆和浏览体验
- 让默认分支专注于代码、文档和轻量示例

如果你需要运行默认推理流程，请自行准备 `checkpoints/best.pth`。

## 5. 结果快照

来自 `photos_test` 的代表性表现：

| Threshold | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `0.20` | `0.8182` | `1.0000` | `0.9000` |
| `0.35` | `1.0000` | `1.0000` | `1.0000` |

## 6. 适用范围

- 项目展示与公开仓库演示
- NTIRE 路线相关研究复现
- FastAPI + React 的图像检测系统部署
- 毕设 / 论文材料归档与说明

## 7. 已知限制

- 当前公开版主要对齐 `base_only` 路径，不等价于公开全部内部实验变体。
- 最终模型仍然更偏向频域证据，语义分支仍有继续增强空间。
- 代表性指标主要来自轻量样例与论文材料，跨域泛化仍建议自行验证。
