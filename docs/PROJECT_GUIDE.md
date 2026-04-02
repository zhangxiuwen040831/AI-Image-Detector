# Project Guide

## 1. 项目定位

本仓库是一个围绕 **AI 生成图像检测** 的完整工程实现，包含：

- 模型训练与评估
- 推理接口与 Web 前端
- 论文 / 答辩材料整理
- 报告图表生成脚本

当前 GitHub 公开版本为 **V5.1**，说明主线对齐到本地最终的 **V10** 模型路线。

## 2. 当前公开版默认设置

- 当前发布版本：`V5.1`
- 默认模型路径：`checkpoints/best.pth`
- 默认推理模式：`base_only`
- 默认阈值档位：`balanced`
- 推荐阈值：`recall-first=0.20`、`balanced=0.35`、`precision-first=0.35`
- 默认骨干：`vit_base_patch16_clip_224.openai`

关键说明：公开仓库只保留模型入口和放置说明，默认不直接附带最终权重文件。

## 3. 为什么公开仓库不包含大文件

为了让仓库更适合 GitHub 维护、克隆与展示，公开版默认不提交以下本地产物：

- 最终模型大权重与历史中间 checkpoint
- NTIRE 原始训练数据集
- 训练输出目录、临时缓存和实验中间文件
- 一次性 smoke / cloud / remote 辅助脚本
- 不再维护的旧版 v9 实验脚本

这类文件会显著放大仓库体积，也会增加公开仓库的理解成本。

## 4. 模型与数据的获取方式

### 模型权重

请将最终权重放到：

```text
checkpoints/best.pth
```

更多信息见 [checkpoints/README.md](../checkpoints/README.md) 与 [MODEL_CARD.md](MODEL_CARD.md)。

### 数据集

训练数据集不保留在公开仓库根目录。使用者应自行准备 NTIRE 数据根目录，并在训练 / 评估命令中通过 `--data-root` 指定。

示例：

```bash
python scripts/train_v10.py --data-root /path/to/NTIRE-RobustAIGenDetection-train --save-dir outputs/v10_experiment --pretrained-backbone
```

## 5. 文档结构

- `README.md`
  - 面向 GitHub 首页访客
  - 项目简介、目录结构、启动方式、阈值与入口说明
- `CHANGELOG.md`
  - 面向版本发布与公开更新记录
- `docs/FINAL_UPDATE.md`
  - 面向本次 V5.1 发布总览
- `docs/DEPLOYMENT.md`
  - 面向部署与运行说明
- `docs/MODEL_CARD.md`
  - 面向模型信息、阈值、模式与限制说明
- `docs/releases/v5.1.md`
  - 面向 release notes
- `docs/Thesis/README.md`
  - 面向论文、毕设和技术报告查看者
- `docs/architecture.md`
  - 面向结构理解与展示

## 6. 当前建议阅读顺序

1. `README.md`
2. `docs/FINAL_UPDATE.md`
3. `docs/DEPLOYMENT.md`
4. `docs/MODEL_CARD.md`
5. `docs/PROJECT_GUIDE.md`
6. `docs/Thesis/README.md`
7. `CHANGELOG.md`

## 7. 版本映射说明

| GitHub 公共版本 | 本地模型演进线 | 说明 |
| --- | --- | --- |
| `V5.1` | `V10` | 当前公开版，对齐最终可交付工程状态 |
| `v5.0.0` | `V10` | 首次大规模公共整理发布 |
| `v3.1` | 旧公开文档整理阶段 | 以 README 与架构整理为主 |
| `v3.0` | NTIRE 主线切换阶段 | 以数据集和训练管线升级为主 |

## 8. 公开版保留与清理原则

### 保留

- 当前有效的前后端代码
- 当前有效的训练、评估、推理脚本
- 关键配置、示例图片和论文材料
- 历史 release 与历史 tag

### 清理

- 大权重与重复模型副本
- 临时输出与缓存目录
- 一次性 smoke / cloud / remote 辅助脚本
- 已被 V10 主线替代的旧版 v9 脚本

## 9. 安全与发布说明

- 不要把服务器地址、账号、密码、临时 token 写入 Markdown 文档或脚本。
- 不要把大模型权重和原始训练数据直接纳入 Git 历史。
- 如果需要继续扩展公开仓库，优先在 `docs/releases/` 和 `CHANGELOG.md` 中记录变化，保持版本线索清晰。
