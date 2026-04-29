# Project Guide

## 1. 项目定位

本仓库是一个围绕 **AI 生成图像检测** 的完整工程实现，包含：

- 模型训练与评估
- 推理接口与 Web 前端
- 论文 / 答辩材料整理
- 报告图表生成脚本

当前 GitHub 公开版本为 `v5`，但其说明主线已经对齐到本地最终的 **V10** 模型路线。GitHub 公开历史不足以还原内部 V5-V10 版本演化，历代模型关系应优先依据本地 checkpoint、报告产物、reference JSON 与评估脚本判断。

## 2. 最终交付模型说明

- 默认模型路径：`checkpoints/best.pth`
- 当前默认推理模式：`base_only`
- 默认阈值档位：`balanced`
- 当前部署阈值：`recall-first=0.20`、`balanced=0.35`、`precision-first=0.35`
- 报告分析阈值：`0.55` 仅用于 `photos_test` 小规模诊断阈值扫描
- 主要公开说明依据：
  - `photos_test`
  - `scripts/evaluate_v10.py`
  - `scripts/generate_report_artifacts.py`
  - `docs/Thesis/`

## 3. 为什么公开仓库不包含大文件

为了让仓库更适合 GitHub 维护和克隆，本次公共版本默认**不提交**以下本地产物：

- NTIRE 训练数据集
- 历史中间 checkpoint
- 重复模型副本
- 本地缓存与临时输出
- 旧版归档目录

这类文件会显著放大仓库体积，也不利于公共协作与版本管理。

## 4. 模型与数据的获取方式

### 模型权重

请将最终权重放到：

```text
checkpoints/best.pth
```

仓库已经保留 [checkpoints/README.md](../checkpoints/README.md) 作为说明占位文件。

### 数据集

训练数据集不再保留在公共仓库根目录。使用者应自行准备 NTIRE 训练数据根目录，并在训练 / 评估命令中通过 `--data-root` 传入。

示例：

```bash
python scripts/train_v10.py --data-root /path/to/NTIRE-RobustAIGenDetection-train --save-dir outputs/v10_experiment
```

## 5. 文档合并说明

本次整理把原先零散的说明文件收敛为更清晰的入口：

- `README.md`
  - 面向 GitHub 首页访客
  - 项目亮点、快速开始、截图、核心命令
- `CHANGELOG.md`
  - 面向版本发布与 release 说明
- `docs/PROJECT_GUIDE.md`
  - 面向项目维护者和复现者
  - 统一承接原先的最终说明、结构说明和公开仓库策略
- `docs/Thesis/README.md`
  - 面向论文、毕设和技术报告查看者
- `docs/architecture.md`
  - 面向结构理解与展示

## 6. 当前建议阅读顺序

如果你是第一次访问本仓库，建议按以下顺序阅读：

1. `README.md`
2. `docs/PROJECT_GUIDE.md`
3. `docs/architecture.md`
4. `docs/Thesis/README.md`
5. `CHANGELOG.md`

## 7. 版本映射说明

| GitHub 公共版本 | 本地模型演进线 | 说明 |
| --- | --- | --- |
| `v5` | `V10` | 当前公开版对齐本地最终路线 |
| `v3.1` | 旧公开文档整理阶段 | 以 README 与架构整理为主 |
| `v3.0` | NTIRE 主线切换阶段 | 以数据集和训练管线升级为主 |

## 8. 安全与发布说明

旧版云训练草稿中曾包含不适合公开保留的远端连接信息。本次公共整理已将这类文档移出公开说明结构，不再作为仓库对外内容的一部分。

后续建议：

- 不要把服务器地址、账号、密码、临时 token 写入 Markdown 文档
- 不要提交本地 IDE 配置、数据库连接快照和个人缓存目录
- 不要把大模型权重和原始训练数据直接纳入 Git 历史
