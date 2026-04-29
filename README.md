# AI Image Detector / 人工智能生成图像检测工具

## 项目简介

AI Image Detector 是一个面向 AIGC 图像检测的前后端分离系统，覆盖模型训练、后端推理服务、前端可视化展示、用户登录注册以及管理员用户与日志管理。系统当前用于本科毕业设计/答辩场景，重点展示图像真实性判定、三分支取证证据和阈值判定逻辑。

## 核心模型结构

当前模型采用三分支结构：

- 语义结构分支：分析 RGB 图像中的内容结构、语义布局和场景一致性。
- 频域分布分支：分析频谱分布、频率异常和生成伪影线索。
- 噪声残差分支：分析 SRM-like 残差和噪声统计证据。

当前部署模式为 `deploy_safe_tri_branch`。三个分支都会真实 forward；在当前旧 checkpoint 下，最终判定使用已训练稳定的 `semantic + frequency` 主路径，即 `stable_sf_logit`。噪声残差分支当前作为辅助证据展示，不参与最终判定，因此 `decision_weights.noise = 0.0`。完整三分支门控判定能力已经保留，重新训练或迁移包含 `tri_fusion.*` 和 `tri_classifier.*` 权重的 checkpoint 后可启用。

## 当前推理逻辑

```text
输入图像
  → RGB 预处理
  → semantic / frequency / noise 三分支特征提取
  → stable_sf_logit 稳定部署决策
  → sigmoid 得到 AIGC probability
  → threshold 阈值判定
  → AIGC / REAL
```

旧 checkpoint 不会使用随机初始化的 `tri_classifier` 作为最终判定路径。如果 checkpoint 缺少 `tri_fusion.*` 或 `tri_classifier.*` 权重，后端会自动 fallback 到 `deploy_safe_tri_branch`。

## 阈值说明

系统按阈值判定，不按 50% 多数规则判定：

- `recall_first = 0.20`
- `balanced = 0.35`
- `precision_first = 0.55`

判定规则：

```text
if AIGC probability >= threshold:
    prediction = AIGC
else:
    prediction = REAL
```

示例：当 `AIGC probability = 41%` 且当前阈值为 `35%` 时，因为 `41% >= 35%`，系统判定为 `AIGC`。

## 前端功能

- 图片上传与检测
- 最终检测结果展示
- AIGC 概率与当前阈值解释
- 三分支 evidence score 与 decision weight 展示
- Fusion Evidence Triangle 三角证据分布图
- 噪声残差图与频谱证据图展示
- 用户登录 / 注册
- 管理员用户管理与操作日志查看

## 后端功能

- FastAPI 检测接口 `/detect`
- 模型推理封装 `DetectorInterface`
- 阈值配置与推理结果统一返回
- 可解释性结果返回：噪声残差、频谱图、分支分数、判定权重、证据权重
- MySQL 用户认证、注册、登录、登出
- 管理员用户列表、密码重置、按用户查看操作日志

## 安装环境

建议环境：

- Python 3.10+
- Node.js 18+
- MySQL 8+
- Git LFS（用于拉取 `checkpoints/best.pth`）

## 后端启动

```bash
pip install -r requirements.txt
uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
```

默认模型路径：

```text
checkpoints/best.pth
```

默认数据库配置位于 `services/api/auth.py`，可通过环境变量覆盖：

```text
DB_HOST
DB_USER
DB_PASSWORD
DB_NAME
JWT_SECRET
```

## 前端启动

```bash
cd frontend
npm install
npm run dev
```

前端默认开发 API 地址为：

```text
http://localhost:8000
```

也可以通过 `VITE_API_URL` 指定后端地址。

## 模型权重说明

默认部署权重路径为：

```text
checkpoints/best.pth
```

本仓库使用 Git LFS 管理 `.pth` 权重文件。首次克隆后请确认已安装 Git LFS：

```bash
git lfs install
git lfs pull
```

如果未拉取到权重文件，请手动将最终 checkpoint 放置到 `checkpoints/best.pth`。

## 回归测试结果

当前发布版本在 `photos_test` 20 张图片上使用 `deploy_safe_tri_branch` 模式回归测试，按文件名前缀标签统计：

```text
20 / 20 correct
accuracy = 1.0
```

结果文件：

```text
outputs/photos_test_deploy_safe_tri_branch_results.csv
```

## 项目目录结构

```text
src/          模型、训练、推理核心代码
services/     FastAPI 后端与认证数据库接口
frontend/     React + Vite 前端
configs/      训练与推理配置
scripts/      训练、挖掘、评估脚本
checkpoints/  部署 checkpoint
outputs/      最终回归测试结果
figures/      论文/答辩图表
```

## 注意事项

- 当前发布版本仍是三分支结构，三个分支都会真实 forward。
- 当前旧 checkpoint 下最终判定使用 `deploy_safe_tri_branch` 安全部署模式。
- 噪声残差分支当前作为辅助证据展示，不参与最终判定。
- 完整三分支同级门控最终判定需要重新训练或迁移包含 `tri_fusion.* / tri_classifier.*` 的 checkpoint 后启用。
- 不要将 `41%` 这类概率理解为必须超过 50% 才能判定 AIGC；系统使用当前阈值进行判定。
