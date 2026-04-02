# FINAL_UPDATE

## V5.1 版本定位

本次 GitHub 公开发布统一命名为 **V5.1**。虽然本地最终训练路线已经演进到 **V10**，但公开仓库不按内部实验编号逐版暴露，因此继续采用公共版本号来描述对外交付状态。

V5.1 相比 `v5.0.0` 的重点不是重新设计工程结构，而是进一步把公开仓库收敛为一个更清晰、可运行、可阅读的交付版本。

## 本次核心变化

- 将公开版说明统一升级到 `V5.1` 口径。
- 重写 README，补充当前目录结构、前后端启动方式、推理接口说明、阈值建议和已知限制。
- 新增 `DEPLOYMENT.md`、`MODEL_CARD.md`、`docs/releases/v5.1.md`。
- 更新 `PROJECT_GUIDE.md` 与 `CHANGELOG.md`，让版本映射、清理策略和兼容性说明更明确。
- 从默认分支移除 `checkpoints/best.pth` 大权重，改为保留放置说明。
- 清理旧版 `v9` 训练/评估脚本，以及一次性的 smoke/cloud/remote 辅助脚本。

## 当前默认部署结论

- 默认模型入口：`checkpoints/best.pth`
- 默认推理模式：`base_only`
- 推荐阈值：`recall-first=0.20`、`balanced=0.35`、`precision-first=0.35`
- 默认推理骨干：`vit_base_patch16_clip_224.openai`

## 与旧版仓库的差异

- 旧版仓库仍以 `v5` 口径为主；V5.1 将对外表述统一到了当前可交付工程状态。
- 旧版 README 更偏项目展示；V5.1 README 更强调启动方式、模式说明、阈值和公共仓库边界。
- 新版删除了不适合长期公开维护的大权重与内部辅助脚本，但保留历史 release，不影响旧版本获取。

## 兼容性说明

- 历史 release 和历史 tag 保留，不做删除。
- `services/api/main.py`、`scripts/start_backend.py`、`scripts/infer_ntire.py`、`scripts/train_v10.py` 等主入口保持不变。
- 如需继续使用最终模型，请自行放置 `checkpoints/best.pth`。

## 建议阅读入口

1. `README.md`
2. `docs/DEPLOYMENT.md`
3. `docs/MODEL_CARD.md`
4. `docs/releases/v5.1.md`
5. `CHANGELOG.md`
