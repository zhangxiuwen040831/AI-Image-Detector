# Checkpoints

公开仓库默认不包含大体积模型权重。

当前公开版本：`V5.1`

默认加载路径：

```text
checkpoints/best.pth
```

默认推理模式：`base_only`

推荐阈值：

- `recall-first = 0.20`
- `balanced = 0.35`
- `precision-first = 0.35`

如果你要运行默认推理流程，请将最终权重手动放到上述位置。仓库中的默认脚本与配置会优先读取这个路径。

更多说明见：

- `docs/MODEL_CARD.md`
- `docs/DEPLOYMENT.md`
- `configs/infer/default.yml`
