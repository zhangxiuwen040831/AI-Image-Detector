# Thesis Materials

这里集中存放毕业设计 / 技术报告相关的公开材料。

## 文件说明

- `report.md`
  - 中文 Markdown 正文
  - 适合继续修改与二次排版
- `report.pdf`
  - 当前导出的答辩 / 归档版 PDF
- `analysis_notebook.ipynb`
  - 图表复现实验 Notebook
- `metrics_tables.xlsx`
  - 指标表格工作簿
- `training_curves.png`
  - 训练曲线图
- `training_metrics.csv`
  - 训练指标 CSV

## 说明

- 论文报告中的部分阈值分析用于实验叙事与误差分析；其中 `0.55` 是 `photos_test` 小规模诊断阈值扫描点，不是当前默认后端部署阈值。
- 如果你主要关心在线部署，请以 `configs/infer/default.yml` 与 `src/ai_image_detector/inference/detector.py` 中的默认配置为准。
