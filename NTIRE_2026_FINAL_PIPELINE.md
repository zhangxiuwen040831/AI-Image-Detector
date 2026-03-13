# NTIRE 2026 Robust AIGC Detection Final Pipeline

## 1) Dataset assumptions and final data pipeline

- ArtiFact/CIFAKE are no longer aligned with NTIRE 2026 robustness objectives.
- NTIRE training shards are designed for in-the-wild robustness shifts and are better aligned to current challenge distributions.
- The implementation does not assume generator metadata and uses only available labels plus optional metadata columns if present.

Implemented files:
- `src/ai_image_detector/ntire/dataset.py`
- `src/ai_image_detector/ntire/augmentations.py`

Core behavior:
- Automatically scans `shard_*`.
- Supports selecting shard IDs.
- Reads `labels.csv` with robust column matching.
- Returns `(image_tensor, label_tensor, metadata)`.
- Metadata includes `shard_name`, `image_path`, `label`, and optional source/distortion-style fields when present.
- Includes sanity report utility.

Optional appendix: validation/test download commands (not required for train because local train path already exists):

```bash
huggingface-cli download deepfakesMSU/NTIRE-RobustAIGenDetection-val --repo-type dataset --local-dir NTIRE-RobustAIGenDetection-val
huggingface-cli download deepfakesMSU/NTIRE-RobustAIGenDetection-test --repo-type dataset --local-dir NTIRE-RobustAIGenDetection-test
```

## 2) Tiny subset creation for local CPU test

Implemented file:
- `scripts/make_tiny_subset.py`

What it does:
- Creates 256–500 image subset from local NTIRE train.
- Preserves real/fake balance as much as possible.
- Saves explicit CSV index (`tiny_subset_500.csv` by default).

Run:

```bash
python scripts/make_tiny_subset.py --subset-size 500 --out-csv tiny_subset_500.csv
```

## 3) Final upgraded model architecture

Implemented file:
- `src/ai_image_detector/ntire/model.py`

Architecture:
- Global semantic branch: timm backbone with CLIP ViT default (`vit_base_patch16_clip_224.openai`) and stable fallback.
- Frequency branch: FFT log-amplitude + lightweight CNN encoder.
- Noise/artifact branch: SRM-style residual filtering + lightweight CNN encoder.
- Fusion: gated softmax weighting across three branches.
- Heads: main classifier + optional lightweight auxiliary heads.
- Interpretability: fusion weights returned per sample.

Why this is stable for NTIRE-style robustness:
- Keeps strong global semantic priors.
- Explicitly models frequency and artifact cues.
- Uses lightweight branch heads and avoids fragile large-MoE complexity.

## 4) Real-world robust training strategy

Implemented files:
- `src/ai_image_detector/ntire/augmentations.py`
- `src/ai_image_detector/ntire/losses.py`
- `src/ai_image_detector/ntire/trainer.py`
- `scripts/train_ntire.py`

### 4.1 Augmentations

- JPEG compression: improves robustness to social-media style re-encoding.
- Resize/rescale: improves robustness to platform resizing.
- Gaussian blur: improves robustness to optical/post-process blur.
- Gaussian/ISO noise: improves robustness to sensor and pipeline noise.
- Mild color jitter: avoids overfitting to exact color statistics.
- Crop policy: random resized crop for train, center-style deterministic for eval.

### 4.2 Sampling

- Primary balancing: real/fake class balancing.
- Secondary balancing: distortion/source if available.
- Fallback: shard-balanced approximation.
- No fabricated generator metadata.

### 4.3 Validation

- Auto split strategy:
  1. source-held-out if source metadata exists
  2. shard-held-out if multiple shards available
  3. stratified random fallback
- Distortion/source grouped reporting when metadata exists.

### 4.4 Multi-scale inference

- Inference supports multiple scales (for example `224 336`) and optional horizontal-flip TTA.
- Logits are averaged across scales/TTA.

### 4.5 Loss

- Main: `BCEWithLogitsLoss`.
- Optional focal term.
- Optional lightweight auxiliary branch supervision.
- Calibration-ready output with temperature scaling.

## 5) Local CPU-only minimal test

Implemented file:
- `scripts/smoke_test_cpu.py`

Verifies in one quick CPU epoch:
- Dataset loading.
- Transform pipeline.
- Model forward.
- Loss computation.
- One-epoch optimization.
- Metrics reporting (AUROC/AUPRC/F1/Precision/Recall/ECE).

Run:

```bash
python scripts/smoke_test_cpu.py --freeze-backbone --batch-size 4 --num-workers 0 --image-size 160
```

## 6) Cloud full training pipeline

### 6.1 Preferred data access options

1. Direct cloud-side download/sync from dataset host.
2. Mounted storage (shared FS/object fuse).
3. Tar/rsync fallback only when direct sync is unavailable.

### 6.2 Full training script

Implemented file:
- `scripts/train_ntire.py`

Supports:
- Full or partial shards (`--shards 0,1,2`).
- Mixed precision on CUDA.
- Checkpointing (`latest`, `best`, rolling epochs).
- EMA.
- Resume.
- Validation and grouped reports.
- Single GPU or multi-GPU via DataParallel (`--data-parallel`).

### 6.3 Recommended hyperparameters

Local CPU tiny test:
- image size: 160
- batch size: 4
- epochs: 1
- lr: 1e-4
- weight decay: 1e-4
- warmup: 0
- grad clip: 1.0
- freeze backbone: true

Single GPU cloud:
- image size: 224
- batch size: 24
- epochs: 20
- lr: 3e-4
- weight decay: 1e-4
- warmup: 2 epochs
- grad clip: 1.0
- focal weight: 0.2
- EMA decay: 0.999

Multi-GPU cloud:
- image size: 224
- per-GPU batch: 24
- global lr: scale with global batch size
- epochs: 20–30
- warmup: 2–3 epochs
- grad clip: 1.0
- EMA decay: 0.999

## 7) Inference and calibration

Implemented file:
- `scripts/infer_ntire.py`

Supports:
- Single image inference.
- Folder inference.
- Multi-scale inference (`--scales 224 336`).
- Optional flip TTA.
- Calibrated probability via checkpoint temperature.
- Configurable threshold (`--threshold`).

Default threshold:
- 0.5 by default.
- Use validation-tuned threshold if application cost is asymmetric.

## 8) Evaluation protocol

Implemented file:
- `scripts/evaluate_ntire.py`

Protocol includes:
- AUROC, AUPRC, F1, Precision, Recall, ECE.
- Held-out validation split (source/shard/stratified fallback).
- Robustness stress conditions: clean + jpeg + blur + resize.
- Optional grouped reporting by distortion/source metadata if available.

Metric priority:
1. AUROC for ranking quality.
2. AUPRC for class-imbalance sensitivity.
3. F1/Precision/Recall for thresholded deployment behavior.
4. ECE for probability reliability.

## 9) Realistic expected gains and risks

Likely gains vs old ArtiFact/CIFAKE-centered setup:
- Better robustness to post-processing artifacts.
- Better cross-domain behavior through held-out shard/source strategy.
- More stable practical training due to moderate complexity.

Realistic target range:
- AUROC often around high-0.8 to low-0.9 on internal held-out NTIRE-style splits for stable baselines.
- Exact values depend strongly on compute budget, split hardness, and augmentation tuning.

Remaining risks:
- Unseen future generators can still reduce performance.
- Distortion overfitting remains possible with aggressive augmentation imbalance.
- Limited metadata constrains explicit domain balancing.

## 10) Final deliverable

### Final design summary

- Stable three-branch hybrid detector with gated fusion and optional auxiliary heads.
- Robust NTIRE-compatible dataset/sampling/validation pipeline.
- EMA, calibration, multi-scale inference, and robustness evaluation integrated.

### Full code delivered

- `src/ai_image_detector/ntire/__init__.py`
- `src/ai_image_detector/ntire/dataset.py`
- `src/ai_image_detector/ntire/augmentations.py`
- `src/ai_image_detector/ntire/model.py`
- `src/ai_image_detector/ntire/losses.py`
- `src/ai_image_detector/ntire/calibration.py`
- `src/ai_image_detector/ntire/metrics.py`
- `src/ai_image_detector/ntire/trainer.py`
- `scripts/make_tiny_subset.py`
- `scripts/smoke_test_cpu.py`
- `scripts/train_ntire.py`
- `scripts/infer_ntire.py`
- `scripts/evaluate_ntire.py`

### Suggested file structure

```text
ai-image-detector/
├─ NTIRE-RobustAIGenDetection-train/
├─ tiny_subset_500.csv
├─ checkpoints_ntire/
├─ scripts/
│  ├─ make_tiny_subset.py
│  ├─ smoke_test_cpu.py
│  ├─ train_ntire.py
│  ├─ infer_ntire.py
│  └─ evaluate_ntire.py
└─ src/ai_image_detector/ntire/
   ├─ __init__.py
   ├─ dataset.py
   ├─ augmentations.py
   ├─ model.py
   ├─ losses.py
   ├─ calibration.py
   ├─ metrics.py
   └─ trainer.py
```

### Local Windows CPU commands

```bash
python scripts/make_tiny_subset.py --subset-size 500 --out-csv tiny_subset_500.csv
python scripts/smoke_test_cpu.py --data-root "C:\Users\32902\Desktop\ai-image-detector\NTIRE-RobustAIGenDetection-train" --tiny-csv tiny_subset_500.csv --freeze-backbone --batch-size 4 --num-workers 0 --image-size 160
```

### Cloud GPU commands

```bash
python scripts/train_ntire.py --data-root /data/NTIRE-RobustAIGenDetection-train --save-dir /checkpoints/ntire_final --epochs 20 --batch-size 24 --num-workers 8 --image-size 224 --backbone-name vit_base_patch16_clip_224.openai --pretrained-backbone --use-balanced-sampler
python scripts/infer_ntire.py --checkpoint /checkpoints/ntire_final/best.pth --folder /data/eval_images --scales 224 336 --tta-flip --out-csv /checkpoints/ntire_final/infer.csv
python scripts/evaluate_ntire.py --data-root /data/NTIRE-RobustAIGenDetection-train --checkpoint /checkpoints/ntire_final/best.pth --image-size 224 --batch-size 32 --num-workers 8 --out-csv /checkpoints/ntire_final/eval_report.csv
```

### Troubleshooting

- If train folder has no `shard_*`, verify actual extraction path and pass it via `--data-root`.
- If CLIP backbone download is blocked, switch `--backbone-name resnet18` for smoke/debug.
- If CPU RAM is limited, lower `--image-size`, `--batch-size`, and `--num-workers`.
- If calibration appears unstable on tiny validation, increase validation size or disable threshold tuning until full run.
