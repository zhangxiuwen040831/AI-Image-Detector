# AI Image Detector v1.0.0-final-defense

## 1. Release Summary

This release finalizes the AI Image Detector system for graduation-defense demonstration. It includes the current model inference pipeline, FastAPI backend, React frontend, MySQL-backed authentication, administrator user/log management, and final regression results.

## 2. Key Features

- AIGC vs REAL image detection.
- Three-branch forensic analysis display.
- Threshold-based decision explanation.
- Noise residual and frequency spectrum evidence views.
- Fusion Evidence Triangle visualization.
- User login, registration, logout, profile update, and admin management.
- MySQL-backed user and operation logs.

## 3. Model Architecture

The model keeps a three-branch structure:

- Semantic structure branch.
- Frequency distribution branch.
- Noise residual branch.

All three branches are executed during inference. The frontend displays branch evidence scores, decision weights, and evidence distribution.

## 4. Deployment Mode

The current deployment mode is `deploy_safe_tri_branch`.

For the current old checkpoint, final prediction uses the trained stable semantic + frequency decision path (`stable_sf_logit`). The noise branch is still forwarded and returned as auxiliary forensic evidence, but it does not participate in final decision weighting.

If a future checkpoint includes trained `tri_fusion.*` and `tri_classifier.*` weights, the preserved full three-branch gated decision path can be enabled after validation.

## 5. Frontend Improvements

- Shows current threshold and decision rule clearly.
- Explains cases such as `AIGC probability 41% >= threshold 35%`, which are correctly classified as AIGC.
- Displays branch evidence score and final decision weight separately.
- Fixes `N/A` display for valid zero weights.
- Fixes Fusion Evidence Triangle point placement by using real evidence weights.
- Keeps Chinese as the default UI language.

## 6. Backend Improvements

- `/detect` returns probability, threshold, decision rule text, logits, branch scores, decision weights, evidence weights, and inference mode.
- Old checkpoints missing `tri_fusion / tri_classifier` weights automatically fallback to safe deployment inference.
- Backend normalization now preserves `0.0` weights and supports dict/list numeric payloads.
- Authentication endpoints return JSON consistently.
- Admin logs are scoped by selected user.

## 7. Validation Results

Validation was run on `photos_test` with 20 images.

```text
mode = deploy_safe_tri_branch
correct = 20 / 20
accuracy = 1.0
```

Detailed CSV:

```text
outputs/photos_test_deploy_safe_tri_branch_results.csv
```

## 8. Known Limitations

- Under the current old checkpoint, the noise branch does not participate in the final decision; it is returned as auxiliary evidence only.
- Full peer-level three-branch gated final decision requires retraining or migrating a checkpoint that contains trained `tri_fusion.*` and `tri_classifier.*` weights.
- The included regression result uses filename-prefix labels in `photos_test`.

## 9. Upgrade Notes

- Install Git LFS before cloning or pulling model weights:

```bash
git lfs install
git lfs pull
```

- Ensure the deployment checkpoint exists at:

```text
checkpoints/best.pth
```

- Start backend:

```bash
uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Start frontend:

```bash
cd frontend
npm install
npm run dev
```
