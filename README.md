

# AI Image Detector

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=0b0f19)
![Release](https://img.shields.io/badge/Release-V5.1-2ea44f)

> Detect AI-generated images using semantic + frequency features for real-world AI safety and content authenticity.

---

## 🚀 Overview

**AI Image Detector** is an open-source tool designed to identify AI-generated images in real-world scenarios.

It provides a **complete end-to-end pipeline**, including:

- 🧠 Model training & evaluation
- ⚡ FastAPI inference service
- 🌐 React-based web interface
- 📊 Explainable detection outputs
- 📦 Reproducible research + deployment pipeline

This project is built for:

- AI-generated content detection
- Deepfake / diffusion image analysis
- Content moderation systems
- Research and benchmarking

---

## ✨ Key Features

- **Hybrid Detection Architecture**  
  Combines **semantic (CLIP-based)** and **frequency-domain** features for robust detection.

- **Production-Ready API**  
  FastAPI backend with structured outputs (probability, threshold, explanations).

- **Web UI for Visualization**  
  React frontend for interactive analysis and debugging.

- **Explainability Support**  
  Includes branch contributions and intermediate signals for inspection.

- **Full Pipeline Support**  
  Covers training → evaluation → inference → deployment.

---

## 🧠 Method (High-Level)

The system uses a **dual-branch architecture**:

- **Semantic Branch** → captures global structure consistency (CLIP ViT)
- **Frequency Branch** → detects compression / texture anomalies
- **Fusion Module** → combines both signals into final prediction

Noise branch is retained as an optional diagnostic component.

---

## 📊 Example Results

| Mode | Precision | Recall | F1 |
|------|---------:|------:|----:|
| Recall-first | 0.82 | 1.00 | 0.90 |
| Balanced     | 1.00 | 1.00 | 1.00 |

> Note: Results are based on internal evaluation (`photos_test`). Cross-dataset validation is recommended.

---

## 🖼️ Demo

| Main Interface | Analysis Panel |
| --- | --- |
| ![UI](docs/assets/main_interface.png) | ![Analysis](docs/assets/analysis_panel.png) |

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
cd frontend && npm install
````

### 2. Prepare model checkpoint

```text
checkpoints/best.pth
```

### 3. Start backend

```bash
python scripts/start_backend.py
```

### 4. Start frontend

```bash
cd frontend
npm run dev
```

Access:

* Frontend → [http://localhost:5173](http://localhost:5173)
* Backend → [http://localhost:8000](http://localhost:8000)

---

## 🔌 API Usage

**POST /detect**

Input:

* image file

Output:

* prediction
* probability
* threshold
* branch contributions
* debug information

---

## 🧪 CLI Inference

```bash
python scripts/infer_ntire.py \
  --image photos_test/aigc7.png \
  --checkpoint checkpoints/best.pth
```

---

## 🏗️ Project Structure

```text
AI-Image-Detector/
├── frontend/          # React UI
├── services/api/      # FastAPI backend
├── src/               # Core model logic
├── scripts/           # Training & inference
├── configs/           # Experiment configs
├── docs/              # Documentation & model card
```

---

## 📌 Use Cases

* AI-generated image detection
* Deepfake identification
* Content authenticity verification
* AI safety research

---

## ⚠️ Limitations

* Model weights are not included
* Performance may vary across datasets
* Threshold tuning is required for new domains

---

## 📄 Documentation

* Model Card → `docs/MODEL_CARD.md`
* Deployment → `docs/DEPLOYMENT.md`
* Project Guide → `docs/PROJECT_GUIDE.md`

---

## 🤝 Contributing

Contributions are welcome!

* Bug reports
* Feature requests
* Model improvements
* Evaluation benchmarks

---

## 📜 License

MIT License (or your license)

---

## 🌱 Maintainer Notes

This project is actively maintained and evolving toward a practical tool for AI-generated image detection in real-world systems.

If you're working on AI safety, detection, or content authenticity, feel free to collaborate.

---

```
