# Architecture

## System Architecture

```mermaid
graph TB
    User[User] --> Frontend[React Frontend]
    Frontend -->|HTTP Request| Backend[FastAPI Backend]
    Backend -->|Load Config| Configs[Configs]
    Backend -->|Load Model| Checkpoints[Checkpoints]
    Backend -->|Inference Request| Inference[Inference Engine]
    Inference -->|Forward Pass| Model[AI Model]
    Inference -->|Generate| GradCAM[GradCAM Explanation]
    Model -->|Prediction| Backend
    GradCAM -->|Heatmap| Backend
    Backend -->|Response| Frontend
    Frontend -->|Display| User
```

## Training Pipeline

```mermaid
graph TB
    Config[Configs/train] -->|Load Params| Trainer[Trainer]
    Dataset[Datasets] -->|Load| DataLoader[DataLoader]
    DataLoader -->|Batch| Transforms[Transforms]
    Transforms -->|Augmentation| Model[Multi-Branch Model]
    Model -->|Output| Loss[Loss Function]
    Loss -->|Backward| Optimizer[Optimizer]
    Optimizer -->|Update| Model
    Trainer -->|Train Epoch| DataLoader
    Trainer -->|Validate| DataLoader
    Trainer -->|Save| Checkpoint[Checkpoint]
    Checkpoint -->|Load| Model
```

## Inference Pipeline

```mermaid
graph TB
    Image[Input Image] -->|Load| Preprocess[Preprocessing]
    Preprocess -->|Resize/Normalize| Model[AI Model]
    Model -->|Feature Extraction| Detector[ForensicDetector]
    Detector -->|Branch Scores| Prediction[Prediction]
    Detector -->|Optional| GradCAM[GradCAM]
    GradCAM -->|Heatmap| Explanation[Explanation Report]
    Prediction -->|Output| Result[Detection Result]
    Explanation -->|Report| Result
```

## Code Architecture

```mermaid
graph TD
    datasets[datasets] -->|Base Dataset| data[data]
    data -->|DataLoader| training[training]
    datasets -->|Fusion Dataset| data
    training -->|Trainer| models[models]
    models -->|MultiBranchDetector| inference[inference]
    inference -->|ForensicDetector| explain[explain]
    explain -->|GradCAM| inference
    inference -->|Detector| evaluation[evaluation]
    training -->|Checkpoint| models
    ntire[ntire] -->|NTIRE Pipeline| models
    ntire -->|NTIRE Dataset| datasets
    utils[utils] -->|Config/Logger| datasets
    utils -->|Config/Logger| models
    utils -->|Config/Logger| training
    utils -->|Config/Logger| inference
    utils -->|Config/Logger| evaluation
    utils -->|Config/Logger| explain
    utils -->|Config/Logger| ntire
```
