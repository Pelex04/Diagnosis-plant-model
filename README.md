# 🌿 PlantDx — Plant Disease Diagnosis

<p align="center">
  <a href="https://github.com/Pelex04/Diagnosis-plant-model/actions/workflows/ci.yml">
    <img src="https://github.com/Pelex04/Diagnosis-plant-model/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://github.com/Pelex04/Diagnosis-plant-model/releases/latest">
    <img src="https://img.shields.io/github/v/release/Pelex04/Diagnosis-plant-model?color=brightgreen&label=release" alt="Latest Release">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
  </a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch" alt="PyTorch 2.2+">
  <img src="https://img.shields.io/badge/model-EfficientNet--B4-orange" alt="EfficientNet-B4">
</p>

<p align="center">
  <strong>Production-grade deep learning system for automated crop disease detection from leaf images.</strong><br>
  Built on EfficientNet-B4 &nbsp;·&nbsp; Trained on PlantVillage &nbsp;·&nbsp; 38 disease classes &nbsp;·&nbsp; MIT Licensed
</p>

---

## Overview

PlantDx is a computer vision pipeline for detecting plant diseases from photographs of leaves. It uses a fine-tuned **EfficientNet-B4** backbone — pre-trained on ImageNet, then adapted for the PlantVillage dataset — to classify leaf images across 38 disease categories covering 14 crop species.

The system is designed for real-world use: a single checkpoint file is fully self-contained (no dataset required at inference time), the CLI handles both single-image and batch prediction, and the training pipeline includes proper train/val/test splits, mixed-precision support, and early stopping.

---

## Supported Classes

PlantDx covers the full 38-class PlantVillage taxonomy:

| Crop | Conditions Covered |
|---|---|
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mite, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Corn (Maize) | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| + 8 more | Strawberry, Peach, Cherry, Squash, Raspberry, Soybean, Orange, Blueberry |

> The model defaults to 38 classes. Retraining on a custom subset is straightforward — see [Training](#training).

---

## Architecture

```
Input Image (any size)
       │
       ▼
  Val/Test Transform                   Train Transform
  ┌─────────────────────┐              ┌───────────────────────────────────┐
  │ Resize → 440px      │              │ RandomResizedCrop(380, scale=0.7) │
  │ CenterCrop → 380px  │              │ RandomHorizontalFlip              │
  │ ToTensor            │              │ RandomVerticalFlip(p=0.2)         │
  │ ImageNet Normalize  │              │ RandomRotation(30°)               │
  └─────────────────────┘              │ ColorJitter                       │
                                       │ ToTensor + ImageNet Normalize     │
                                       └───────────────────────────────────┘
       │
       ▼
  EfficientNet-B4 Backbone (ImageNet pretrained)
  ┌──────────────────────────────────────────────┐
  │  MBConv blocks · Compound scaled             │
  │  ~17.5M parameters                           │
  │  Output: feature vector (1792-dim)           │
  └──────────────────────────────────────────────┘
       │
       ▼
  Custom Classifier Head
  ┌────────────────────┐
  │  Dropout(p=0.4)    │
  │  Linear(1792 → N)  │   N = number of disease classes
  └────────────────────┘
       │
       ▼
  Softmax → Top-K Predictions with Confidence Scores
```

**Training configuration:**
- Optimiser: AdamW (`lr=3e-4`, `weight_decay=1e-4`)
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss with label smoothing (`ε=0.1`)
- Regularisation: Dropout + gradient clipping (`max_norm=1.0`)
- Precision: Mixed-precision (FP16) when CUDA is available
- Early stopping: patience = 7 epochs

---

## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- pip
- CUDA-capable GPU recommended (CPU works but is slower for training)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Pelex04/Diagnosis-plant-model.git
cd Diagnosis-plant-model

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install PyTorch (select the right variant for your hardware)
#    GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install -r requirements.txt
```

### Download Pre-trained Weights

Model weights are distributed via [GitHub Releases](https://github.com/Pelex04/Diagnosis-plant-model/releases/latest) and are **not** committed to the repository.

```bash
mkdir -p checkpoints
wget https://github.com/Pelex04/Diagnosis-plant-model/releases/download/v2.0.0/best_model.pth \
     -O checkpoints/best_model.pth
```

---

## Inference

### Single image

```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pth \
    --image      path/to/leaf.jpg \
    --top_k      3
```

**Example output:**
```
  📷  tomato_leaf.jpg
     1. Tomato__Tomato_mosaic_virus            94.2%  ████████████████████████████
     2. Tomato_healthy                          4.1%  █
     3. Tomato__Late_blight                     1.7%
```

### Batch inference on a directory

```bash
python scripts/predict.py \
    --checkpoint  checkpoints/best_model.pth \
    --image_dir   /path/to/images/ \
    --top_k       1 \
    --output_json results.json
```

### Python API

```python
from plantdx import PlantDiseaseClassifier

# Load — fully self-contained, no dataset directory needed
clf = PlantDiseaseClassifier.from_checkpoint("checkpoints/best_model.pth")

# Single prediction
results = clf.predict("leaf.jpg", top_k=3)
for r in results:
    print(f"{r['class']:<45}  {r['confidence']*100:.1f}%")

# With confidence threshold
results = clf.predict("leaf.jpg", top_k=5, confidence_threshold=0.05)

# Batch
paths   = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = clf.predict_batch(paths, top_k=1)
```

---

## Training

### Dataset setup

PlantDx expects the **PlantVillage** directory structure:

```
data/
├── Tomato__Tomato_mosaic_virus/
│   ├── 0a1b2c3d.jpg
│   └── ...
├── Tomato_healthy/
│   └── ...
└── ...
```

**Download PlantVillage:**
```bash
# Via Kaggle CLI
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/
```

### Run training

```bash
# Default configuration (30 epochs, batch 32, lr 3e-4)
python scripts/train.py --data_dir data/

# Custom run
python scripts/train.py \
    --data_dir   data/              \
    --output_dir checkpoints/run_01 \
    --epochs     50                 \
    --batch_size 64                 \
    --lr         1e-4               \
    --workers    8                  \
    --patience   10
```

Training outputs to `--output_dir`:
- `best_model.pth` — best checkpoint by validation accuracy
- `training_curves.png` — loss and accuracy curves

### Resume from checkpoint

```bash
python scripts/train.py \
    --data_dir data/ \
    --resume   checkpoints/best_model.pth
```

---

## Project Structure

```
Diagnosis-plant-model/
│
├── src/plantdx/              # Core library
│   ├── __init__.py           # Public API surface
│   ├── model.py              # EfficientNet-B4 factory + PlantDiseaseClassifier
│   ├── dataset.py            # PlantDiseaseDataset + build_dataloaders()
│   └── trainer.py            # Trainer with early stopping + checkpointing
│
├── scripts/
│   ├── train.py              # Training CLI
│   └── predict.py            # Inference CLI
│
├── tests/
│   └── test_model.py         # 20+ unit tests (pytest)
│
├── configs/
│   └── default.json          # Default hyperparameters
│
├── archive/                  # Original v1.0.0 source (preserved for reference)
│
├── .github/
│   ├── workflows/ci.yml      # GitHub Actions CI
│   └── ISSUE_TEMPLATE/       # Bug report & feature request templates
│
├── data/                     # Dataset directory (not committed — download separately)
├── checkpoints/              # Model weights (not committed — see Releases)
│
├── requirements.txt
├── .gitignore
├── LICENSE
├── CHANGELOG.md
└── README.md
```

---

## Development

```bash
# Run the full test suite
pytest tests/ -v --tb=short

# Run with coverage report
pytest tests/ --cov=src/plantdx --cov-report=term-missing

# Lint
pip install ruff
ruff check src/ scripts/ tests/
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a full version history.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- **PlantVillage Dataset** — Hughes & Salathé (2015). [doi:10.1371/journal.pcbi.1004993](https://doi.org/10.1371/journal.pcbi.1004993)
- **EfficientNet** — Tan & Le (2019). [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- **PyTorch & torchvision** — Meta AI Research
