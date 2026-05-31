# Changelog

All notable changes to PlantDx are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [2.0.0] — 2026-05-31

Major rewrite. Breaking changes from v1.0.0.

### Added
- **EfficientNet-B4 backbone** replacing the deprecated `efficientnet_pytorch` EfficientNet-B0; higher accuracy, actively maintained via `torchvision`
- **Proper three-way data split** (train / val / test) with stratification — eliminates data leakage present in v1.0.0
- **`PlantDiseaseClassifier`** inference class: fully self-contained, loaded from checkpoint with no dataset dependency
- **Top-k predictions** with confidence scores and configurable threshold filtering
- **Mixed-precision training** (`torch.amp`) for faster GPU training
- **EarlyStopping** with configurable patience
- **CosineAnnealingLR** scheduler replacing ReduceLROnPlateau
- **Label smoothing** in CrossEntropyLoss to reduce overconfidence
- **Gradient clipping** (`max_norm=1.0`) for training stability
- **Checkpoints** now store `class_names` and `num_classes` — inference is fully portable
- **Training curves** automatically saved as `checkpoints/training_curves.png`
- **CLI scripts**: `scripts/train.py` and `scripts/predict.py` with `argparse`
- **Batch inference** via `PlantDiseaseClassifier.predict_batch()`
- **Full test suite** (`tests/test_model.py`) — 20+ unit tests covering model, transforms, dataset, inference, and checkpoint round-trip
- **CI pipeline** via GitHub Actions (Python 3.10 & 3.11, pytest + coverage, ruff lint)
- **`requirements.txt`** with pinned versions for reproducibility
- **`configs/default.json`** for experiment configuration management
- **GitHub issue templates** for bugs and feature requests
- **MIT License**
- **`.gitignore`** excluding model weights, virtual environments, pycache, IDE files

### Changed
- Backbone upgraded from EfficientNet-B0 to EfficientNet-B4 (higher input resolution: 380px)
- `train_transform` now includes vertical flip, stronger colour jitter, and random grayscale
- `DataLoader` now uses `num_workers=4` and `pin_memory=True` for GPU setups
- Repository restructured: source code moved to `src/plantdx/`, scripts to `scripts/`, tests to `tests/`
- Model weights removed from Git history; distributed via GitHub Releases

### Fixed
- `num_classes=2` hardcoded in `train.py` — now derived automatically from the data directory
- `class_names` not persisted at save time — prediction after fresh load no longer requires re-scanning dataset
- Test set used as validation during training (data leakage) — now uses a dedicated validation split
- `__pycache__` committed to repository

### Removed
- Dependency on `efficientnet_pytorch` (unmaintained since 2022)
- Committed `.pth` model weights (now in GitHub Releases)
- `__pycache__/` from repository

---

## [1.0.0] — 2025-01-26

Initial release.

### Added
- EfficientNet-B0 classifier via `efficientnet_pytorch`
- `PlantDiseaseDetector` class with `prepare_data`, `train`, `evaluate`, `predict`, `save_model`, `load_model`
- Basic training script (`train.py`) and inference script (`test.py`)
- PlantVillage-style dataset loader with 80/20 train/test split
- Pre-trained weights (`plant_disease_model.pth`) for 2-class tomato model
