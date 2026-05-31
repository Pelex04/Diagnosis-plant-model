"""
PlantDx — Plant Disease Diagnosis via Deep Learning
────────────────────────────────────────────────────
A production-grade EfficientNet-B4 classifier trained on the PlantVillage
dataset for automated detection of crop diseases from leaf images.

Modules
-------
plantdx.model    — Model factory and inference engine (PlantDiseaseClassifier)
plantdx.dataset  — Dataset class and DataLoader builder
plantdx.trainer  — Training engine with early stopping and checkpointing
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("plantdx")
except PackageNotFoundError:
    __version__ = "2.0.0"

from .model   import PlantDiseaseClassifier, build_model, get_train_transform, get_val_transform
from .dataset import PlantDiseaseDataset, build_dataloaders
from .trainer import Trainer

__all__ = [
    "PlantDiseaseClassifier",
    "build_model",
    "get_train_transform",
    "get_val_transform",
    "PlantDiseaseDataset",
    "build_dataloaders",
    "Trainer",
]
