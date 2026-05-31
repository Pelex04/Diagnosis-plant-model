"""
plantdx/dataset.py
──────────────────
Dataset utilities for PlantDx.

Supports the PlantVillage directory layout:
    data/
    ├── Tomato__Healthy/
    │   ├── image001.jpg
    │   └── ...
    ├── Tomato__Leaf_Blight/
    │   └── ...
    └── ...

Key design decisions
────────────────────
- Proper three-way split (train / val / test) with stratification.
- Class names are derived deterministically (sorted) so label indices are
  reproducible across runs without a saved mapping.
- DataLoader configuration exposes num_workers and pin_memory for GPU
  training performance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PlantDiseaseDataset(Dataset):
    """
    PyTorch Dataset for a directory-structured plant disease image collection.

    Args:
        image_paths: Absolute paths to image files.
        labels:      Integer class index for each image.
        class_names: Ordered list of class name strings.
        transform:   torchvision transform pipeline.
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        class_names: List[str],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        assert len(image_paths) == len(labels), "Mismatch between paths and labels."
        self.image_paths = image_paths
        self.labels      = labels
        self.class_names = class_names
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def class_distribution(self) -> Dict[str, int]:
        """Return a dict mapping class name → sample count."""
        dist: Dict[str, int] = {name: 0 for name in self.class_names}
        for label in self.labels:
            dist[self.class_names[label]] += 1
        return dist


# ─── Data loading helpers ─────────────────────────────────────────────────────

def _scan_directory(data_dir: Path) -> Tuple[List[Path], List[int], List[str]]:
    """
    Walk a PlantVillage-style directory and return (paths, labels, class_names).

    Classes are determined by top-level subdirectory names, sorted
    alphabetically for reproducibility.
    """
    class_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not class_dirs:
        raise ValueError(f"No class subdirectories found in '{data_dir}'.")

    class_names = [d.name for d in class_dirs]
    image_paths: List[Path] = []
    labels: List[int] = []

    for class_idx, class_dir in enumerate(class_dirs):
        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in _SUPPORTED_EXTENSIONS
        ]
        if not images:
            logger.warning("Class '%s' has no images — skipping.", class_dir.name)
            continue
        image_paths.extend(images)
        labels.extend([class_idx] * len(images))
        logger.debug("  %-35s  %4d images", class_dir.name, len(images))

    logger.info(
        "Scanned '%s': %d classes | %d total images",
        data_dir, len(class_names), len(image_paths),
    )
    return image_paths, labels, class_names


def build_dataloaders(
    data_dir: str | Path,
    train_transform: transforms.Compose,
    val_transform:   transforms.Compose,
    batch_size:    int   = 32,
    val_size:      float = 0.15,
    test_size:     float = 0.15,
    num_workers:   int   = 4,
    random_state:  int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build train / val / test DataLoaders from a PlantVillage-style directory.

    The split is stratified on class labels so every split has proportional
    class representation.

    Args:
        data_dir:        Root directory containing one sub-folder per class.
        train_transform: Augmented transform for the training set.
        val_transform:   Deterministic transform for val and test sets.
        batch_size:      Samples per batch.
        val_size:        Fraction of data for validation (default 15%).
        test_size:       Fraction of data for test (default 15%).
        num_workers:     DataLoader worker processes.
        random_state:    Random seed for reproducibility.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_dir = Path(data_dir)
    image_paths, labels, class_names = _scan_directory(data_dir)

    # ── 1. Carve out the test set ──────────────────────────────────────────────
    paths_tv, paths_test, labels_tv, labels_test = train_test_split(
        image_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    # ── 2. Split the remainder into train / val ────────────────────────────────
    # Adjust val_size relative to the train+val pool
    val_size_adjusted = val_size / (1.0 - test_size)
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_tv, labels_tv,
        test_size=val_size_adjusted,
        stratify=labels_tv,
        random_state=random_state,
    )

    logger.info(
        "Split → train: %d | val: %d | test: %d",
        len(paths_train), len(paths_val), len(paths_test),
    )

    # ── 3. Construct datasets ──────────────────────────────────────────────────
    train_ds = PlantDiseaseDataset(paths_train, labels_train, class_names, train_transform)
    val_ds   = PlantDiseaseDataset(paths_val,   labels_val,   class_names, val_transform)
    test_ds  = PlantDiseaseDataset(paths_test,  labels_test,  class_names, val_transform)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    return train_loader, val_loader, test_loader, class_names
