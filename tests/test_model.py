"""
tests/test_model.py
───────────────────
Unit tests for the PlantDx model, dataset, and inference engine.

Run with:
    pytest tests/ -v --tb=short
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from PIL import Image

# Ensure the src package is importable without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plantdx.model import (
    PlantDiseaseClassifier,
    build_model,
    get_train_transform,
    get_val_transform,
)
from plantdx.dataset import PlantDiseaseDataset


# ─── Fixtures ─────────────────────────────────────────────────────────────────

NUM_CLASSES = 5
BATCH_SIZE  = 4
IMG_SIZE    = 380


@pytest.fixture(scope="module")
def model() -> nn.Module:
    """A fresh EfficientNet-B4 model with NUM_CLASSES output classes."""
    return build_model(num_classes=NUM_CLASSES)


@pytest.fixture(scope="module")
def dummy_image() -> Image.Image:
    """A random RGB PIL image at the expected input resolution."""
    import numpy as np
    arr = (torch.rand(IMG_SIZE, IMG_SIZE, 3).numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


@pytest.fixture(scope="module")
def class_names() -> list:
    return [f"Class_{i}" for i in range(NUM_CLASSES)]


# ─── Model construction ───────────────────────────────────────────────────────

class TestBuildModel:
    def test_output_shape(self, model):
        """Model forward pass should produce (B, NUM_CLASSES) logits."""
        dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (2, NUM_CLASSES)

    def test_correct_num_classes(self):
        for n in [2, 10, 38]:
            m = build_model(num_classes=n)
            x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            m.eval()
            with torch.no_grad():
                out = m(x)
            assert out.shape[1] == n, f"Expected {n} outputs, got {out.shape[1]}"

    def test_model_is_nn_module(self, model):
        assert isinstance(model, nn.Module)

    def test_parameters_exist(self, model):
        params = list(model.parameters())
        assert len(params) > 0

    def test_custom_dropout(self):
        m = build_model(num_classes=3, dropout_rate=0.5)
        assert m is not None


# ─── Transforms ───────────────────────────────────────────────────────────────

class TestTransforms:
    def test_train_transform_output_shape(self, dummy_image):
        t = get_train_transform()
        tensor = t(dummy_image)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_val_transform_output_shape(self, dummy_image):
        t = get_val_transform()
        tensor = t(dummy_image)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_val_transform_is_deterministic(self, dummy_image):
        t = get_val_transform()
        t1 = t(dummy_image)
        t2 = t(dummy_image)
        assert torch.allclose(t1, t2)

    def test_train_transform_normalised(self, dummy_image):
        """Normalised tensors should have values outside [0, 1]."""
        t = get_train_transform()
        tensor = t(dummy_image)
        # After ImageNet normalisation, values will extend below 0 and above 1
        assert tensor.min().item() < 0 or tensor.max().item() > 1


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TestPlantDiseaseDataset:
    def _make_dataset(self, dummy_image, class_names, n=10):
        paths  = [dummy_image] * n   # reuse PIL image as source
        labels = list(range(n % len(class_names))) + [0] * (n - n % len(class_names))
        labels = labels[:n]
        # Patch open to return the dummy image directly
        ds = PlantDiseaseDataset.__new__(PlantDiseaseDataset)
        ds.image_paths = [MagicMock()] * n
        ds.labels      = labels
        ds.class_names = class_names
        ds.transform   = get_val_transform()
        return ds, dummy_image

    def test_len(self, dummy_image, class_names):
        ds, _ = self._make_dataset(dummy_image, class_names, n=8)
        assert len(ds) == 8

    def test_num_classes(self, dummy_image, class_names):
        ds, _ = self._make_dataset(dummy_image, class_names)
        assert ds.num_classes == NUM_CLASSES

    def test_class_distribution_keys(self, dummy_image, class_names):
        ds, _ = self._make_dataset(dummy_image, class_names)
        dist = ds.class_distribution()
        assert set(dist.keys()) == set(class_names)


# ─── Inference engine ─────────────────────────────────────────────────────────

class TestPlantDiseaseClassifier:
    @pytest.fixture
    def clf(self, model, class_names):
        device = torch.device("cpu")
        model.eval()
        return PlantDiseaseClassifier(model, class_names, device)

    def test_predict_returns_list(self, clf, dummy_image):
        results = clf.predict(dummy_image, top_k=3)
        assert isinstance(results, list)

    def test_predict_top_k_length(self, clf, dummy_image):
        for k in [1, 3, 5]:
            results = clf.predict(dummy_image, top_k=k)
            assert len(results) == k

    def test_predict_confidence_sum_approx_one(self, clf, dummy_image):
        results = clf.predict(dummy_image, top_k=NUM_CLASSES)
        total = sum(r["confidence"] for r in results)
        assert abs(total - 1.0) < 1e-4

    def test_predict_confidence_sorted_descending(self, clf, dummy_image):
        results = clf.predict(dummy_image, top_k=NUM_CLASSES)
        confs = [r["confidence"] for r in results]
        assert confs == sorted(confs, reverse=True)

    def test_predict_threshold_filters(self, clf, dummy_image):
        results = clf.predict(dummy_image, top_k=NUM_CLASSES, confidence_threshold=0.99)
        # With threshold=0.99 most classes should be filtered out
        assert all(r["confidence"] >= 0.99 for r in results)

    def test_predict_class_names_valid(self, clf, dummy_image, class_names):
        results = clf.predict(dummy_image, top_k=NUM_CLASSES)
        for r in results:
            assert r["class"] in class_names

    def test_predict_batch(self, clf, dummy_image):
        # Use MagicMock paths and patch PIL.Image.open
        with patch("plantdx.model.Image.open", return_value=dummy_image):
            paths   = [Path("fake1.jpg"), Path("fake2.jpg"), Path("fake3.jpg")]
            results = clf.predict_batch(paths, top_k=2)
        assert len(results) == 3
        assert all(len(r) == 2 for r in results)


# ─── Checkpoint round-trip ────────────────────────────────────────────────────

class TestCheckpointRoundTrip:
    def test_save_and_load(self, tmp_path, model, class_names):
        """Save a checkpoint and reload it via from_checkpoint."""
        ckpt_path = tmp_path / "test_model.pth"
        torch.save(
            {
                "epoch":            1,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  {},
                "val_loss":         0.5,
                "val_acc":          80.0,
                "class_names":      class_names,
                "num_classes":      NUM_CLASSES,
            },
            ckpt_path,
        )
        clf = PlantDiseaseClassifier.from_checkpoint(ckpt_path)
        assert clf.class_names == class_names
        assert len(clf.class_names) == NUM_CLASSES

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PlantDiseaseClassifier.from_checkpoint(tmp_path / "nonexistent.pth")
