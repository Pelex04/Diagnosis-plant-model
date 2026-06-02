"""
plantdx/model.py
────────────────
Core model definition and inference engine for PlantDx.

Architecture : EfficientNet-B4 (torchvision, ImageNet-pretrained)
               Fine-tuned classifier head for N plant disease classes.

Key design decisions
────────────────────
- Uses torchvision.models instead of the deprecated efficientnet_pytorch package.
- Class names are persisted alongside model weights so inference is fully
  self-contained (no need to re-scan the dataset directory).
- predict() returns top-k results with calibrated softmax probabilities.
- All public methods are type-annotated and fully documented.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B4_Weights

logger = logging.getLogger(__name__)

# ─── ImageNet normalisation constants ─────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Transform pipelines ──────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """Augmented pipeline used during training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(380, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    """Deterministic pipeline used for validation and inference."""
    return transforms.Compose([
        transforms.Resize(440),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(num_classes: int, dropout_rate: float = 0.4) -> nn.Module:
    """
    Build an EfficientNet-B4 model with a custom classification head.

    The backbone is initialised with ImageNet weights; only the classifier
    head is modified so fine-tuning converges faster on small datasets.

    Args:
        num_classes:   Number of target disease categories.
        dropout_rate:  Dropout probability before the final linear layer.

    Returns:
        A ready-to-train nn.Module.
    """
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    logger.info(
        "Built EfficientNet-B4 | classes=%d | backbone params=%s | head params=%s",
        num_classes,
        sum(p.numel() for p in model.features.parameters()),
        sum(p.numel() for p in model.classifier.parameters()),
    )
    return model


# ─── Inference engine ─────────────────────────────────────────────────────────

class PlantDiseaseClassifier:
    """
    High-level wrapper for loading a trained PlantDx model and running
    inference on single images or batches.

    Usage
    -----
    >>> clf = PlantDiseaseClassifier.from_checkpoint("checkpoints/best.pth")
    >>> result = clf.predict("leaf.jpg", top_k=3)
    >>> print(result)
    [{'class': 'Tomato_Leaf_Blight', 'confidence': 0.942},
     {'class': 'Tomato_Healthy',     'confidence': 0.041},
     {'class': 'Tomato_Mosaic_Virus','confidence': 0.017}]
    """

    def __init__(self, model: nn.Module, class_names: list[str], device: torch.device) -> None:
        self.model      = model.to(device)
        self.class_names = class_names
        self.device     = device
        self._transform = get_val_transform()

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> PlantDiseaseClassifier:
        """
        Load a classifier from a .pth checkpoint saved by the Trainer.

        The checkpoint must contain:
            - ``model_state_dict``  : model weights
            - ``class_names``       : ordered list of class label strings
            - ``num_classes``       : int
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        num_classes  = checkpoint["num_classes"]
        class_names  = checkpoint["class_names"]

        model = build_model(num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logger.info(
            "Loaded checkpoint '%s' | classes=%d | device=%s",
            checkpoint_path.name, num_classes, device,
        )
        return cls(model, class_names, device)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        image_source: str | Path | Image.Image,
        top_k: int = 5,
        confidence_threshold: float = 0.0,
    ) -> list[dict[str, float | str]]:
        """
        Run inference on a single image.

        Args:
            image_source:         File path or a pre-loaded PIL Image.
            top_k:                Return the top-k most likely classes.
            confidence_threshold: Suppress results below this probability.

        Returns:
            A list of dicts sorted by confidence (descending), e.g.:
            [{'class': 'Tomato_Leaf_Blight', 'confidence': 0.942}, ...]
        """
        if isinstance(image_source, str | Path):
            image = Image.open(image_source).convert("RGB")
        else:
            image = image_source.convert("RGB")

        tensor = self._transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0)

        top_k = min(top_k, len(self.class_names))
        top_probs, top_indices = torch.topk(probs, k=top_k)

        results = []
        for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
            if prob >= confidence_threshold:
                results.append({
                    "class":      self.class_names[idx],
                    "confidence": round(prob, 6),
                })

        return results

    def predict_batch(
        self,
        image_paths: list[str | Path],
        top_k: int = 1,
    ) -> list[list[dict[str, float | str]]]:
        """
        Run inference on a list of image paths.

        Returns a list of prediction results (one per image).
        """
        results = []
        for path in image_paths:
            results.append(self.predict(path, top_k=top_k))
        return results
