#!/usr/bin/env python3
"""
scripts/gradcam.py
──────────────────
Generate Grad-CAM visualisations for PlantDx predictions.

Grad-CAM highlights the regions of a leaf image the model focused on
when making a prediction — essential for model explainability and
agronomist trust-building.

Implementation uses the pytorch-grad-cam library which supports
EfficientNet out of the box via its built-in layer auto-detection.

Examples
────────
# Single image — show top prediction with heatmap
    python scripts/gradcam.py \
        --checkpoint checkpoints/best_model.pth \
        --image      leaf.jpg

# Save output instead of displaying
    python scripts/gradcam.py \
        --checkpoint checkpoints/best_model.pth \
        --image      leaf.jpg \
        --output     reports/gradcam_leaf.png

# Batch — process every image in a directory
    python scripts/gradcam.py \
        --checkpoint checkpoints/best_model.pth \
        --image_dir  /path/to/images/ \
        --output_dir reports/gradcam/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plantdx import PlantDiseaseClassifier, get_val_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plantdx.gradcam")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _check_gradcam_installed() -> None:
    try:
        import pytorch_grad_cam  # noqa: F401
    except ImportError:
        logger.error(
            "pytorch-grad-cam is not installed.\n"
            "Install it with:  pip install grad-cam\n"
            "Or:               pip install plantdx[gradcam]"
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Grad-CAM explanations for PlantDx predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a trained .pth checkpoint.")
    p.add_argument("--image",      type=Path, default=None,
                   help="Single image to visualise.")
    p.add_argument("--image_dir",  type=Path, default=None,
                   help="Directory of images for batch Grad-CAM.")
    p.add_argument("--output",     type=Path, default=None,
                   help="Output path for single image (default: display only).")
    p.add_argument("--output_dir", type=Path, default=Path("reports/gradcam"),
                   help="Output directory for batch mode.")
    p.add_argument("--top_k",      type=int,  default=1,
                   help="Number of top predictions to annotate.")
    p.add_argument("--alpha",      type=float, default=0.5,
                   help="Heatmap overlay transparency (0=no overlay, 1=heatmap only).")
    return p.parse_args()


def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the final convolutional block of EfficientNet-B4.

    Grad-CAM needs the last spatial feature map before global average pooling.
    For EfficientNet this is the last MBConv block in features[-1].
    """
    return model.features[-1]


def generate_gradcam(
    clf: PlantDiseaseClassifier,
    image_path: Path,
    top_k: int = 1,
    alpha: float = 0.5,
) -> tuple[np.ndarray, list[dict]]:
    """
    Generate a Grad-CAM heatmap overlay for a single image.

    Returns:
        overlay  : H×W×3 uint8 numpy array — original image with heatmap overlaid
        results  : top-k prediction dicts from clf.predict()
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # ── Prepare image ─────────────────────────────────────────────────────────
    pil_image  = Image.open(image_path).convert("RGB")
    transform  = get_val_transform()
    input_tensor = transform(pil_image).unsqueeze(0).to(clf.device)

    # Normalised float image for overlay blending
    img_array = np.array(pil_image.resize((380, 380))).astype(np.float32) / 255.0

    # ── Run inference to get top prediction ───────────────────────────────────
    results = clf.predict(image_path, top_k=top_k)
    top_class_idx = clf.class_names.index(results[0]["class"])

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    target_layers = [_get_target_layer(clf.model)]
    targets       = [ClassifierOutputTarget(top_class_idx)]

    clf.model.eval()
    with GradCAM(model=clf.model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    overlay = show_cam_on_image(img_array, grayscale_cam, use_rgb=True, image_weight=1.0 - alpha)
    return overlay, results


def plot_gradcam(
    image_path: Path,
    overlay: np.ndarray,
    results: list[dict],
    output: Path | None = None,
) -> None:
    """Render side-by-side original + Grad-CAM overlay."""
    original = np.array(Image.open(image_path).convert("RGB").resize((380, 380)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"PlantDx Grad-CAM — {image_path.name}", fontsize=14, fontweight="bold")

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    label_lines = "\n".join(
        f"#{i+1}  {r['class']}  ({r['confidence']*100:.1f}%)"
        for i, r in enumerate(results)
    )
    axes[1].set_title(f"Grad-CAM\n{label_lines}", fontsize=11, loc="left")
    axes[1].axis("off")

    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        logger.info("Saved → %s", output)
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    _check_gradcam_installed()
    args = parse_args()

    if args.image is None and args.image_dir is None:
        logger.error("Provide --image or --image_dir.")
        sys.exit(1)

    clf = PlantDiseaseClassifier.from_checkpoint(args.checkpoint)

    if args.image:
        if not args.image.exists():
            logger.error("Image not found: %s", args.image)
            sys.exit(1)
        overlay, results = generate_gradcam(clf, args.image, args.top_k, args.alpha)
        plot_gradcam(args.image, overlay, results, output=args.output)

    if args.image_dir:
        if not args.image_dir.is_dir():
            logger.error("Not a directory: %s", args.image_dir)
            sys.exit(1)
        images = [
            p for p in sorted(args.image_dir.iterdir())
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        ]
        if not images:
            logger.warning("No images found in %s", args.image_dir)
            sys.exit(0)
        logger.info("Generating Grad-CAM for %d images…", len(images))
        for img_path in images:
            out = args.output_dir / f"{img_path.stem}_gradcam.png"
            try:
                overlay, results = generate_gradcam(clf, img_path, args.top_k, args.alpha)
                plot_gradcam(img_path, overlay, results, output=out)
            except Exception as exc:
                logger.warning("Skipped %s: %s", img_path.name, exc)


if __name__ == "__main__":
    main()
