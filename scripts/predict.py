#!/usr/bin/env python3
"""
scripts/predict.py
──────────────────
CLI entry-point for running inference with a trained PlantDx model.

Examples
────────
# Single image
    python scripts/predict.py \
        --checkpoint checkpoints/best_model.pth \
        --image      leaf.jpg

# Top-3 predictions with confidence threshold
    python scripts/predict.py \
        --checkpoint checkpoints/best_model.pth \
        --image      leaf.jpg \
        --top_k      3         \
        --threshold  0.05

# Batch inference on a directory of images
    python scripts/predict.py \
        --checkpoint checkpoints/best_model.pth \
        --image_dir  /path/to/images/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402, I001

from plantdx import PlantDiseaseClassifier  # noqa: E402, I001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plantdx.predict")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PlantDx inference on one image or a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a trained .pth checkpoint.")
    p.add_argument("--image",      type=Path, default=None,
                   help="Path to a single image file.")
    p.add_argument("--image_dir",  type=Path, default=None,
                   help="Directory of images for batch inference.")
    p.add_argument("--top_k",      type=int,   default=5,
                   help="Number of top predictions to return.")
    p.add_argument("--threshold",  type=float, default=0.0,
                   help="Suppress predictions below this confidence score.")
    p.add_argument("--output_json", type=Path, default=None,
                   help="Optionally write results to a JSON file.")
    return p.parse_args()


def format_results(image_name: str, results: list) -> str:
    lines = [f"\n  📷  {image_name}"]
    for rank, r in enumerate(results, 1):
        bar   = "█" * int(r["confidence"] * 30)
        lines.append(f"     {rank}. {r['class']:<40}  {r['confidence']*100:5.1f}%  {bar}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if args.image is None and args.image_dir is None:
        logger.error("Provide either --image or --image_dir.")
        sys.exit(1)

    clf = PlantDiseaseClassifier.from_checkpoint(args.checkpoint)

    all_results = {}

    if args.image:
        if not args.image.exists():
            logger.error("Image not found: %s", args.image)
            sys.exit(1)
        results = clf.predict(
            args.image,
            top_k=args.top_k,
            confidence_threshold=args.threshold,
        )
        print(format_results(args.image.name, results))
        all_results[str(args.image)] = results

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

        logger.info("Running batch inference on %d images…", len(images))
        for img_path in images:
            results = clf.predict(
                img_path,
                top_k=args.top_k,
                confidence_threshold=args.threshold,
            )
            print(format_results(img_path.name, results))
            all_results[str(img_path)] = results

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results written → %s", args.output_json)


if __name__ == "__main__":
    main()
