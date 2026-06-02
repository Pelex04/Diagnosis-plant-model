#!/usr/bin/env python3
"""
scripts/evaluate.py
───────────────────
Evaluate a trained PlantDx model on a labelled dataset split.

Produces:
  - Per-class precision, recall, F1-score (classification report)
  - Overall accuracy, macro-F1, weighted-F1
  - Confusion matrix saved as a high-resolution PNG
  - JSON results file for downstream processing

Examples
────────
# Evaluate on the test split of a dataset directory
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pth \
        --data_dir   data/                       \
        --split      test                        \
        --output_dir reports/

# Evaluate on an arbitrary directory (treated as one flat class)
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pth \
        --data_dir   /path/to/labelled/images/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from plantdx import PlantDiseaseClassifier, build_dataloaders, get_val_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plantdx.evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a PlantDx checkpoint and generate a confusion matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a trained .pth checkpoint.")
    p.add_argument("--data_dir",   type=Path, required=True,
                   help="Root data directory (PlantVillage layout).")
    p.add_argument("--split",      choices=["train", "val", "test"], default="test",
                   help="Which data split to evaluate on.")
    p.add_argument("--output_dir", type=Path, default=Path("reports"),
                   help="Directory to write confusion matrix PNG and JSON report.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--val_size",   type=float, default=0.15)
    p.add_argument("--test_size",  type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--figsize",    type=int,   default=20,
                   help="Width/height of the confusion matrix figure in inches.")
    return p.parse_args()


@torch.no_grad()
def collect_predictions(
    clf: PlantDiseaseClassifier,
    loader: torch.utils.data.DataLoader,
) -> tuple[list[int], list[int]]:
    """Run the model over a DataLoader and return (all_labels, all_preds)."""
    clf.model.eval()
    all_labels, all_preds = [], []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(clf.device)
        logits = clf.model(images)
        preds  = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return all_labels, all_preds


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
    figsize: int = 20,
) -> None:
    """Save a normalised confusion matrix as a high-res PNG."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        linecolor="lightgrey",
        ax=ax,
        annot_kws={"size": max(6, 11 - n // 8)},
    )

    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=12)
    ax.set_ylabel("True Label",      fontsize=13, labelpad=12)
    ax.set_title("PlantDx — Normalised Confusion Matrix", fontsize=15, fontweight="bold", pad=16)
    ax.tick_params(axis="x", rotation=45, labelsize=max(6, 10 - n // 10))
    ax.tick_params(axis="y", rotation=0,  labelsize=max(6, 10 - n // 10))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    clf = PlantDiseaseClassifier.from_checkpoint(args.checkpoint)
    logger.info("Classes (%d): %s", len(clf.class_names), clf.class_names)

    # ── Build data loaders ────────────────────────────────────────────────────
    transform = get_val_transform()
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        data_dir       = args.data_dir,
        train_transform= transform,
        val_transform  = transform,
        batch_size     = args.batch_size,
        val_size       = args.val_size,
        test_size      = args.test_size,
        num_workers    = args.workers,
        random_state   = args.seed,
    )

    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader     = loader_map[args.split]
    logger.info("Evaluating on '%s' split (%d batches)…", args.split, len(loader))

    # ── Collect predictions ───────────────────────────────────────────────────
    y_true, y_pred = collect_predictions(clf, loader)

    # ── Classification report ─────────────────────────────────────────────────
    report_str = classification_report(
        y_true, y_pred,
        target_names=clf.class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true, y_pred,
        target_names=clf.class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    print("\n" + "─" * 70)
    print(f"  PlantDx Evaluation — split: {args.split}")
    print("─" * 70)
    print(report_str)

    accuracy    = report_dict["accuracy"]
    macro_f1    = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]

    print(f"  Accuracy      : {accuracy*100:.2f}%")
    print(f"  Macro F1      : {macro_f1*100:.2f}%")
    print(f"  Weighted F1   : {weighted_f1*100:.2f}%")
    print("─" * 70 + "\n")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_path = args.output_dir / f"confusion_matrix_{args.split}.png"
    save_confusion_matrix(y_true, y_pred, clf.class_names, cm_path, args.figsize)

    # ── Save JSON report ──────────────────────────────────────────────────────
    report_dict["_meta"] = {
        "checkpoint": str(args.checkpoint),
        "split":      args.split,
        "n_samples":  len(y_true),
        "num_classes": len(clf.class_names),
    }
    json_path = args.output_dir / f"eval_report_{args.split}.json"
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("JSON report saved → %s", json_path)


if __name__ == "__main__":
    main()
