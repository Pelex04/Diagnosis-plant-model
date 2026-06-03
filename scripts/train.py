#!/usr/bin/env python3
"""
scripts/train.py
────────────────
CLI entry-point for training a PlantDx model.

Examples
────────
# Quickstart with defaults (data/ directory, 30 epochs)
    python scripts/train.py --data_dir data/

# Custom configuration
    python scripts/train.py \
        --data_dir   /datasets/PlantVillage \
        --output_dir checkpoints/run_01     \
        --epochs     50                      \
        --batch_size 64                      \
        --lr         1e-4                    \
        --workers    8

# Resume / fine-tune from a checkpoint
    python scripts/train.py \
        --data_dir  data/    \
        --resume    checkpoints/best_model.pth
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402, I001

from plantdx import Trainer, build_dataloaders, build_model, get_train_transform, get_val_transform  # noqa: E402, I001

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plantdx.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a PlantDx EfficientNet-B4 disease classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",   type=Path, required=True,
                   help="Root data directory (one sub-folder per class).")
    p.add_argument("--output_dir", type=Path, default=Path("checkpoints"),
                   help="Directory for checkpoints and training plots.")
    p.add_argument("--epochs",     type=int,   default=30,
                   help="Maximum training epochs.")
    p.add_argument("--batch_size", type=int,   default=32,
                   help="Samples per mini-batch.")
    p.add_argument("--lr",         type=float, default=3e-4,
                   help="Initial learning rate (AdamW).")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="AdamW weight decay.")
    p.add_argument("--patience",   type=int,   default=7,
                   help="Early-stopping patience (epochs).")
    p.add_argument("--workers",    type=int,   default=4,
                   help="DataLoader worker processes.")
    p.add_argument("--val_size",   type=float, default=0.15,
                   help="Fraction of data for validation.")
    p.add_argument("--test_size",  type=float, default=0.15,
                   help="Fraction of data for test.")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--resume",     type=Path,  default=None,
                   help="Path to a checkpoint to resume training from.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        logger.error("data_dir '%s' does not exist.", args.data_dir)
        sys.exit(1)

    # ── Build data loaders ────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir       = args.data_dir,
        train_transform= get_train_transform(),
        val_transform  = get_val_transform(),
        batch_size     = args.batch_size,
        val_size       = args.val_size,
        test_size      = args.test_size,
        num_workers    = args.workers,
        random_state   = args.seed,
    )

    logger.info("Classes (%d): %s", len(class_names), class_names)

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(num_classes=len(class_names))

    if args.resume:
        import torch
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Resumed from checkpoint: %s", args.resume)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        class_names  = class_names,
        output_dir   = args.output_dir,
        learning_rate= args.lr,
        weight_decay = args.weight_decay,
        num_epochs   = args.epochs,
        patience     = args.patience,
    )
    trainer.fit()

    # ── Final evaluation on held-out test set ─────────────────────────────────
    logger.info("─── Evaluating on held-out test set ──────────────────────")
    trainer.evaluate_test(test_loader)


if __name__ == "__main__":
    main()
