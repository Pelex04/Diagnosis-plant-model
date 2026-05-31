"""
plantdx/trainer.py
──────────────────
Training engine for PlantDx.

Features
────────
- Mixed-precision training (torch.amp) when CUDA is available.
- Early stopping with configurable patience.
- Best-model checkpointing (saves only when val accuracy improves).
- Training curves saved to disk as a publication-quality PNG.
- Class names and metadata persisted in every checkpoint so inference
  is fully self-contained.
- Structured logging throughout — no bare print() calls.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Stop training when validation loss fails to improve for `patience` epochs.

    Args:
        patience:  Epochs to wait before stopping.
        min_delta: Minimum improvement to reset the counter.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            logger.info("EarlyStopping: %d/%d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Manages the full training lifecycle: fit → evaluate → checkpoint → plot.

    Args:
        model:        nn.Module to train.
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        class_names:  Ordered list of class label strings.
        output_dir:   Where to write checkpoints and plots.
        learning_rate: Initial LR for AdamW.
        weight_decay:  L2 regularisation coefficient.
        num_epochs:   Maximum training epochs.
        patience:     Early-stopping patience (epochs).
        label_smoothing: Label smoothing for CrossEntropyLoss (reduces overconfidence).
    """

    def __init__(
        self,
        model:         nn.Module,
        train_loader:  DataLoader,
        val_loader:    DataLoader,
        class_names:   List[str],
        output_dir:    str | Path = "checkpoints",
        learning_rate: float = 3e-4,
        weight_decay:  float = 1e-4,
        num_epochs:    int   = 30,
        patience:      int   = 7,
        label_smoothing: float = 0.1,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.class_names  = class_names
        self.output_dir   = Path(output_dir)
        self.num_epochs   = num_epochs
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self.model.to(self.device)

        self.criterion  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer  = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler  = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        self.scaler     = GradScaler(enabled=self.device.type == "cuda")
        self.early_stop = EarlyStopping(patience=patience)

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }
        self.best_val_acc = 0.0

        logger.info(
            "Trainer ready | device=%s | epochs=%d | LR=%.0e | classes=%d",
            self.device, num_epochs, learning_rate, len(class_names),
        )

    # ── Core loop ─────────────────────────────────────────────────────────────

    def fit(self) -> Dict[str, List[float]]:
        """Run the full training loop. Returns the training history dict."""
        logger.info("─── Training started ───────────────────────────────────")
        t0 = time.time()

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss,   val_acc   = self._eval_epoch(self.val_loader, desc="Val")

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            logger.info(
                "Epoch %2d/%d | "
                "train loss %.4f acc %.2f%% | "
                "val loss %.4f acc %.2f%%",
                epoch, self.num_epochs,
                train_loss, train_acc,
                val_loss,   val_acc,
            )

            # ── Checkpoint if best ────────────────────────────────────────────
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_loss, val_acc)

            # ── Early stopping ────────────────────────────────────────────────
            if self.early_stop.step(val_loss):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        elapsed = time.time() - t0
        logger.info(
            "─── Training complete | %.1fs | best val acc: %.2f%% ───",
            elapsed, self.best_val_acc,
        )
        self._save_training_curves()
        return self.history

    def evaluate_test(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate on the held-out test set.

        Returns:
            (test_loss, test_accuracy_percent)
        """
        loss, acc = self._eval_epoch(test_loader, desc="Test")
        logger.info("Test  | loss %.4f | acc %.2f%%", loss, acc)
        return loss, acc

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} train", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.device.type == "cuda"):
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader), 100.0 * correct / total

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, desc: str = "Eval") -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = total = 0

        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            with autocast(enabled=self.device.type == "cuda"):
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)
            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        return total_loss / len(loader), 100.0 * correct / total

    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float) -> None:
        path = self.output_dir / "best_model.pth"
        torch.save(
            {
                "epoch":            epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state":  self.optimizer.state_dict(),
                "val_loss":         val_loss,
                "val_acc":          val_acc,
                "class_names":      self.class_names,
                "num_classes":      len(self.class_names),
            },
            path,
        )
        logger.info("  ✓ Saved best checkpoint  val_acc=%.2f%%  → %s", val_acc, path)

    def _save_training_curves(self) -> None:
        """Save a clean loss + accuracy training curve PNG."""
        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("PlantDx Training Curves", fontsize=14, fontweight="bold")

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], label="Train", linewidth=2)
        axes[0].plot(epochs, self.history["val_loss"],   label="Val",   linewidth=2, linestyle="--")
        axes[0].set_title("Cross-Entropy Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, self.history["train_acc"], label="Train", linewidth=2)
        axes[1].plot(epochs, self.history["val_acc"],   label="Val",   linewidth=2, linestyle="--")
        axes[1].set_title("Accuracy (%)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        out = self.output_dir / "training_curves.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  ✓ Training curves saved → %s", out)
