"""
LSTM training loop with early stopping, learning rate scheduling, and gradient clipping.

Regularization techniques applied:
  1. Dropout (inside LSTMPredictor architecture)
  2. L2 weight decay (weight_decay in Adam optimizer)
  3. Early stopping (stops when val loss stagnates for `patience` epochs)

AI-generated with Claude Code; reviewed and adapted by student.
"""

import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state: Optional[Dict] = None
        self.triggered = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                if self.restore_best and self.best_state:
                    model.load_state_dict(self.best_state)
                return True
        return False


class LSTMTrainer:
    """
    Manages the full training lifecycle: optimizer, scheduler, early stopping,
    and history tracking for training curve visualization.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        max_grad_norm: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.max_grad_norm = max_grad_norm

        # Adam with L2 weight decay (regularization technique #2)
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # ReduceLROnPlateau: halves LR when val loss plateaus for 5 epochs
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=False,
        )

        # Early stopping (regularization technique #3)
        self.early_stopping = EarlyStopping(patience=patience, restore_best=True)

        weight_tensor = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []
        }

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Runs one training epoch. Returns (mean_loss, accuracy)."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()

            # Gradient clipping prevents exploding gradients common in LSTMs
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluates model on a DataLoader. Returns (mean_loss, accuracy)."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

        return total_loss / total, correct / total

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping and LR scheduling.
        Returns training history for curve visualization.
        """
        logger.info(
            f"Training on {self.device} | "
            f"lr={self.optimizer.param_groups[0]['lr']}, "
            f"wd={self.optimizer.param_groups[0]['weight_decay']}"
        )

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            self.scheduler.step(val_loss)

            if verbose and epoch % 10 == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train_loss={tr_loss:.4f}, val_loss={val_loss:.4f} | "
                    f"train_acc={tr_acc:.4f}, val_acc={val_acc:.4f} | "
                    f"lr={current_lr:.2e} | {elapsed:.1f}s"
                )

            if self.early_stopping.step(val_loss, self.model):
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best val_loss={self.early_stopping.best_loss:.4f})"
                )
                break

        return self.history

    @torch.no_grad()
    def predict_proba(self, loader: DataLoader) -> np.ndarray:
        """Returns (N, 2) softmax probability array for a DataLoader."""
        self.model.eval()
        all_proba = []
        for X, _ in loader:
            X = X.to(self.device)
            proba = torch.softmax(self.model(X), dim=-1)
            all_proba.append(proba.cpu().numpy())
        return np.concatenate(all_proba, axis=0)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        """Returns (N,) predicted class array."""
        return self.predict_proba(loader).argmax(axis=1)
