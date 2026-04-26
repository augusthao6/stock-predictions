"""
Systematic hyperparameter search using grid search on validation data.
Tests at least 4 configurations (requirement: >= 3) with documented results.

AI-generated with Claude Code; reviewed and adapted by student.
"""

import logging
import copy
from itertools import product
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.lstm_model import LSTMPredictor
from .trainer import LSTMTrainer

logger = logging.getLogger(__name__)

# Grid of configurations to search
HYPERPARAMETER_GRID = {
    "hidden_size":   [64, 128, 256],
    "num_layers":    [1, 2],
    "dropout":       [0.2, 0.3],
    "learning_rate": [1e-3, 5e-4],
    "weight_decay":  [1e-4],
}

# Reduced grid for quick evaluation during development
SMALL_GRID = {
    "hidden_size":   [64, 128, 256],
    "num_layers":    [2],
    "dropout":       [0.2, 0.3],
    "learning_rate": [1e-3],
    "weight_decay":  [1e-4],
}


class HyperparameterSearch:
    """
    Grid search over LSTM hyperparameters evaluated on a held-out validation set.
    Keeps results from every configuration for the comparison table.
    """

    def __init__(
        self,
        input_size: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        grid: Dict[str, List] = None,
        epochs_per_trial: int = 30,
        patience: int = 10,
        device: torch.device = None,
        class_weights: torch.Tensor = None,
    ):
        self.input_size = input_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grid = grid or SMALL_GRID
        self.epochs_per_trial = epochs_per_trial
        self.patience = patience
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_weights = class_weights
        self.results: List[Dict] = []

    def _expand_grid(self) -> List[Dict]:
        keys = list(self.grid.keys())
        combos = list(product(*[self.grid[k] for k in keys]))
        return [dict(zip(keys, combo)) for combo in combos]

    def run(self) -> Tuple[Dict, List[Dict]]:
        """
        Trains one LSTM per configuration and returns:
          (best_config, all_results_sorted_by_val_acc)
        """
        configs = self._expand_grid()
        logger.info(f"Running hyperparameter search over {len(configs)} configurations")

        for i, cfg in enumerate(configs):
            logger.info(f"\n--- Config {i+1}/{len(configs)}: {cfg} ---")

            model = LSTMPredictor(
                input_size=self.input_size,
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
            )

            trainer = LSTMTrainer(
                model=model,
                learning_rate=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
                patience=self.patience,
                class_weights=self.class_weights,
                device=self.device,
            )

            history = trainer.fit(
                self.train_loader,
                self.val_loader,
                epochs=self.epochs_per_trial,
                verbose=False,
            )

            best_val_acc = max(history["val_acc"])
            best_val_loss = min(history["val_loss"])
            actual_epochs = len(history["train_loss"])

            result = {
                **cfg,
                "val_acc": round(best_val_acc, 4),
                "val_loss": round(best_val_loss, 4),
                "epochs_trained": actual_epochs,
                "early_stopped": actual_epochs < self.epochs_per_trial,
            }
            self.results.append(result)
            logger.info(
                f"  val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}, "
                f"epochs={actual_epochs}"
            )

        self.results.sort(key=lambda x: x["val_acc"], reverse=True)
        best = self.results[0]
        logger.info(f"\nBest config: {best}")
        return best, self.results

    def summary_table(self) -> str:
        """Returns a formatted comparison table for documentation."""
        if not self.results:
            return "No results yet. Run search first."
        header = (
            "| hidden | layers | dropout | lr     | val_acc | val_loss | epochs |"
        )
        sep = "|--------|--------|---------|--------|---------|----------|--------|"
        rows = [header, sep]
        for r in self.results:
            rows.append(
                f"| {r['hidden_size']:6d} | {r['num_layers']:6d} | "
                f"{r['dropout']:7.1f} | {r['learning_rate']:.4f} | "
                f"{r['val_acc']:7.4f} | {r['val_loss']:8.4f} | {r['epochs_trained']:6d} |"
            )
        return "\n".join(rows)
