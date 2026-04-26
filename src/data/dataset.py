"""
PyTorch Dataset and DataLoader utilities for sequence-based stock prediction.
Supports batching, shuffling, and variable sequence lengths.

AI-generated with Claude Code; reviewed and adapted by student.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class StockSequenceDataset(Dataset):
    """
    Sliding-window dataset for LSTM input.
    Each sample is a (seq_len, n_features) tensor with a binary label:
      - Label 1: next-day return > 0 (up)
      - Label 0: next-day return <= 0 (down)
    """

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        seq_len: int = 20,
    ):
        """
        Args:
            features: (T, n_features) array of normalized feature values
            returns:  (T,) array of daily log returns
            seq_len:  lookback window length in trading days
        """
        self.seq_len = seq_len
        self.X, self.y = self._build_sequences(features, returns)

    def _build_sequences(
        self, features: np.ndarray, returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.seq_len, len(features)):
            X.append(features[i - self.seq_len : i])
            # Predict direction of the NEXT day's return
            y.append(1 if returns[i] > 0 else 0)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

    @property
    def feature_dim(self) -> int:
        return self.X.shape[2] if len(self.X) > 0 else 0

    @property
    def class_distribution(self) -> dict:
        classes, counts = np.unique(self.y, return_counts=True)
        return {int(c): int(n) for c, n in zip(classes, counts)}


def create_dataloaders(
    train_dataset: StockSequenceDataset,
    val_dataset: StockSequenceDataset,
    test_dataset: StockSequenceDataset,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train/val/test DataLoaders.
    Train loader uses shuffling; val and test do not (preserves temporal order for analysis).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
