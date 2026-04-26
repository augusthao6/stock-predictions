"""
Data preprocessing module: cleans stock price data, handles missing values,
removes outliers, and normalizes features.

Data quality challenges addressed:
  1. Missing values  - forward-fill then backward-fill for trading halts / holidays
  2. Outliers        - IQR-based clipping on daily returns to remove data errors
  3. Class imbalance - tracked and handled via class weights during training

AI-generated with Claude Code; reviewed and adapted by student.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Cleans and normalizes raw OHLCV data from StockDataCollector.
    Tracks imputation and outlier statistics for documentation.
    """

    def __init__(self, iqr_multiplier: float = 3.0, scaler_type: str = "robust"):
        self.iqr_multiplier = iqr_multiplier
        self.scaler_type = scaler_type
        self.scaler: Optional[RobustScaler] = None
        self.stats: Dict = {}

    # ------------------------------------------------------------------
    # Challenge 1: Missing Value Handling
    # ------------------------------------------------------------------

    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Forward-fills then backward-fills missing OHLCV values.
        Stock data has structured missingness (weekends, holidays) rather than random,
        so fill-forward is the domain-appropriate strategy.
        """
        n_before = df.isnull().sum().sum()
        df = df.ffill().bfill()
        n_after = df.isnull().sum().sum()

        stats = {
            "missing_before": int(n_before),
            "missing_after": int(n_after),
            "imputed": int(n_before - n_after),
            "imputation_rate": float((n_before - n_after) / max(df.size, 1)),
        }
        logger.info(f"Missing value imputation: {stats['imputed']} cells filled ({stats['imputation_rate']:.2%})")
        return df, stats

    # ------------------------------------------------------------------
    # Challenge 2: Outlier Handling
    # ------------------------------------------------------------------

    def clip_return_outliers(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Clips daily return outliers using IQR method.
        Extreme single-day moves (>IQR * multiplier) are likely data errors or
        stock splits not fully adjusted, so we clip rather than delete.
        """
        q1 = returns.quantile(0.25)
        q3 = returns.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr

        n_before = ((returns < lower) | (returns > upper)).sum().sum()
        returns_clipped = returns.clip(lower=lower, upper=upper, axis=1)
        n_after = ((returns_clipped < lower) | (returns_clipped > upper)).sum().sum()

        stats = {
            "outliers_clipped": int(n_before - n_after),
            "iqr_multiplier": self.iqr_multiplier,
            "clip_bounds_sample": {
                col: {"lower": float(lower[col]), "upper": float(upper[col])}
                for col in list(returns.columns)[:3]
            },
        }
        logger.info(f"Outlier clipping: {stats['outliers_clipped']} return values clipped")
        return returns_clipped, stats

    # ------------------------------------------------------------------
    # Challenge 3: Class Imbalance Tracking
    # ------------------------------------------------------------------

    def compute_class_weights(self, labels: np.ndarray) -> np.ndarray:
        """
        Computes inverse-frequency class weights for imbalanced up/down labels.
        Financial returns are roughly symmetric but can drift depending on bull/bear market period.
        """
        classes, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = total / (len(classes) * counts)
        weight_map = dict(zip(classes.tolist(), weights.tolist()))
        self.stats["class_weights"] = weight_map
        self.stats["class_distribution"] = {
            int(c): int(n) for c, n in zip(classes, counts)
        }
        logger.info(f"Class distribution: {self.stats['class_distribution']}")
        logger.info(f"Class weights: {weight_map}")
        return weights

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def fit_scaler(self, X: np.ndarray) -> None:
        """Fits a scaler on training data. Must be called before transform."""
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        self.scaler.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Call fit_scaler before transform")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit_scaler(X)
        return self.scaler.transform(X)

    # ------------------------------------------------------------------
    # Full preprocessing pipeline
    # ------------------------------------------------------------------

    def preprocess_prices(
        self, prices_df: pd.DataFrame, ticker: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Runs the full preprocessing pipeline for a single ticker:
          1. Extract OHLCV
          2. Handle missing values
          3. Compute daily returns
          4. Clip return outliers
          5. Return cleaned DataFrame with metadata
        """
        ohlcv = prices_df[ticker].copy()
        ohlcv, mv_stats = self.handle_missing_values(ohlcv)

        returns = ohlcv["Close"].pct_change()
        returns = returns.to_frame(name="Return")
        returns, out_stats = self.clip_return_outliers(returns)

        ohlcv["Return"] = returns["Return"]
        ohlcv["LogReturn"] = np.log1p(ohlcv["Return"])
        ohlcv = ohlcv.dropna()

        stats = {**mv_stats, **out_stats, "n_rows_final": len(ohlcv)}
        self.stats.update(stats)
        return ohlcv, stats

    def train_val_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronological train/val/test split to prevent data leakage.
        Ratios: 70% train / 15% val / 15% test.
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info(
            f"Split sizes — Train: {len(train)} ({train_ratio:.0%}), "
            f"Val: {len(val)} ({val_ratio:.0%}), "
            f"Test: {len(test)} ({1 - train_ratio - val_ratio:.0%})"
        )
        return train, val, test
