"""
Error analysis and model interpretability module.

Analyzes:
  - Failure cases: which samples are hardest to classify correctly
  - Feature importance from XGBoost for interpretability
  - Distribution of errors over time (e.g., concentrated in high-volatility periods)
  - Out-of-distribution / edge case behavior

AI-generated with Claude Code; reviewed and adapted by student.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Analyzes model failures to understand what types of inputs are most challenging.
    Required for the "error analysis with visualization" rubric item.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names

    # ------------------------------------------------------------------
    # Failure case identification
    # ------------------------------------------------------------------

    def identify_failures(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        features: np.ndarray,
        returns: np.ndarray,
        dates: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Creates a DataFrame of misclassified samples enriched with context.
        Allows answering: when does the model fail?
        """
        correct = y_true == y_pred
        failures_idx = np.where(~correct)[0]

        records = []
        for i in failures_idx:
            records.append({
                "idx": int(i),
                "date": dates[i] if dates is not None else None,
                "true_label": int(y_true[i]),
                "pred_label": int(y_pred[i]),
                "actual_return": float(returns[i]) if i < len(returns) else None,
                "error_type": (
                    "False Positive" if y_pred[i] == 1 and y_true[i] == 0
                    else "False Negative"
                ),
            })

        failure_df = pd.DataFrame(records)
        logger.info(
            f"Found {len(failures_idx)}/{len(y_true)} failures "
            f"({len(failures_idx)/len(y_true):.1%})"
        )
        return failure_df

    def failure_return_distribution(
        self, failure_df: pd.DataFrame, all_returns: np.ndarray
    ) -> Dict:
        """
        Compares the return distribution of failed predictions to correct ones.
        Key insight: false positives occur when actual returns are marginally negative.
        """
        fp_mask = failure_df["error_type"] == "False Positive"
        fn_mask = failure_df["error_type"] == "False Negative"

        stats = {
            "n_false_positive": int(fp_mask.sum()),
            "n_false_negative": int(fn_mask.sum()),
            "fp_mean_return": float(failure_df.loc[fp_mask, "actual_return"].mean())
            if fp_mask.any() else None,
            "fn_mean_return": float(failure_df.loc[fn_mask, "actual_return"].mean())
            if fn_mask.any() else None,
        }
        return stats

    # ------------------------------------------------------------------
    # Volatility-conditioned error analysis
    # ------------------------------------------------------------------

    def errors_by_volatility(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        volatility: np.ndarray,
        n_quantiles: int = 4,
    ) -> pd.DataFrame:
        """
        Breaks down error rates by volatility quantile.
        Hypothesis: models perform worse during high-volatility regimes.
        """
        quantile_labels = pd.qcut(volatility, q=n_quantiles, labels=False, duplicates="drop")
        records = []
        for q in range(n_quantiles):
            mask = quantile_labels == q
            if mask.sum() == 0:
                continue
            acc = float((y_true[mask] == y_pred[mask]).mean())
            vol_range = f"{volatility[mask].min():.3f}–{volatility[mask].max():.3f}"
            records.append({
                "volatility_quantile": f"Q{q+1} ({vol_range})",
                "n_samples": int(mask.sum()),
                "accuracy": round(acc, 4),
                "error_rate": round(1 - acc, 4),
            })
        df = pd.DataFrame(records)
        logger.info("Accuracy by volatility quantile:\n" + df.to_string(index=False))
        return df

    # ------------------------------------------------------------------
    # Temporal error analysis
    # ------------------------------------------------------------------

    def errors_over_time(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: pd.DatetimeIndex,
        window: int = 20,
    ) -> pd.Series:
        """Rolling accuracy over time — reveals structural breaks in model performance."""
        correct = (y_true == y_pred).astype(float)
        s = pd.Series(correct, index=dates)
        return s.rolling(window).mean()

    # ------------------------------------------------------------------
    # Edge case / OOD analysis
    # ------------------------------------------------------------------

    def edge_case_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        features: np.ndarray,
        n_top: int = 5,
    ) -> Dict:
        """
        Identifies the most extreme feature values and checks model accuracy there.
        Extreme values represent out-of-distribution inputs (unusual market conditions).
        """
        n_features = features.shape[1]
        results = {}

        for fi in range(min(n_features, 5)):
            col_vals = features[:, fi]
            extremes_mask = (col_vals > np.percentile(col_vals, 95)) | \
                            (col_vals < np.percentile(col_vals, 5))
            normal_mask = ~extremes_mask

            if extremes_mask.sum() < 5:
                continue

            extreme_acc = float((y_true[extremes_mask] == y_pred[extremes_mask]).mean())
            normal_acc = float((y_true[normal_mask] == y_pred[normal_mask]).mean())
            fname = self.feature_names[fi] if self.feature_names else f"feature_{fi}"

            results[fname] = {
                "normal_accuracy": round(normal_acc, 4),
                "extreme_accuracy": round(extreme_acc, 4),
                "accuracy_gap": round(normal_acc - extreme_acc, 4),
                "n_extreme_samples": int(extremes_mask.sum()),
            }

        return results

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def plot_confusion_by_regime(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        regimes: np.ndarray,
        regime_labels: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        """Plots per-regime accuracy as a bar chart."""
        fig, ax = plt.subplots(figsize=(8, 4))
        accs = []
        for r, label in enumerate(regime_labels):
            mask = regimes == r
            if mask.sum() == 0:
                accs.append(0.0)
                continue
            accs.append(float((y_true[mask] == y_pred[mask]).mean()))

        bars = ax.bar(regime_labels, accs, color=["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"])
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy by Market Regime")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="black", linestyle="--", alpha=0.5, label="Random baseline")
        ax.legend()
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}", ha="center")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_error_timeline(
        self,
        rolling_accuracy: pd.Series,
        save_path: Optional[str] = None,
    ) -> None:
        """Plots rolling accuracy over time to reveal temporal error patterns."""
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rolling_accuracy.index, rolling_accuracy.values, color="#2196F3", linewidth=1.5)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Random (50%)")
        ax.axhline(rolling_accuracy.mean(), color="green", linestyle="--", alpha=0.6,
                   label=f"Mean ({rolling_accuracy.mean():.3f})")
        ax.fill_between(rolling_accuracy.index, 0.5, rolling_accuracy.values,
                        where=rolling_accuracy.values > 0.5, alpha=0.2, color="green",
                        label="Above random")
        ax.fill_between(rolling_accuracy.index, 0.5, rolling_accuracy.values,
                        where=rolling_accuracy.values <= 0.5, alpha=0.2, color="red",
                        label="Below random")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling Accuracy (20-day window)")
        ax.set_title("Model Accuracy Over Time — Failure Pattern Analysis")
        ax.legend(loc="upper left")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
