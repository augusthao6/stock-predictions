"""
Evaluation metrics combining ML classification metrics with financial performance metrics.
Uses at least three distinct and appropriate metric types.

AI-generated with Claude Code; reviewed and adapted by student.
"""

import numpy as np
import logging
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

logger = logging.getLogger(__name__)


class TradingMetrics:
    """
    Computes both ML classification metrics and financial performance metrics.

    ML metrics:  accuracy, precision, recall, F1, AUC-ROC
    Financial:   Sharpe ratio, Max drawdown, Annualized return, Calmar ratio
    Timing:      Inference time, throughput
    """

    # ------------------------------------------------------------------
    # ML Classification Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict:
        """Returns at least 5 classification metrics."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if y_proba is not None and len(np.unique(y_true)) > 1:
            try:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                metrics["auc_roc"] = float("nan")

        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        return metrics

    # ------------------------------------------------------------------
    # Financial Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def sharpe_ratio(daily_returns: np.ndarray, risk_free_rate: float = 0.05) -> float:
        """Annualized Sharpe ratio. Standard risk-adjusted return metric in finance."""
        excess = daily_returns - risk_free_rate / 252
        if excess.std() < 1e-10:
            return 0.0
        return float(np.sqrt(252) * excess.mean() / excess.std())

    @staticmethod
    def max_drawdown(cumulative_returns: np.ndarray) -> float:
        """Maximum peak-to-trough loss. Key risk metric for trading strategies."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1e-10)
        return float(drawdown.min())

    @staticmethod
    def annualized_return(daily_returns: np.ndarray) -> float:
        """Compound annualized growth rate from daily returns."""
        total_return = (1 + daily_returns).prod()
        n_years = len(daily_returns) / 252
        if n_years <= 0:
            return 0.0
        return float(total_return ** (1 / n_years) - 1)

    @staticmethod
    def calmar_ratio(daily_returns: np.ndarray) -> float:
        """Annualized return divided by maximum drawdown. Better than Sharpe in trending markets."""
        cum = (1 + daily_returns).cumprod()
        mdd = abs(TradingMetrics.max_drawdown(cum))
        ann_ret = TradingMetrics.annualized_return(daily_returns)
        if mdd < 1e-10:
            return 0.0
        return float(ann_ret / mdd)

    @staticmethod
    def win_rate(signals: np.ndarray, actual_returns: np.ndarray) -> float:
        """Fraction of trades that were profitable."""
        trade_mask = signals != 0
        if trade_mask.sum() == 0:
            return 0.0
        profitable = ((signals[trade_mask] == 1) & (actual_returns[trade_mask] > 0)) | \
                     ((signals[trade_mask] == -1) & (actual_returns[trade_mask] < 0))
        return float(profitable.sum() / trade_mask.sum())

    @classmethod
    def full_report(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        daily_returns: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "Model",
    ) -> Dict:
        """Returns a complete metrics report combining ML and financial metrics."""
        ml_metrics = cls.classification_metrics(y_true, y_pred, y_proba)
        cum_ret = (1 + daily_returns).cumprod()

        financial_metrics = {
            "sharpe_ratio": cls.sharpe_ratio(daily_returns),
            "max_drawdown": cls.max_drawdown(cum_ret),
            "annualized_return": cls.annualized_return(daily_returns),
            "calmar_ratio": cls.calmar_ratio(daily_returns),
            "total_return": float(cum_ret[-1] - 1) if len(cum_ret) > 0 else 0.0,
        }

        report = {
            "model": model_name,
            "ml_metrics": ml_metrics,
            "financial_metrics": financial_metrics,
        }

        logger.info(
            f"{model_name}: acc={ml_metrics['accuracy']:.4f}, "
            f"f1={ml_metrics['f1']:.4f}, "
            f"sharpe={financial_metrics['sharpe_ratio']:.3f}, "
            f"annual_ret={financial_metrics['annualized_return']:.2%}"
        )
        return report

    # ------------------------------------------------------------------
    # Inference timing
    # ------------------------------------------------------------------

    @staticmethod
    def measure_inference_time(
        model_fn, inputs, n_warmup: int = 3, n_trials: int = 20
    ) -> Dict:
        """
        Measures inference latency and throughput.
        n_warmup runs are excluded from statistics.
        """
        import time
        # Warmup
        for _ in range(n_warmup):
            model_fn(inputs)

        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            model_fn(inputs)
            times.append(time.perf_counter() - t0)

        batch_size = len(inputs) if hasattr(inputs, "__len__") else 1
        mean_ms = float(np.mean(times)) * 1000
        std_ms = float(np.std(times)) * 1000
        throughput = float(batch_size / np.mean(times))

        return {
            "mean_latency_ms": round(mean_ms, 3),
            "std_latency_ms": round(std_ms, 3),
            "throughput_samples_per_sec": round(throughput, 1),
            "n_trials": n_trials,
        }
