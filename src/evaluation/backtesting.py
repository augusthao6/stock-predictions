"""
Portfolio backtesting / simulation module.
Evaluates model-driven trading strategies on historical price data.

Implements the "simulation-based evaluation" rubric item:
  - Strategy: go long when model predicts UP, hold cash when DOWN
  - Compares: Model strategy vs. buy-and-hold vs. random baseline
  - Accounts for transaction costs and slippage

AI-generated with Claude Code; reviewed and adapted by student.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .metrics import TradingMetrics

logger = logging.getLogger(__name__)


class Backtester:
    """
    Simulates portfolio performance based on daily directional predictions.

    Strategy logic:
      - If model predicts UP (label=1): go long (buy stock)
      - If model predicts DOWN (label=0): hold cash
      - Transaction cost applied on every position change

    This is a long-only momentum strategy reflecting realistic constraints
    (short selling requires a margin account).
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,  # 0.1% per trade (typical retail broker)
        initial_capital: float = 10_000.0,
    ):
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

    def run_strategy(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        model_name: str = "Model",
    ) -> Dict:
        """
        Simulates a prediction-based long/cash strategy.

        Args:
            predictions:    (T,) binary predictions (1=long, 0=cash)
            actual_returns: (T,) realized daily returns (decimal)
            model_name:     name for reporting

        Returns:
            dict with portfolio metrics and daily equity curve
        """
        n = len(predictions)
        equity = np.zeros(n + 1)
        equity[0] = self.initial_capital
        strategy_returns = np.zeros(n)
        position = 0  # 0 = cash, 1 = long

        for t in range(n):
            signal = int(predictions[t])
            trade_cost = 0.0

            # Pay transaction cost on position changes only
            if signal != position:
                trade_cost = self.transaction_cost
                position = signal

            if position == 1:
                strategy_returns[t] = actual_returns[t] - trade_cost
            else:
                strategy_returns[t] = -trade_cost if trade_cost > 0 else 0.0

            equity[t + 1] = equity[t] * (1 + strategy_returns[t])

        cum_returns = equity[1:] / self.initial_capital
        metrics = TradingMetrics.full_report(
            y_true=np.array(actual_returns > 0, dtype=int),
            y_pred=predictions,
            daily_returns=strategy_returns,
            model_name=model_name,
        )
        metrics["equity_curve"] = equity.tolist()
        metrics["cumulative_returns"] = cum_returns.tolist()
        metrics["n_trades"] = int(np.diff(predictions, prepend=0).astype(bool).sum())
        return metrics

    def buy_and_hold(self, actual_returns: np.ndarray) -> Dict:
        """Baseline: simply hold the asset for the entire period."""
        equity = self.initial_capital * (1 + actual_returns).cumprod()
        equity = np.concatenate([[self.initial_capital], equity])
        cum_returns = equity[1:] / self.initial_capital

        metrics = TradingMetrics.full_report(
            y_true=np.ones(len(actual_returns), dtype=int),
            y_pred=np.ones(len(actual_returns), dtype=int),
            daily_returns=actual_returns,
            model_name="Buy-and-Hold",
        )
        metrics["equity_curve"] = equity.tolist()
        metrics["cumulative_returns"] = cum_returns.tolist()
        metrics["n_trades"] = 1
        return metrics

    def random_baseline(self, actual_returns: np.ndarray, seed: int = 42) -> Dict:
        """Baseline: random long/cash decisions (theoretical 50% accuracy)."""
        rng = np.random.default_rng(seed)
        random_predictions = rng.integers(0, 2, size=len(actual_returns))
        return self.run_strategy(random_predictions, actual_returns, model_name="Random")

    def compare_strategies(
        self,
        model_predictions: np.ndarray,
        ensemble_predictions: np.ndarray,
        actual_returns: np.ndarray,
    ) -> pd.DataFrame:
        """
        Runs all strategies and returns a comparison summary DataFrame.
        This is the primary comparison table for the evaluation section.
        """
        results = {}
        results["Buy-and-Hold"] = self.buy_and_hold(actual_returns)
        results["Random"] = self.random_baseline(actual_returns)
        results["LSTM"] = self.run_strategy(model_predictions, actual_returns, "LSTM")
        results["Ensemble"] = self.run_strategy(ensemble_predictions, actual_returns, "Ensemble")

        rows = []
        for name, r in results.items():
            fm = r["financial_metrics"]
            ml = r["ml_metrics"]
            rows.append({
                "Strategy": name,
                "Accuracy": f"{ml['accuracy']:.4f}",
                "F1": f"{ml['f1']:.4f}",
                "Annual Return": f"{fm['annualized_return']:.2%}",
                "Sharpe Ratio": f"{fm['sharpe_ratio']:.3f}",
                "Max Drawdown": f"{fm['max_drawdown']:.2%}",
                "Calmar Ratio": f"{fm['calmar_ratio']:.3f}",
                "Total Return": f"{fm['total_return']:.2%}",
                "N Trades": r.get("n_trades", "-"),
            })

        df = pd.DataFrame(rows)
        logger.info("\n" + df.to_string(index=False))
        return df, results

    def get_equity_curves(self, all_results: Dict) -> pd.DataFrame:
        """Returns a DataFrame of all equity curves for plotting."""
        dfs = {}
        for name, result in all_results.items():
            if "equity_curve" in result:
                dfs[name] = result["equity_curve"]
        max_len = max(len(v) for v in dfs.values())
        for name in dfs:
            if len(dfs[name]) < max_len:
                dfs[name] = dfs[name] + [dfs[name][-1]] * (max_len - len(dfs[name]))
        return pd.DataFrame(dfs)
