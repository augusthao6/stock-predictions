"""
Multi-stage ML pipeline.

Pipeline stages:
  Stage 1: Data collection (StockDataCollector + news fetch)
  Stage 2: Sentiment analysis (SentimentAnalyzer via Claude API)
  Stage 3: Feature engineering (TechnicalIndicators)
  Stage 4: Preprocessing (DataPreprocessor: missing values, outliers, normalization)
  Stage 5: LSTM training (LSTMTrainer)
  Stage 6: Tree model training (TreeEnsemble)
  Stage 7: Ensemble combination (EnsembleModel)
  Stage 8: Backtesting / portfolio simulation (Backtester)

Each stage's output is the input to the next stage, making this a true multi-stage
ML pipeline where different model outputs are chained together.

AI-generated with Claude Code; reviewed and adapted by student.
"""

import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .data.collector import StockDataCollector
from .data.preprocessor import DataPreprocessor
from .data.dataset import StockSequenceDataset, create_dataloaders
from .features.technical import TechnicalIndicators
from .models.lstm_model import LSTMPredictor
from .models.tree_models import TreeEnsemble
from .models.sentiment import SentimentAnalyzer
from .models.ensemble import EnsembleModel
from .training.trainer import LSTMTrainer
from .evaluation.backtesting import Backtester
from .evaluation.metrics import TradingMetrics

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parents[2] / "models"


class TradeSagePipeline:
    """
    Orchestrates the full TradeSage multi-stage ML pipeline.
    Each method corresponds to a pipeline stage and can be called independently
    for experimentation or as part of the full end-to-end run.
    """

    def __init__(
        self,
        tickers: List[str] = None,
        seq_len: int = 20,
        batch_size: int = 64,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_sentiment: bool = True,
        anthropic_api_key: Optional[str] = None,
        device: Optional[str] = None,
    ):
        from .data.collector import TICKERS
        self.tickers = tickers or TICKERS[:5]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_sentiment = use_sentiment

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"TradeSage pipeline initialized | device={self.device} | tickers={self.tickers}")

        # Components (initialized lazily)
        self.collector = StockDataCollector(tickers=self.tickers)
        self.preprocessor = DataPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer(api_key=anthropic_api_key)
        self.tree_models = TreeEnsemble()
        self.ensemble = EnsembleModel()
        self.backtester = Backtester()

        # Data storage
        self.prices: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.sentiment_df: Optional[pd.DataFrame] = None

        # Model storage
        self.lstm_model: Optional[LSTMPredictor] = None
        self.lstm_trainer: Optional[LSTMTrainer] = None

        # Results
        self.results: Dict = {}

    # ---------------------------------------------------------------
    # Stage 1: Data Collection
    # ---------------------------------------------------------------

    def stage1_collect_data(
        self, start: str = "2015-01-01", end: str = "2024-01-01"
    ) -> pd.DataFrame:
        """Downloads and caches OHLCV data + news headlines."""
        logger.info("=== Stage 1: Data Collection ===")
        self.prices = self.collector.download_prices(start=start, end=end)
        logger.info(f"Prices shape: {self.prices.shape}")

        news_df = self.collector.collect_all_news(n_articles_per_ticker=20)
        self.results["collection_summary"] = self.collector.get_collection_summary(self.prices)
        self.results["news_count"] = len(news_df)
        return self.prices

    # ---------------------------------------------------------------
    # Stage 2: Sentiment Analysis via Claude API
    # ---------------------------------------------------------------

    def stage2_sentiment_analysis(self) -> pd.DataFrame:
        """Scores news headlines via Claude API, aggregates to daily scores."""
        logger.info("=== Stage 2: Sentiment Analysis (Claude API) ===")

        data_dir = Path(__file__).parents[2] / "data"
        news_path = data_dir / "news_headlines.csv"

        if not news_path.exists():
            logger.warning("No news headlines found; skipping sentiment")
            return pd.DataFrame()

        news_df = pd.read_csv(news_path, parse_dates=["published_at"])
        self.sentiment_df = self.sentiment_analyzer.aggregate_daily_sentiment(
            news_df, tickers=self.tickers
        )
        self.results["n_sentiment_records"] = len(self.sentiment_df)
        logger.info(f"Sentiment records: {len(self.sentiment_df)}")
        return self.sentiment_df

    # ---------------------------------------------------------------
    # Stage 3+4: Feature Engineering + Preprocessing
    # ---------------------------------------------------------------

    def stage3_feature_engineering(self, ticker: str) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """
        Computes technical indicators and merges with sentiment scores.
        Returns (feature_array, return_array, date_index).
        """
        logger.info(f"=== Stage 3+4: Feature Engineering for {ticker} ===")

        ohlcv, preproc_stats = self.preprocessor.preprocess_prices(self.prices, ticker)
        tech_features = TechnicalIndicators.compute_all(ohlcv)
        self.results[f"{ticker}_preproc_stats"] = preproc_stats

        # Merge sentiment scores
        if self.use_sentiment and self.sentiment_df is not None and len(self.sentiment_df) > 0:
            ticker_sent = self.sentiment_df[self.sentiment_df["ticker"] == ticker].copy()
            ticker_sent = ticker_sent.set_index("date")["sentiment_score"]
            tech_features = tech_features.join(ticker_sent, how="left")
            tech_features["sentiment_score"] = tech_features["sentiment_score"].fillna(0.0)
        else:
            tech_features["sentiment_score"] = 0.0

        combined = tech_features.join(ohlcv[["LogReturn"]], how="inner").dropna()

        returns = combined["LogReturn"].values
        feature_cols = [c for c in combined.columns if c != "LogReturn"]
        feature_array = combined[feature_cols].values
        dates = combined.index

        logger.info(f"  Features: {feature_array.shape}, Returns: {returns.shape}")
        return feature_array, returns, dates, feature_cols

    # ---------------------------------------------------------------
    # Stage 5: LSTM Training
    # ---------------------------------------------------------------

    def stage5_train_lstm(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[LSTMPredictor, Dict]:
        """Builds and trains the LSTM model with early stopping."""
        logger.info("=== Stage 5: LSTM Training ===")

        n = len(features)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        # Fit scaler on training data only to prevent leakage
        self.preprocessor.fit_scaler(features[:train_end])
        features_scaled = self.preprocessor.transform(features)

        train_ds = StockSequenceDataset(features_scaled[:train_end], returns[:train_end], self.seq_len)
        val_ds   = StockSequenceDataset(features_scaled[train_end:val_end], returns[train_end:val_end], self.seq_len)
        test_ds  = StockSequenceDataset(features_scaled[val_end:], returns[val_end:], self.seq_len)

        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds, batch_size=self.batch_size
        )

        self.lstm_model = LSTMPredictor(
            input_size=features_scaled.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        logger.info(f"LSTM: {self.lstm_model}")

        self.lstm_trainer = LSTMTrainer(
            model=self.lstm_model,
            learning_rate=self.lr,
            weight_decay=1e-4,
            patience=15,
            class_weights=class_weights,
            device=self.device,
        )

        history = self.lstm_trainer.fit(train_loader, val_loader, epochs=self.epochs)
        self.results["lstm_history"] = history
        self.results["train_val_test_sizes"] = {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
            "ratios": "70% / 15% / 15%",
        }
        return self.lstm_model, {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "features_scaled": features_scaled,
            "train_end": train_end,
            "val_end": val_end,
        }

    # ---------------------------------------------------------------
    # Stage 6: Tree Model Training
    # ---------------------------------------------------------------

    def stage6_train_trees(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        train_end: int,
        val_end: int,
    ) -> TreeEnsemble:
        """Trains XGBoost and Random Forest on the same feature set (no sequences needed)."""
        logger.info("=== Stage 6: Tree Model Training ===")

        labels = (returns > 0).astype(int)

        X_train, y_train = features[:train_end], labels[:train_end]
        X_val, y_val     = features[train_end:val_end], labels[train_end:val_end]

        tree_metrics = self.tree_models.fit(X_train, y_train, X_val, y_val)
        self.results["tree_metrics"] = tree_metrics
        return self.tree_models

    # ---------------------------------------------------------------
    # Stage 7: Ensemble
    # ---------------------------------------------------------------

    def stage7_ensemble(
        self,
        loaders: Dict,
        features: np.ndarray,
        returns: np.ndarray,
        train_end: int,
        val_end: int,
    ) -> EnsembleModel:
        """Trains meta-learner on validation-set predictions from all models."""
        logger.info("=== Stage 7: Ensemble ===")

        val_lstm_proba  = self.lstm_trainer.predict_proba(loaders["val_loader"])
        val_xgb_proba, val_rf_proba = self.tree_models.predict_proba(
            features[train_end:val_end][:len(val_lstm_proba)]
        )
        val_labels = (returns[train_end:val_end][:len(val_lstm_proba)] > 0).astype(int)

        if self.use_sentiment and self.sentiment_df is not None:
            val_sentiment = np.zeros(len(val_lstm_proba))
        else:
            val_sentiment = None

        ensemble_metrics = self.ensemble.fit_meta_learner(
            val_lstm_proba, val_xgb_proba, val_rf_proba, val_labels, val_sentiment
        )
        self.results["ensemble_metrics"] = ensemble_metrics
        return self.ensemble

    # ---------------------------------------------------------------
    # Stage 8: Backtesting
    # ---------------------------------------------------------------

    def stage8_backtest(
        self,
        loaders: Dict,
        features: np.ndarray,
        returns: np.ndarray,
        val_end: int,
    ) -> pd.DataFrame:
        """Runs portfolio simulation on test set and compares strategies."""
        logger.info("=== Stage 8: Backtesting ===")

        test_lstm_proba  = self.lstm_trainer.predict_proba(loaders["test_loader"])
        test_xgb_proba, test_rf_proba = self.tree_models.predict_proba(features[val_end:])
        n_test = min(len(test_lstm_proba), len(test_xgb_proba))

        lstm_preds = test_lstm_proba[:n_test].argmax(axis=1)
        ens_preds, _ = self.ensemble.predict(
            test_lstm_proba[:n_test], test_xgb_proba[:n_test], test_rf_proba[:n_test]
        )
        test_returns = returns[val_end : val_end + n_test]

        comparison_df, all_results = self.backtester.compare_strategies(
            lstm_preds, ens_preds, test_returns
        )
        self.results["backtest"] = all_results
        self.results["strategy_comparison"] = comparison_df
        logger.info("\nStrategy Comparison:\n" + comparison_df.to_string(index=False))
        return comparison_df

    # ---------------------------------------------------------------
    # Full run
    # ---------------------------------------------------------------

    def run(
        self,
        ticker: str = "AAPL",
        start: str = "2015-01-01",
        end: str = "2024-01-01",
    ) -> Dict:
        """
        Runs the complete 8-stage pipeline end-to-end for a single ticker.
        Returns a results dictionary with metrics from every stage.
        """
        self.stage1_collect_data(start=start, end=end)
        self.stage2_sentiment_analysis()

        features, returns, dates, feature_cols = self.stage3_feature_engineering(ticker)

        labels = (returns > 0).astype(int)
        class_weights_np = self.preprocessor.compute_class_weights(labels)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

        _, loaders = self.stage5_train_lstm(features, returns, class_weights)
        train_end = loaders["train_end"]
        val_end = loaders["val_end"]

        features_scaled = loaders["features_scaled"]
        self.stage6_train_trees(features_scaled, returns, train_end, val_end)
        self.stage7_ensemble(loaders, features_scaled, returns, train_end, val_end)
        comparison_df = self.stage8_backtest(loaders, features_scaled, returns, val_end)

        self.results["feature_cols"] = feature_cols
        self.results["ticker"] = ticker
        return self.results
