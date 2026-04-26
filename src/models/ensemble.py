"""
Ensemble model combining LSTM, XGBoost, Random Forest, and sentiment scores.

Combination strategy: learned meta-weights via logistic regression on validation
predictions. This is "stacking" rather than simple averaging, allowing the ensemble
to learn which base model to trust in different market conditions.

AI-generated with Claude Code; reviewed and adapted by student.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Meta-learner that combines predictions from:
      1. LSTM (temporal patterns in price/volume features)
      2. XGBoost (non-linear interactions in technical indicators)
      3. Random Forest (variance reduction, decorrelated with XGBoost)
      4. Sentiment score (news-based signal)

    The meta-learner is a logistic regression trained on held-out validation
    predictions, so it cannot overfit to the training data.
    """

    def __init__(self, weights: Optional[List[float]] = None):
        # weights = [lstm_w, xgb_w, rf_w, sentiment_w]
        self.weights = np.array(weights) if weights else None
        self.meta_learner = LogisticRegression(C=1.0, max_iter=500, random_state=42,
                                               class_weight='balanced')
        self._meta_trained = False

    # ------------------------------------------------------------------
    # Building prediction matrices
    # ------------------------------------------------------------------

    def build_prediction_matrix(
        self,
        lstm_proba: np.ndarray,
        xgb_proba: np.ndarray,
        rf_proba: np.ndarray,
        sentiment_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Stacks model output probabilities and optional sentiment into a feature matrix
        for the meta-learner.

        Args:
            lstm_proba:       (N, 2) softmax probabilities from LSTM
            xgb_proba:        (N, 2) probabilities from XGBoost
            rf_proba:         (N, 2) probabilities from Random Forest
            sentiment_scores: (N,) daily sentiment scores in [-1, 1]

        Returns:
            (N, K) feature matrix where K = 6 or 7 (with sentiment)
        """
        parts = [lstm_proba, xgb_proba, rf_proba]
        if sentiment_scores is not None:
            # Normalize sentiment to [0, 1] and broadcast as a 2-col feature
            s_norm = (sentiment_scores + 1.0) / 2.0
            sentiment_2col = np.stack([1 - s_norm, s_norm], axis=1)
            parts.append(sentiment_2col)
        return np.concatenate(parts, axis=1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_meta_learner(
        self,
        val_lstm_proba: np.ndarray,
        val_xgb_proba: np.ndarray,
        val_rf_proba: np.ndarray,
        val_labels: np.ndarray,
        val_sentiment: Optional[np.ndarray] = None,
    ) -> Dict:
        """Trains the logistic regression meta-learner on validation-set predictions."""
        X_meta = self.build_prediction_matrix(
            val_lstm_proba, val_xgb_proba, val_rf_proba, val_sentiment
        )
        self.meta_learner.fit(X_meta, val_labels)
        self._meta_trained = True

        preds = self.meta_learner.predict(X_meta)
        metrics = {
            "ensemble_val_acc": float(accuracy_score(val_labels, preds)),
            "ensemble_val_f1": float(f1_score(val_labels, preds, zero_division=0)),
            "meta_coefficients": self.meta_learner.coef_.tolist(),
        }
        logger.info(f"Ensemble val acc: {metrics['ensemble_val_acc']:.4f}")
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        lstm_proba: np.ndarray,
        xgb_proba: np.ndarray,
        rf_proba: np.ndarray,
        sentiment_scores: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predicted_labels, predicted_probabilities_up).
        Uses meta-learner if trained, otherwise falls back to weighted average.
        """
        if self._meta_trained:
            X_meta = self.build_prediction_matrix(
                lstm_proba, xgb_proba, rf_proba, sentiment_scores
            )
            labels = self.meta_learner.predict(X_meta)
            proba = self.meta_learner.predict_proba(X_meta)[:, 1]
        else:
            proba = self._weighted_average(lstm_proba, xgb_proba, rf_proba, sentiment_scores)
            labels = (proba > 0.5).astype(int)
        return labels, proba

    def _weighted_average(
        self,
        lstm_proba: np.ndarray,
        xgb_proba: np.ndarray,
        rf_proba: np.ndarray,
        sentiment_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simple fallback: weighted average of up-probabilities."""
        w = self.weights if self.weights is not None else np.array([0.4, 0.3, 0.2, 0.1])
        proba_up = lstm_proba[:, 1]
        xgb_up = xgb_proba[:, 1]
        rf_up = rf_proba[:, 1]

        if sentiment_scores is not None:
            sent_up = (sentiment_scores + 1.0) / 2.0
            combined = (w[0] * proba_up + w[1] * xgb_up + w[2] * rf_up + w[3] * sent_up)
        else:
            w_norm = w[:3] / w[:3].sum()
            combined = w_norm[0] * proba_up + w_norm[1] * xgb_up + w_norm[2] * rf_up
        return combined

    def evaluate(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        model_name: str = "Ensemble",
    ) -> Dict:
        return {
            "model": model_name,
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
        }
