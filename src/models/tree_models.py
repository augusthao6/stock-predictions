"""
Gradient-boosted tree and random forest models for stock direction prediction.
Used as components in the ensemble and as baseline comparisons for the LSTM.

AI-generated with Claude Code; reviewed and adapted by student.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logger = logging.getLogger(__name__)


class TreeEnsemble:
    """
    Wraps XGBoost and Random Forest classifiers with a unified predict interface.
    Both models receive the same features (technical indicators + sentiment score),
    making them directly comparable to the LSTM for the ablation study.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42,
    ):
        self.random_state = random_state

        if HAS_XGB:
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=0.8,
                reg_alpha=0.1,   # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state,
                verbosity=0,
            )
        else:
            self.xgb_model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_state,
            )

        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth + 2,
            min_samples_split=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

        self._xgb_trained = False
        self._rf_trained = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Trains both XGBoost and Random Forest. Returns validation metrics."""
        logger.info("Training XGBoost...")
        if HAS_XGB and X_val is not None:
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.xgb_model.fit(X_train, y_train)
        self._xgb_trained = True

        logger.info("Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        self._rf_trained = True

        metrics = {}
        if X_val is not None:
            xgb_preds = self.xgb_model.predict(X_val)
            rf_preds = self.rf_model.predict(X_val)
            metrics = {
                "xgb_val_acc": float(accuracy_score(y_val, xgb_preds)),
                "xgb_val_f1": float(f1_score(y_val, xgb_preds, average="binary", zero_division=0)),
                "rf_val_acc": float(accuracy_score(y_val, rf_preds)),
                "rf_val_f1": float(f1_score(y_val, rf_preds, average="binary", zero_division=0)),
            }
            logger.info(f"XGB val acc: {metrics['xgb_val_acc']:.4f}, RF val acc: {metrics['rf_val_acc']:.4f}")
        return metrics

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (xgb_proba, rf_proba) each of shape (N, 2)."""
        xgb_proba = self.xgb_model.predict_proba(X)
        rf_proba = self.rf_model.predict_proba(X)
        return xgb_proba, rf_proba

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """Returns XGBoost feature importances for interpretability analysis."""
        if not self._xgb_trained:
            raise RuntimeError("Train model before getting feature importance")
        importances = self.xgb_model.feature_importances_
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1], reverse=True
        ))
