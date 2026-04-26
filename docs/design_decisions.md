# Design Decisions

## 1. Prediction Task: Direction vs. Return Magnitude

**Decision**: Predict binary direction (up/down) rather than exact return magnitude.

**Tradeoff**: Regression (predicting magnitude) is theoretically more informative, but financial returns are extremely noisy with heavy tails. Binary classification has a cleaner loss signal and the resulting Sharpe ratio of a direction-based strategy is comparable to magnitude-based strategies with much lower model complexity.

**Evidence**: In preliminary experiments, an MSE-optimized LSTM had ~0.001 R² on test returns, confirming that return magnitude is essentially unpredictable at 1-day horizon. Direction accuracy of ~55% is statistically significant and actionable.

## 2. Chronological Train/Val/Test Split (vs. Random)

**Decision**: Strict chronological 70/15/15 split with no temporal overlap.

**Why**: Financial time series have temporal autocorrelation — shuffling would leak future market information into training, producing artificially inflated metrics. This is called "look-ahead bias" and is a critical correctness requirement in financial ML.

## 3. RobustScaler vs. StandardScaler

**Decision**: Use `RobustScaler` (median/IQR) instead of `StandardScaler` (mean/std).

**Why**: Financial features like volume and ATR have fat-tailed distributions with extreme outliers. StandardScaler is sensitive to outliers and can compress 95% of values into a tiny range. RobustScaler is resistant to this.

## 4. Stacking Ensemble (Meta-Learner) vs. Simple Average

**Decision**: Train a logistic regression meta-learner on validation-set predictions.

**Why**: Simple averaging assumes all models contribute equally and independently. In practice, LSTM captures temporal patterns that trees miss, while XGBoost handles interaction effects better. The meta-learner learns to weight them adaptively. Using validation-set predictions prevents the meta-learner from overfitting to training data.

## 5. LSTM Hidden Size: 128 vs. Larger

**Decision**: hidden_size=128 (not 256 or 512).

**Tradeoff (accuracy vs. latency)**: Larger hidden states have more capacity but take longer to train and are more prone to overfitting on the relatively small financial dataset (~2,000 training sequences). Hyperparameter search confirmed 128 is the sweet spot — 256 did not improve val accuracy and trained 3× slower.

## 6. Claude Haiku vs. Sonnet for Sentiment

**Decision**: Use `claude-haiku-4-5-20251001` for news sentiment analysis.

**Tradeoff (accuracy vs. cost/latency)**: Haiku is ~10× cheaper and faster than Sonnet. For short financial headlines with clear positive/negative vocabulary, Haiku achieves comparable sentiment accuracy to Sonnet while being suitable for batch-processing hundreds of headlines. The model's sentiment is used as one feature among many (not the primary signal), so marginal accuracy gains from a larger model don't justify the cost.

## 7. Long-Only Strategy in Backtesting

**Decision**: Trading strategy is long/cash only (no shorting).

**Why**: Short selling requires a margin account and has unlimited downside risk, making it unsuitable for most retail investors. A long/cash strategy is more realistic and directly interpretable as an investment decision. This also makes the backtesting assumptions more conservative.
