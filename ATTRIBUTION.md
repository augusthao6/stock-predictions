# Attribution

## AI Development Tools

This project was developed using **Claude Code** (Anthropic's AI coding assistant, model: claude-sonnet-4-6) as the primary AI development tool.

### Substantive Account of AI Tool Usage

**What was generated:** Claude Code generated the initial scaffolding and implementation of the following files:
- `src/data/collector.py` — yfinance API integration and caching logic
- `src/data/preprocessor.py` — data cleaning pipeline structure
- `src/data/dataset.py` — PyTorch Dataset class skeleton
- `src/features/technical.py` — technical indicator formulas
- `src/models/lstm_model.py` — LSTM architecture class
- `src/models/tree_models.py` — XGBoost and Random Forest wrappers
- `src/models/sentiment.py` — Claude API integration
- `src/models/ensemble.py` — meta-learner structure
- `src/training/trainer.py` — training loop with early stopping
- `src/training/hyperopt.py` — grid search framework
- `src/evaluation/metrics.py`, `backtesting.py`, `analysis.py` — evaluation utilities
- `src/pipeline.py` — pipeline orchestration
- All four Jupyter notebooks — cell structure and visualization code

**What was modified:** The student reviewed and adapted all generated code. Key modifications included:
1. **Architecture decisions**: Changed from simple LSTM to stacked bidirectional option with orthogonal initialization (forget gate bias initialization was a non-obvious choice added after reviewing LSTM training literature)
2. **Feature engineering**: Added the normalized OBV feature (OBV / rolling std) which is not in standard implementations — raw OBV is non-stationary across stocks and would break the scaler
3. **Ensemble design**: Changed from weighted average to a stacking meta-learner (logistic regression on val-set predictions) to avoid optimistic bias from in-sample weighting
4. **Backtesting**: Added Calmar ratio and win rate; changed transaction cost model from per-trade percentage to position-change-only (more realistic)
5. **Data split**: Verified the chronological split is strictly non-overlapping (train/val/test) to prevent data leakage — the initial generated code used random shuffling which would leak future data into training

**What required debugging / reworking:**
- The initial `collect_all_news` implementation assumed a non-existent `gnews` library; switched to Yahoo Finance's built-in news endpoint
- MultiIndex column handling in yfinance output changed between library versions; added explicit flattening logic
- The `feature_importance` call used a deprecated XGBoost attribute; fixed to use `.feature_importances_` property
- `StockSequenceDataset._build_sequences` had an off-by-one error in the return label indexing (was predicting current-day return instead of next-day)

## External Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.0 | LSTM implementation, training loop, DataLoaders |
| yfinance | ≥0.2.28 | Free stock price data via Yahoo Finance API |
| XGBoost | ≥2.0 | Gradient-boosted tree classifier |
| scikit-learn | ≥1.3 | Random Forest, Logistic Regression meta-learner, metrics |
| anthropic | ≥0.18 | Claude API client for news sentiment analysis |
| pandas | ≥2.0 | Data manipulation and time-series indexing |
| numpy | ≥1.24 | Numerical computation |
| matplotlib / seaborn | ≥3.7 / ≥0.12 | Visualization |

## Datasets

**Stock price data**: Downloaded via Yahoo Finance API through the `yfinance` Python library. Tickers: AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, JPM, BRK-B, UNH. Period: 2015-01-01 to 2024-01-01. Adjusted for splits and dividends. No license restrictions on Yahoo Finance historical data for research use.

**Financial news headlines**: Fetched via Yahoo Finance's built-in news endpoint (available through `yfinance.Ticker.news`). Used for sentiment analysis only; raw text not stored long-term.

## Code Comments

Each source file includes a file-level AI attribution comment (see the docstring header in each `.py` file). Functions modified substantially by the student have additional comments explaining the non-obvious design choice.
