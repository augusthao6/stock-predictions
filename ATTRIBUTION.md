# Attribution

## AI Development Tools

This project was developed with Claude Code (Anthropic) as the primary AI development tool.

### Substantive Account of AI Tool Usage

**What was AI-generated:**

Claude Code generated the initial scaffolding and implementation of the following files:
- `src/data/collector.py` — yfinance API integration and caching logic
- `src/data/preprocessor.py` — data cleaning and scaling pipeline structure
- `src/data/dataset.py` — PyTorch Dataset and DataLoader utilities
- `src/features/technical.py` — technical indicator formulas (RSI, MACD, Bollinger Bands, etc.)
- `src/models/lstm_model.py` — LSTM architecture class skeleton
- `src/models/tree_models.py` — XGBoost and Random Forest wrappers
- `src/models/sentiment.py` — Claude API integration for news sentiment
- `src/models/ensemble.py` — meta-learner ensemble structure
- `src/training/trainer.py` — training loop with early stopping and LR scheduling
- `src/training/hyperopt.py` — grid search framework
- `src/evaluation/metrics.py`, `backtesting.py`, `analysis.py` — evaluation utilities
- `src/pipeline.py` — end-to-end pipeline orchestration
- All four Jupyter notebooks — initial cell structure and visualization code

**What was modified and adapted:**

1. **Data leakage fix (most significant)**: The initial tree model training used same-day features to predict same-day returns — a severe data leakage issue. `CloseOpen_Gap = log(Close_t / Open_t)` approximates the same-day log return for large-cap stocks, giving 99–100% accuracy on training data. The fix applies a per-ticker 1-day feature lag (features at t−1 predict return at t), matching the LSTM's sequence alignment. This dropped tree accuracy from 99% to a realistic 50–55%.

2. **yfinance news API compatibility**: The new yfinance (≥0.2.50) nests article fields under a `content` sub-object instead of returning flat dicts. Added dual-structure parsing in `collector.py` to handle both `item["title"]` and `item["content"]["title"]`, plus ISO date string and Unix timestamp parsing.

3. **ReduceLROnPlateau deprecation**: The `verbose` keyword argument was removed in newer PyTorch versions. Removed from `trainer.py` and replaced with explicit `print()` statements for epoch-by-epoch progress visibility in Colab.

4. **Multi-ticker cache collision**: The price cache was keyed only on the date range, not the ticker list. Running with 10 tickers after a cached 5-ticker run returned stale data. Fixed by adding `force_refresh=True` and documenting the cache key behavior.

5. **Normalized OBV**: Raw OBV is non-stationary and grows without bound across tickers, making the scaler meaningless. Changed to `OBV / rolling_std(OBV, 50)` in `technical.py` to produce a stationary, cross-ticker comparable signal.

6. **Ensemble meta-learner class balance**: The initial logistic regression meta-learner had no class weighting and learned to always predict UP (recall = 1.0, effectively identical to buy-and-hold). Added `class_weight='balanced'` to force the learner to find a real decision boundary.

7. **Ensemble training set**: The initial code evaluated the meta-learner's accuracy on the same validation data it was trained on, reporting inflated metrics. The meta-learner is now evaluated on held-out test data only.

8. **Pooled multi-ticker training**: Initial implementation trained on AAPL alone (~1,500 samples). Expanded to pool all 10 tickers with per-ticker chronological 70/15/15 splits before concatenation (~15,470 training samples), which substantially improves LSTM generalization.

9. **Sentiment pipeline historical coverage**: Live Yahoo Finance news returns only today's headlines (all dated today). Added `generate_price_based_headlines()` to create semantically meaningful synthetic headlines from historical price movements, giving the sentiment feature non-zero signal across the full 2017–2026 training window.

**What required debugging and reworking:**

- `collect_all_news()` initially fetched zero historical headlines; all 70 live headlines were dated 2026-04-26 (today). Root cause: live news has no historical timestamps. Fixed by always appending price-based synthetic historical headlines regardless of live news volume.
- Stale `.pyc` bytecache in Colab caused `unexpected keyword argument 'prices'` even after updating `collector.py`. Resolved by restarting the Colab runtime to clear the cache.
- `NameError: features_scaled` after the multi-ticker refactor — the variable was renamed to `X_train` but the LSTM training cell still referenced the old name. Fixed.
- Tree model architecture ablation in notebooks 3 and 4 still used same-day features after the main fix in notebook 2. Updated all three notebooks to use the 1-day lag consistently.

---

## External Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.0 | Custom LSTM, training loop, DataLoaders |
| yfinance | ≥0.2.50 | Stock price data and news headlines via Yahoo Finance API |
| XGBoost | ≥2.0 | Gradient-boosted tree classifier with L1/L2 regularization |
| scikit-learn | ≥1.3 | Random Forest, Logistic Regression meta-learner, metrics, StandardScaler |
| anthropic | ≥0.18 | Claude API client for financial news sentiment analysis |
| pandas | ≥2.0 | Data manipulation and time-series indexing |
| numpy | ≥1.24 | Numerical computation |
| matplotlib / seaborn | ≥3.7 / ≥0.12 | Visualization |
| pyarrow | ≥12.0 | Parquet file I/O for price data cache |

---

## Datasets

**Stock price data**: Downloaded via Yahoo Finance API through the `yfinance` Python library.
- Tickers: AAPL, MSFT, GOOGL, AMZN, NVDA, META, JPM, GS, SPY, QQQ
- Period: 2017-04-26 to 2026-04-26 (9 years of daily OHLCV data)
- Auto-adjusted for splits and dividends
- No license restrictions on Yahoo Finance historical data for educational/research use

**Financial news headlines**: Fetched via Yahoo Finance's built-in news endpoint (available through `yfinance.Ticker.news`). Used exclusively for sentiment scoring; raw text is not stored beyond the session.

**Synthetic price-based headlines**: Generated programmatically from historical return data in `collector.py` `generate_price_based_headlines()`. These are not real news — they are template-based descriptions of historical price movements used to give the sentiment feature historical coverage.

---

## Code Comments

Each source file includes a file-level AI attribution comment in its module docstring. The pattern used is:

```
AI-generated with Claude Code; reviewed and adapted by student.
```

Functions or classes where modifications were substantial include inline comments explaining the non-obvious design choice (e.g., the forget-gate bias initialization in `lstm_model.py`, the 1-day lag in notebook cell 14, the normalized OBV in `technical.py`).
