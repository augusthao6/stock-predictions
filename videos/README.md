# Videos

Place your video files here:

- `demo.mp4` — 3–5 minute demo video (for non-specialist audience, no code shown)
- `walkthrough.mp4` — 5–10 minute technical walkthrough (for ML engineers)

## Recording Tips

### Demo Video (3–5 min)
- Open with: "TradeSage predicts stock price direction by combining LSTM neural networks, gradient-boosted trees, and AI news sentiment analysis"
- Show the equity curve comparison chart from `docs/backtest_results.png`
- Show the strategy comparison table
- Explain why this matters: better risk-adjusted returns than buy-and-hold in several periods

### Technical Walkthrough (5–10 min)
- Walk through `src/pipeline.py` showing the 8 stages
- Show `src/models/lstm_model.py` and explain the LSTM architecture (attention → batch norm → MLP head)
- Explain how attention works in transformers (for the "architecture explanation" rubric item)
- Show `notebooks/02_model_training.ipynb` training curves
- Show `notebooks/03_ablation_study.ipynb` comparison table
- Show `notebooks/04_evaluation.ipynb` error analysis
- Discuss design decisions from `docs/design_decisions.md`
