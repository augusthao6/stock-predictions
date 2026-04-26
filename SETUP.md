# Setup Instructions

## Requirements

- Python 3.10 or higher
- pip
- 4 GB RAM minimum (8 GB recommended for full dataset)
- GPU optional (CPU-only training works fine, ~5-10 min per model)

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "import torch; import yfinance; import anthropic; print('All packages installed successfully')"
```

## API Keys

### Claude API (Optional — for live sentiment analysis)

TradeSage uses the Anthropic Claude API to analyze financial news sentiment. Without an API key, the system automatically falls back to a rule-based sentiment scorer, which still works correctly.

To enable Claude API sentiment analysis:

```bash
# macOS/Linux
export ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Windows Command Prompt
set ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

You can get a free API key at https://console.anthropic.com/

### Yahoo Finance (No API key required)

Stock price data is downloaded via `yfinance`, which is free and requires no API key.

## Running the Project

### Option 1: Jupyter Notebooks (Recommended for grading)

```bash
jupyter notebook notebooks/
```

Run notebooks in order:
1. `01_data_exploration.ipynb` — Data collection, preprocessing, EDA
2. `02_model_training.ipynb` — Model training, hyperparameter search, training curves
3. `03_ablation_study.ipynb` — Ablation study comparing design choices
4. `04_evaluation.ipynb` — Full evaluation, backtesting, error analysis

### Option 2: Python script (Full pipeline)

```bash
# From the TradeSage root directory:
python -c "
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from src.pipeline import TradeSagePipeline

pipe = TradeSagePipeline(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    epochs=60,
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
)
results = pipe.run(ticker='AAPL', start='2015-01-01', end='2024-01-01')
print(results['strategy_comparison'])
"
```

## Data

Stock price data is downloaded automatically on first run and cached in the `data/` directory as Parquet files. Re-downloads are only triggered when the cache is missing or `force_refresh=True`.

Expected download time: ~30 seconds for 5 tickers × 9 years.

## Troubleshooting

**`yfinance` rate limiting**: If download fails, wait 60 seconds and retry. The collector retries automatically.

**`torch` not found**: Ensure you ran `pip install -r requirements.txt` in the correct virtual environment.

**Notebook kernel not found**: Run `python -m ipykernel install --user --name=venv` after activating your virtual environment.

**Claude API 401 error**: Check that `ANTHROPIC_API_KEY` is set correctly. The system will fall back to rule-based sentiment automatically.
