# Setup Instructions

## Requirements

- Python 3.10 or higher
- pip
- 4 GB RAM minimum (8 GB recommended for full 10-ticker dataset)
- GPU strongly recommended for training (CPU works but takes ~10 min per model)

---

## Option A: Google Colab (free GPU!)
Run all 4 notebooks in colab

### Step 1 — Upload the project to Google Drive

Clone or download the repository and place the `stock-predictions/` folder in the root of your Google Drive (`My Drive/stock-predictions/`).

### Step 2 — Open a notebook in Colab

Open any notebook from `notebooks/` in Google Colab. In each notebook, the first two cells handle setup automatically:

```python
from google.colab import drive
drive.mount('/content/gdrive')

%cd /content/gdrive/MyDrive/stock-predictions
```

### Step 3 — Set GPU runtime

In Colab: **Runtime → Change runtime type → T4 GPU**

### Step 4 — (Optional) Add Claude API key for live sentiment

In Colab, go to the key icon (🔑) in the left sidebar → **Secrets** → add a secret named `ANTHROPIC_API_KEY` with your key value.

Then in the notebook cell before creating the sentiment analyzer:

```python
import os
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
```

You can get a free API key at https://console.anthropic.com/

Without a key, the system automatically falls back to a rule-based sentiment scorer.

### Step 5 — Run notebooks in order

1. `notebooks/1_data.ipynb` — Data collection, EDA, sentiment analysis (~5 min)
2. `notebooks/2_modeltraining.ipynb` — LSTM + tree training, hyperparameter search (~10 min with GPU)
3. `notebooks/3_ablationstudy.ipynb` — Ablation study (~8 min)
4. `notebooks/4_evaluation.ipynb` — Backtesting, error analysis (~5 min)

---

## Option B: Local Installation

### Step 1 — Create a virtual environment

```bash
python -m venv venv

# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Verify installation

```bash
python -c "import torch; import yfinance; import anthropic; import xgboost; print('All packages installed successfully')"
```

### Step 4 — (Optional) Set Claude API key

```bash
# macOS/Linux
export ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Windows Command Prompt
set ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

### Step 5 — Launch notebooks

```bash
jupyter notebook notebooks/
```

Run in order: `1_data.ipynb` → `2_modeltraining.ipynb` → `3_ablationstudy.ipynb` → `4_evaluation.ipynb`

---

## Data

Stock price data is downloaded automatically on first run via the Yahoo Finance API (`yfinance`) and cached in `data/prices_<start>_<end>.parquet`. News headlines are cached in `data/news_headlines.csv`. Re-downloads only happen when `force_refresh=True`.

Expected download time: ~30 seconds for 10 tickers × 9 years of daily data.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `yfinance` rate limiting error | Wait 60 seconds; the collector retries automatically up to 3 times |
| `torch` not found | Confirm you activated your virtual environment before `pip install` |
| Notebook kernel not found | Run `python -m ipykernel install --user --name=venv` after activating venv |
| Claude API 401 error | Check `ANTHROPIC_API_KEY` is set; system auto-falls back to rule-based scorer |
| `ModuleNotFoundError: pyarrow` | Run `pip install pyarrow` — required for Parquet cache files |
| Colab disconnects mid-training | Re-run from cell 5 in `2_modeltraining.ipynb` (data is cached, won't re-download) |
