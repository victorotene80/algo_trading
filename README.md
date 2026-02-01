# FX AI Bot (M5) — LightGBM + Calibrated Probabilities + Paper Trading

## What it does
- Pulls M5 candles for EUR/USD and USD/JPY from Twelve Data
- Stores candles in SQLite
- Builds features (returns, EMA diff, volatility, ATR)
- Trains a LightGBM classifier to predict probability of positive return over next horizon
- Calibrates probabilities (isotonic) so p_up is more meaningful
- Runs a live paper-trading loop
- Enforces risk rules:
  - Max 2% risk per trade
  - Max 3% daily loss (halts trading for the day)
  - Max 5 open positions total
- Uses:
  - SL = 1.5 × ATR(14)
  - TP = 1.2 × SL (1.2R)
  - Time stop: 6 bars (30 minutes)

## Setup
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set TWELVEDATA_API_KEY
python scripts/fetch_history.py
python scripts/train_model.py
python scripts/run_paper.py
