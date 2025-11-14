# FAI — Fundamental AI Crypto Analytics

FAI is a multi-language crypto analytics dashboard that combines:

- On-chain & market metrics
- Simple F-Score style rating
- 24h move probabilities
- Risk / macro commentary
- News sentiment and headlines
- Portfolio-style comparison of multiple assets

## How it works

- **Backend:** FastAPI (`main.py`)
- **Frontend:** Single-page HTML (`index.html`)
- **Dependencies:** see `requirements.txt`

The project is currently focused on crypto assets (BTC, ETH, SOL, BNB, etc.) and is built as an MVP that can later be extended to stocks, ETFs and other markets.

## Local run (dev)

```bash
# install dependencies
pip install -r requirements.txt

# run backend
uvicorn main:app --reload --port 8000
