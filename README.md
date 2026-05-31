# 📊 Market Regime Analyzer

Public web app that classifies any tradeable asset's current market regime using a Hidden Markov Model.

Type a symbol — get back:
- Current regime classification (crash / bear / neutral / bull / euphoria)
- HMM confidence and posterior probabilities
- BIC-selected optimal regime count
- Recent price chart with regime-colored bands
- Per-regime statistics + forward return analysis

## Supported symbol formats

| Asset class | Examples |
|---|---|
| Stocks / ETFs | `SPY`, `NVDA`, `AAPL`, `QQQ`, `IWM` |
| Futures | `ES=F`, `NQ=F`, `CL=F`, `GC=F`, `ZN=F` |
| Indices | `^GSPC`, `^IXIC`, `^VIX` |
| Forex | `EURUSD=X`, `USDJPY=X` |
| Crypto (yfinance) | `BTC-USD`, `ETH-USD` |
| Crypto (Kraken via CCXT) | `BTC/USDT`, `ETH/USDT` |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

App will open at `http://localhost:8501`.

## Deploy to Streamlit Cloud (free)

1. Push this folder to a public GitHub repo
2. Visit https://share.streamlit.io and sign in with GitHub
3. Click **New app** → pick the repo → main file = `app.py`
4. Click **Deploy** — done in ~3 minutes
5. Share the public URL (e.g. `regime-analyzer.streamlit.app`)

## Disclaimer

This is an educational tool, not investment advice. The HMM classifies historical regimes and the current bar; it does not predict future returns. Trading involves substantial risk. Verify all data independently before making any investment decisions.
