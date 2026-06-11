# Portfolio Risk Engine

A Python portfolio risk analytics project that loads a sample ETF portfolio, fetches historical market data, computes risk metrics, runs VaR backtests, applies stress scenarios, and visualizes results in a Streamlit dashboard.

## Features

- Portfolio-level volatility, VaR, CVaR, and drawdown metrics
- Asset-level risk contribution and attribution
- Historical stress testing
- VaR backtesting
- CSV export of analysis results
- Interactive Streamlit dashboard

## Project structure

```text
PortfolioRiskEngine/
  run.py
  dashboard.py
  requirements.txt
  README.md
  .gitignore
  data/
    portfolio.csv
  src/
    __init__.py
    backtesting.py
    data_ingestion.py
    risk_attribution.py
    risk_engine.py
    risk_metrics.py
    stress_testing.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the risk engine

```bash
python run.py --refresh
```

Optional flags:

```bash
python run.py --portfolio data/portfolio.csv --value 1000000 --years 5 --refresh --report
```

## Run the dashboard

```bash
python -m streamlit run dashboard.py
```

## Portfolio input format

The default portfolio lives at `data/portfolio.csv` and should use this format:

```csv
ticker,weight,asset_class,description
SPY,0.40,Equity,S&P 500 ETF
QQQ,0.25,Equity,Nasdaq 100 ETF
TLT,0.20,Bond,20+ Year Treasury ETF
GLD,0.10,Commodity,Gold ETF
VNQ,0.05,Real Estate,REIT ETF
```

Generated files such as price history, returns, and analysis outputs are ignored by Git and recreated when the project runs.
