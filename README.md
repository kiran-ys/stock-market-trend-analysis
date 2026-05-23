# Stock Market Trend Analysis — Web App

**DADV Mini-Project, Team No. 3, Sapthagiri NPS University**

An interactive web application that analyses 5 years of historical stock data for
Indian (Reliance, TCS, Infosys) and US (Apple, Microsoft, Google) stocks using live data
from Yahoo Finance. Built with Python, Streamlit and Plotly.

## What it does

- Pulls **live historical data** from Yahoo Finance for any stock you select.
- Shows a **sidebar** to pick stocks, choose a date range, and toggle technical indicators.
- Renders **interactive Plotly charts** across six tabs: Overview, Trends, Risk, Correlation, Comparison, Conclusion.
- Computes moving averages, Bollinger Bands, daily returns, volatility, Sharpe ratio, and correlation.
- Compares stocks across markets with a normalised-performance chart.

## Files

| File | What it is |
|---|---|
| `app.py` | The Streamlit web app (main project). |
| `requirements.txt` | Python dependencies. |
| `Stock_Market_Analysis.ipynb` | The original Jupyter notebook version (same analysis, non-interactive). |
| `index.html` | Static HTML viewer (offline snapshot — works without Python). |
| `README.md` | This file. |

## How to run the Streamlit web app

### 1. Install Python 3.9 or newer

Check with: `python3 --version`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

### 4. Use it

- Pick one or more stocks from the **sidebar**.
- Choose a **date range** (default: 2020-01-01 to today).
- Toggle **indicators** (MA50, MA200, Bollinger Bands, Volume).
- Click through the **6 tabs** to see different analyses.

## How to use the static HTML viewer (no Python)

Just **double-click `index.html`** — it opens in any browser and shows the project with pre-computed data. No installation, no internet needed.

### Refreshing the HTML viewer with the latest data

The shipped `index.html` was built with the data available at snapshot time. To regenerate it with the latest Yahoo Finance data on your machine:

```bash
python make_snapshot.py
```

This pulls fresh data and overwrites `index.html`. Internet required.

## Deploying online (free)

1. Push these files to a public **GitHub repo**.
2. Sign up at [streamlit.io/cloud](https://streamlit.io/cloud) with your GitHub account.
3. Click "New app", choose the repo, point to `app.py`, deploy.
4. You'll get a public URL like `https://team3-stock-analysis.streamlit.app/`.

## Team

| Name | SRN |
|---|---|
| Kiran Y S | 24SUUBECS0942 |
| Kiran R | 24SUUBECS0940 |
| Keshava R | 24SUUBECS0929 |
| Kiran Kumar D K | 24SUUBECS0935 |
| K R Kotresh | 24SUUBELCS035 |
