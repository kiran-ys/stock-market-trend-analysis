"""
Stock Market Trend Analysis Dashboard

Features:
  - Live Yahoo Finance data with auto-refresh
  - Custom ticker search
  - Quick date-range presets (1M / 3M / 6M / 1Y / 5Y / MAX)
  - Line / Candlestick chart toggle
  - Major-event markers (COVID, War, Fed hikes, ChatGPT, etc.)
  - NIFTY 50 and S&P 500 benchmark comparison
  - AI-style auto-generated commentary
  - Buy recommendation with concrete forward projections

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    IST = None
import warnings
warnings.filterwarnings('ignore')

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
DEFAULT_TICKERS = {
    "Reliance":   "RELIANCE.NS",
    "TCS":        "TCS.NS",
    "Infosys":    "INFY.NS",
    "Apple":      "AAPL",
    "Microsoft":  "MSFT",
    "Google":     "GOOGL",
}
NIFTY = "^NSEI"   # NIFTY 50 — Indian benchmark
SP500 = "^GSPC"   # S&P 500   — US benchmark

# (date, label, color)
EVENTS = [
    ("2020-03-23", "COVID Market Crash",        "#ef4444"),
    ("2020-11-09", "COVID Vaccine Announced",   "#22c55e"),
    ("2022-02-24", "Russia–Ukraine War",        "#ef4444"),
    ("2022-03-16", "Fed Begins Rate Hikes",     "#fbbf24"),
    ("2022-11-30", "ChatGPT Launches (AI Boom)","#a855f7"),
    ("2023-03-10", "SVB Bank Collapse",         "#ef4444"),
    ("2024-08-05", "Yen Carry-Trade Unwind",    "#fbbf24"),
    ("2025-04-02", "Tariff Shock Sell-off",     "#ef4444"),
]

COLORS = ['#38bdf8', '#f97316', '#10b981', '#ef4444', '#a855f7',
          '#fbbf24', '#ec4899', '#06b6d4', '#84cc16']

# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_data(tickers, start, end):
    """Download historical OHLCV data for a list of tickers and enrich it."""
    if not tickers:
        return {}
    raw = yf.download(tickers, start=start, end=end, group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t].copy() if len(tickers) > 1 else raw.copy()
        except Exception:
            continue
        df = df.dropna(how="all").ffill()
        if df.empty or "Close" not in df:
            continue
        df["Daily Return"] = df["Close"].pct_change() * 100
        df["MA50"]   = df["Close"].rolling(50).mean()
        df["MA200"]  = df["Close"].rolling(200).mean()
        df["BB_Mid"]   = df["Close"].rolling(20).mean()
        df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
        df["BB_Lower"] = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()
        out[t] = df
    return out


@st.cache_data(ttl=30, show_spinner=False)
def get_live_quotes(tickers):
    """Get latest intraday quotes for the live ticker bar (30s cache)."""
    out = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period="1d", interval="1m")
            if hist.empty:
                continue
            last = hist.iloc[-1]
            open_today = hist.iloc[0]["Open"]
            out[t] = {
                "price":      float(last["Close"]),
                "change":     float(last["Close"] - open_today),
                "change_pct": float((last["Close"] / open_today - 1) * 100),
                "as_of":      hist.index[-1].strftime("%H:%M:%S"),
                "volume":     int(hist["Volume"].sum()),
            }
        except Exception:
            continue
    return out


# ──────────────────────────────────────────────────────────────
# ANALYTICS
# ──────────────────────────────────────────────────────────────
def project_future(close, days_ahead=90, lookback=60):
    """Project the next N business days using linear regression on last K days.
       Returns DataFrame with date, projection, lower 95%, upper 95%."""
    recent = close.dropna().tail(lookback)
    if len(recent) < 10:
        return None
    x = np.arange(len(recent), dtype=float)
    coef = np.polyfit(x, recent.values, 1)            # slope, intercept
    fit = coef[0] * x + coef[1]
    residual_std = float((recent.values - fit).std())
    future_x = np.arange(len(recent), len(recent) + days_ahead, dtype=float)
    proj  = coef[0] * future_x + coef[1]
    upper = proj + 1.96 * residual_std
    lower = proj - 1.96 * residual_std
    last_date = recent.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                  periods=days_ahead)
    return pd.DataFrame({"date": future_dates, "projection": proj,
                         "lower": lower, "upper": upper})


def score_stocks(stocks, ticker_to_label):
    """Composite score for each stock — higher is better."""
    rows = []
    for t, df in stocks.items():
        close = df["Close"]
        ret   = df["Daily Return"].dropna()
        if len(close) < 60 or len(ret) < 30:
            continue
        lookback = min(252, len(close) - 1)
        annual_ret = (close.iloc[-1] / close.iloc[-lookback] - 1) * 100
        sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0
        ma200 = df["MA200"].iloc[-1]
        trend = 1 if (not np.isnan(ma200)) and close.iloc[-1] > ma200 else 0
        mom_20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0
        vol = float(ret.std())
        roll_max = close.cummax()
        max_dd = float(((close - roll_max) / roll_max).min() * 100)

        score = (0.30 * annual_ret
                 + 0.20 * sharpe * 10
                 + 0.15 * trend * 10
                 + 0.20 * mom_20
                 - 0.10 * vol * 10
                 - 0.05 * abs(max_dd))

        rows.append({
            "Stock":          ticker_to_label.get(t, t),
            "Ticker":         t,
            "Current Price":  round(close.iloc[-1], 2),
            "1Y Return %":    round(annual_ret, 2),
            "Sharpe":         round(sharpe, 2),
            "20d Momentum %": round(mom_20, 2),
            "Volatility %":   round(vol, 2),
            "Max Drawdown %": round(max_dd, 2),
            "Trend":          "Bullish" if trend else "Bearish",
            "Score":          round(score, 2),
        })
    return sorted(rows, key=lambda x: x["Score"], reverse=True)


def generate_commentary(stocks, ticker_to_label):
    """Generate plain-English findings from the computed statistics."""
    points = []
    if not stocks:
        return ["No data available."]

    # Best performer
    growth = {n: (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
              for n, df in [(ticker_to_label.get(t, t), df) for t, df in stocks.items()]}
    best_growth = max(growth, key=growth.get)
    points.append(f"**{best_growth}** delivered the strongest growth over the period, "
                  f"returning **{growth[best_growth]:+.1f}%**.")

    # Lowest volatility
    vols = {ticker_to_label.get(t, t): df["Daily Return"].std()
            for t, df in stocks.items() if df["Daily Return"].notna().sum() > 10}
    if vols:
        steadiest = min(vols, key=vols.get)
        points.append(f"**{steadiest}** is the steadiest performer with daily volatility of "
                      f"only **{vols[steadiest]:.2f}%** — well suited to risk-averse investors.")

    # Highest momentum
    moms = {}
    for t, df in stocks.items():
        c = df["Close"]
        if len(c) > 21:
            moms[ticker_to_label.get(t, t)] = (c.iloc[-1] / c.iloc[-21] - 1) * 100
    if moms:
        hottest = max(moms, key=moms.get)
        if moms[hottest] > 0:
            points.append(f"**{hottest}** has the strongest 20-day momentum at "
                          f"**{moms[hottest]:+.2f}%** — currently in an upswing.")

    # Golden cross check
    crosses = []
    for t, df in stocks.items():
        if df["MA50"].notna().any() and df["MA200"].notna().any():
            if df["MA50"].iloc[-1] > df["MA200"].iloc[-1]:
                crosses.append(ticker_to_label.get(t, t))
    if crosses:
        points.append(f"**Golden cross signal active** (50-MA above 200-MA, a classic "
                      f"bullish indicator) for: " + ", ".join(crosses) + ".")

    # Correlation insight
    ret_df = pd.DataFrame({ticker_to_label.get(t, t): df["Daily Return"]
                           for t, df in stocks.items()})
    corr = ret_df.corr()
    if len(corr) >= 2:
        # Use a writable copy: df.values can be read-only on newer NumPy/pandas,
        # which makes np.fill_diagonal raise a ValueError on Streamlit Cloud.
        corr_vals = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_vals, np.nan)
        corr = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)
        lo_pair = corr.stack().idxmin()
        points.append(f"**{lo_pair[0]}** and **{lo_pair[1]}** have the lowest correlation "
                      f"({corr.loc[lo_pair[0], lo_pair[1]]:.2f}) — pairing them in a "
                      f"portfolio would maximise diversification.")
    return points


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
st.sidebar.title("📈 Stock Analysis")

# Live tracking
st.sidebar.subheader("⚡ Live Tracking")
auto_refresh = st.sidebar.toggle("Auto-refresh data", value=False,
                                  help="Re-pull data automatically on a timer.")
if auto_refresh:
    refresh_secs = st.sidebar.slider("Refresh every (seconds)", 10, 300, 30, 5)
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_secs * 1000, key="auto_refresh")
    else:
        st.sidebar.warning("Install `streamlit-autorefresh` for auto-update:\n"
                           "`pip install streamlit-autorefresh`")

if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Stock selection
st.sidebar.subheader("📊 Stocks")
selected_labels = st.sidebar.multiselect(
    "Choose from defaults",
    options=list(DEFAULT_TICKERS.keys()),
    default=list(DEFAULT_TICKERS.keys()),
)
custom_input = st.sidebar.text_input("Add custom ticker(s)",
                                      placeholder="e.g. TSLA, NVDA, HDFCBANK.NS",
                                      help="Comma-separate. Yahoo symbols only.").strip()

selected_tickers = [DEFAULT_TICKERS[lbl] for lbl in selected_labels]
ticker_to_label  = {v: k for k, v in DEFAULT_TICKERS.items()}
if custom_input:
    for tkr in [t.strip().upper() for t in custom_input.split(",") if t.strip()]:
        if tkr not in selected_tickers:
            selected_tickers.append(tkr)
            ticker_to_label[tkr] = tkr

# Stocks added from the main Overview page (persist across reruns)
st.session_state.setdefault("extra_tickers", [])
for tkr in st.session_state["extra_tickers"]:
    if tkr not in selected_tickers:
        selected_tickers.append(tkr)
        ticker_to_label.setdefault(tkr, tkr)

# Date range
st.sidebar.subheader("📅 Date Range")
preset = st.sidebar.radio("Quick presets",
                          ["1M", "3M", "6M", "1Y", "5Y", "MAX", "Custom"],
                          index=4, horizontal=True)
today = date.today()
preset_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 365 * 5, "MAX": 365 * 8}
if preset in preset_map:
    start_date = today - timedelta(days=preset_map[preset])
    end_date   = today
else:
    c1, c2 = st.sidebar.columns(2)
    with c1: start_date = st.date_input("Start", date(2020, 1, 1), max_value=today)
    with c2: end_date   = st.date_input("End", today, min_value=start_date, max_value=today)

# Chart style
st.sidebar.subheader("📈 Chart Style")
chart_style = st.sidebar.radio("Price chart type",
                                ["Line", "Candlestick"],
                                horizontal=True)

# Indicators
st.sidebar.subheader("🔧 Indicators")
show_ma50  = st.sidebar.checkbox("50-day Moving Average",  True)
show_ma200 = st.sidebar.checkbox("200-day Moving Average", True)
show_bb    = st.sidebar.checkbox("Bollinger Bands (20-day)", False)
show_vol   = st.sidebar.checkbox("Volume bars",            False)
show_events = st.sidebar.checkbox("Major event markers",    True)

st.sidebar.markdown("---")
st.sidebar.caption("Data: Yahoo Finance (live).")

# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
header_l, header_r = st.columns([0.7, 0.3])
with header_l:
    st.title("📈 Stock Market Trend Analysis")
    st.caption("Real-time analytics with live Yahoo Finance data.")
with header_r:
    now = datetime.now(IST).strftime("%a, %b %d %Y · %H:%M:%S IST") if IST \
          else datetime.now().strftime("%a, %b %d %Y · %H:%M:%S")
    badge = ("🟢 <b>LIVE</b>" if auto_refresh else "⚪ Static")
    st.markdown(f"""
    <div style="text-align:right; padding-top:18px;">
      <div style="font-size:18px;">{badge}</div>
      <div style="font-size:12px; color:#888;">{now}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Guard
if not selected_tickers:
    st.warning("👈 Pick at least one stock from the sidebar.")
    st.stop()

# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
with st.spinner("Pulling data from Yahoo Finance..."):
    stocks = load_data(selected_tickers, start_date, end_date)
    bench = load_data([NIFTY, SP500], start_date, end_date)
    quotes = get_live_quotes(selected_tickers)

if not stocks:
    st.error("Yahoo Finance returned no data. Check tickers and date range.")
    st.stop()

# Warn about any requested ticker that returned no usable data
missing = [t for t in selected_tickers if t not in stocks]
if missing:
    st.warning(
        "Couldn't load data for: " + ", ".join(missing) +
        ". Check the symbol (Indian stocks need the correct NSE name, e.g. HDFCBANK.NS) "
        "or pick a wider date range — newly listed stocks may lack history."
    )

# ──────────────────────────────────────────────────────────────
# LIVE TICKER BAR
# ──────────────────────────────────────────────────────────────
if quotes:
    cols = st.columns(min(len(quotes), 6))
    for col, (t, q) in zip(cols, quotes.items()):
        col.metric(
            label=f"{ticker_to_label.get(t, t)} ({t})",
            value=f"{q['price']:,.2f}",
            delta=f"{q['change']:+.2f} ({q['change_pct']:+.2f}%)",
        )
    st.caption(f"💹 Live intraday quotes — most recent tick at {list(quotes.values())[0]['as_of']}.")
    st.markdown("---")

# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "📈 Trends",
    "🎲 Risk & Returns",
    "🔗 Correlation",
    "⚖️ Comparison vs Benchmark",
    "🤖 Insights & Recommendation",
])

# ============== TAB 1 — OVERVIEW ==============
with tabs[0]:
    st.header("Overview")

    # ---- Add-a-stock search (flows into every tab automatically) ----
    st.markdown("**➕ Add a stock** — search any company and it appears across all tabs.")
    ac1, ac2, ac3 = st.columns([0.55, 0.30, 0.15])
    with ac1:
        new_symbol = st.text_input(
            "Stock symbol", key="add_symbol",
            placeholder="e.g. HDFCBANK, TATAMOTORS, TSLA, NVDA",
            label_visibility="collapsed")
    with ac2:
        new_market = st.selectbox(
            "Market", ["🇮🇳 India (NSE)", "🇺🇸 US"], key="add_market",
            label_visibility="collapsed")
    with ac3:
        add_clicked = st.button("➕ Add", use_container_width=True)

    if add_clicked and new_symbol.strip():
        sym = new_symbol.strip().upper()
        if new_market.startswith("🇮🇳") and not sym.endswith(".NS"):
            sym = sym + ".NS"
        if sym in selected_tickers:
            st.info(f"{sym} is already shown.")
        else:
            st.session_state["extra_tickers"].append(sym)
            st.rerun()

    # Show / clear stocks added from this page
    if st.session_state["extra_tickers"]:
        rc1, rc2 = st.columns([0.8, 0.2])
        rc1.caption("Added here: " + ", ".join(st.session_state["extra_tickers"]))
        if rc2.button("Clear added", use_container_width=True):
            st.session_state["extra_tickers"] = []
            st.rerun()

    # ---- Live price cards for every selected stock ----
    if quotes:
        st.markdown("##### 💹 Live prices")
        qitems = list(quotes.items())
        for i in range(0, len(qitems), 4):
            row = qitems[i:i + 4]
            cols = st.columns(len(row))
            for col, (t, q) in zip(cols, row):
                col.metric(
                    label=f"{ticker_to_label.get(t, t)} ({t})",
                    value=f"{q['price']:,.2f}",
                    delta=f"{q['change']:+.2f} ({q['change_pct']:+.2f}%)",
                )

    st.markdown("---")
    st.subheader("Descriptive Statistics")
    st.markdown("Summary of selected stocks: starting price, current price, range, average, volatility.")
    rows = []
    for t, df in stocks.items():
        c = df["Close"]
        rows.append({
            "Company":     ticker_to_label.get(t, t),
            "Ticker":      t,
            "Start Price": round(c.iloc[0], 2),
            "End Price":   round(c.iloc[-1], 2),
            "Min":         round(c.min(), 2),
            "Max":         round(c.max(), 2),
            "Mean":        round(c.mean(), 2),
            "Std Dev":     round(c.std(), 2),
            "Return %":    round((c.iloc[-1] / c.iloc[0] - 1) * 100, 2),
        })
    summary = pd.DataFrame(rows)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Per-stock CSV download
    with st.expander("⬇  Download CSV"):
        for t, df in stocks.items():
            csv = df.reset_index().to_csv(index=False).encode()
            st.download_button(
                label=f"Download {ticker_to_label.get(t, t)} ({t}) data",
                data=csv,
                file_name=f"{t}_history.csv",
                mime="text/csv",
                key=f"dl_{t}"
            )

# ============== TAB 2 — TRENDS ==============
with tabs[1]:
    st.header("Price Trends & Indicators")
    st.markdown(f"Currently showing **{chart_style}** chart. "
                "Toggle indicators and event markers in the sidebar.")

    for t, df in stocks.items():
        st.subheader(f"{ticker_to_label.get(t, t)} ({t})")
        rows_n = 2 if show_vol else 1
        fig = make_subplots(rows=rows_n, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25] if show_vol else [1],
                            vertical_spacing=0.04,
                            subplot_titles=("Price", "Volume") if show_vol else ("Price",))

        # Price trace
        if chart_style == "Candlestick" and {"Open", "High", "Low", "Close"}.issubset(df.columns):
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                          low=df["Low"], close=df["Close"], name="OHLC",
                                          increasing_line_color="#22c55e",
                                          decreasing_line_color="#ef4444",
                                          showlegend=False), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                                     line=dict(color="#38bdf8", width=1.6)), row=1, col=1)

        # Indicators
        if show_ma50:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50",
                                     line=dict(color="orange", width=1.4)), row=1, col=1)
        if show_ma200:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="MA200",
                                     line=dict(color="red", width=1.4)), row=1, col=1)
        if show_bb:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                                     line=dict(color="gray", dash="dot", width=0.9)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                                     line=dict(color="gray", dash="dot", width=0.9),
                                     fill="tonexty",
                                     fillcolor="rgba(128,128,128,0.08)"), row=1, col=1)
        if show_vol:
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                                 marker_color="teal", opacity=0.5,
                                 showlegend=False), row=2, col=1)

        # Event markers
        if show_events:
            x_start, x_end = df.index.min(), df.index.max()
            for d, label, colr in EVENTS:
                evt = pd.Timestamp(d)
                if x_start <= evt <= x_end:
                    fig.add_vline(x=evt, line_width=1, line_dash="dash",
                                  line_color=colr, row=1, col=1)
                    fig.add_annotation(x=evt, y=1, yref="paper",
                                       text=label, showarrow=False,
                                       font=dict(size=9, color=colr),
                                       bgcolor="rgba(0,0,0,0.6)",
                                       textangle=-90, xanchor="left",
                                       row=1, col=1)

        fig.update_layout(height=560 if show_vol else 420,
                          margin=dict(l=20, r=20, t=40, b=20),
                          legend=dict(orientation="h", y=1.08),
                          hovermode="x unified",
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# ============== TAB 3 — RISK & RETURNS ==============
with tabs[2]:
    st.header("Risk & Returns")
    st.markdown("Daily-return distributions and volatility ranking for selected stocks.")
    returns = pd.DataFrame({ticker_to_label.get(t, t): df["Daily Return"] for t, df in stocks.items()})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Daily Returns Distribution")
        melted = returns.melt(var_name="Stock", value_name="Daily Return %").dropna()
        fig = px.histogram(melted, x="Daily Return %", color="Stock",
                           nbins=80, barmode="overlay", opacity=0.55)
        fig.add_vline(x=0, line_color="white", line_dash="dash", line_width=0.8)
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Volatility Ranking")
        vol = returns.std().sort_values().reset_index()
        vol.columns = ["Stock", "Volatility %"]
        fig = px.bar(vol, x="Stock", y="Volatility %",
                     color="Volatility %",
                     color_continuous_scale="RdYlGn_r",
                     text=vol["Volatility %"].round(2))
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20),
                          showlegend=False, coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sharpe Ratio (Risk-Adjusted Return)")
    st.caption("Mean daily return ÷ std-dev × √252.  Higher = better risk-adjusted return.")
    sharpe = ((returns.mean() / returns.std()) * np.sqrt(252)).round(3) \
                 .sort_values(ascending=False).reset_index()
    sharpe.columns = ["Stock", "Sharpe Ratio"]
    st.dataframe(sharpe, use_container_width=True, hide_index=True)

# ============== TAB 4 — CORRELATION ==============
with tabs[3]:
    st.header("Correlation of Daily Returns")
    st.markdown("+1 = perfectly correlated · 0 = independent · −1 = opposite.")
    returns = pd.DataFrame({ticker_to_label.get(t, t): df["Daily Return"] for t, df in stocks.items()})
    corr = returns.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ============== TAB 5 — COMPARISON vs BENCHMARK ==============
with tabs[4]:
    st.header("Comparison vs Market Benchmark")
    st.markdown("Each stock and benchmark indexed to **100** at the start date so growth is "
                "directly comparable across markets.")

    norm = pd.DataFrame()
    for t, df in stocks.items():
        norm[ticker_to_label.get(t, t)] = (df["Close"] / df["Close"].iloc[0]) * 100
    benchmark_lines = {}
    if NIFTY in bench and not bench[NIFTY].empty:
        norm["NIFTY 50"] = (bench[NIFTY]["Close"] / bench[NIFTY]["Close"].iloc[0]) * 100
        benchmark_lines["NIFTY 50"] = "#fbbf24"
    if SP500 in bench and not bench[SP500].empty:
        norm["S&P 500"] = (bench[SP500]["Close"] / bench[SP500]["Close"].iloc[0]) * 100
        benchmark_lines["S&P 500"] = "#a855f7"

    fig = go.Figure()
    for i, col in enumerate(norm.columns):
        if col in benchmark_lines:
            fig.add_trace(go.Scatter(x=norm.index, y=norm[col], name=col,
                                     line=dict(width=2.5, dash="dash",
                                               color=benchmark_lines[col])))
        else:
            fig.add_trace(go.Scatter(x=norm.index, y=norm[col], name=col,
                                     line=dict(width=1.6,
                                               color=COLORS[i % len(COLORS)])))
    fig.add_hline(y=100, line_color="white", line_dash="dot", line_width=0.8)
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20),
                      yaxis_title="Indexed Price (Start = 100)",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Alpha table — outperformance vs benchmark
    st.subheader("Alpha — Excess Return Over Benchmark")
    alpha_rows = []
    for t, df in stocks.items():
        is_indian = t.endswith(".NS")
        bench_key = NIFTY if is_indian else SP500
        if bench_key in bench and not bench[bench_key].empty:
            stock_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
            bench_ret = (bench[bench_key]["Close"].iloc[-1] / bench[bench_key]["Close"].iloc[0] - 1) * 100
            alpha_rows.append({
                "Stock":      ticker_to_label.get(t, t),
                "Stock Return %":     round(stock_ret, 2),
                "Benchmark":  "NIFTY 50" if is_indian else "S&P 500",
                "Benchmark Return %": round(bench_ret, 2),
                "Alpha %":    round(stock_ret - bench_ret, 2),
                "Outperformed?": "✅" if stock_ret > bench_ret else "❌",
            })
    if alpha_rows:
        st.dataframe(pd.DataFrame(alpha_rows), use_container_width=True, hide_index=True)

# ============== TAB 6 — INSIGHTS & RECOMMENDATION ==============
with tabs[5]:
    st.header("🤖 Auto-Generated Insights")
    insights = generate_commentary(stocks, ticker_to_label)
    for i, p in enumerate(insights, 1):
        st.markdown(f"**{i}.** {p}")

    st.markdown("---")
    st.header("💡 Recommendation — Which Stock to Buy")
    st.caption("Composite-score rank: 30 % annual return + 20 % Sharpe + 15 % trend + "
               "20 % momentum − 10 % volatility − 5 % drawdown.")

    scored = score_stocks(stocks, ticker_to_label)
    if scored:
        best = scored[0]
        # Big card
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0ea5e9,#1e40af);
                    color:white; padding:24px; border-radius:14px;
                    box-shadow:0 4px 16px rgba(0,0,0,.2)">
          <div style="font-size:13px; text-transform:uppercase; letter-spacing:2px; opacity:.85">
            Top Pick
          </div>
          <div style="font-size:34px; font-weight:700; margin-top:6px;">
            {best['Stock']} <span style="opacity:.7; font-size:20px;">({best['Ticker']})</span>
          </div>
          <div style="margin-top:14px; font-size:15px; line-height:1.7;">
            Current price: <b>{best['Current Price']}</b><br>
            1-Year Return: <b>{best['1Y Return %']:+.2f}%</b> · Sharpe: <b>{best['Sharpe']:.2f}</b> ·
            20-day Momentum: <b>{best['20d Momentum %']:+.2f}%</b><br>
            Volatility: <b>{best['Volatility %']:.2f}%</b> · Max Drawdown: <b>{best['Max Drawdown %']:.2f}%</b> ·
            Trend: <b>{best['Trend']}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Forward projection
        st.subheader("📅 Projected Prices (Statistical Forecast)")
        best_df = stocks[best["Ticker"]]
        proj = project_future(best_df["Close"], days_ahead=90, lookback=60)

        if proj is not None:
            today_str = date.today().isoformat()
            p30 = proj.iloc[29]
            p60 = proj.iloc[59]
            p90 = proj.iloc[89]
            cur = best["Current Price"]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(label=f"In 30 days ({p30['date'].date()})",
                          value=f"{p30['projection']:,.2f}",
                          delta=f"{(p30['projection']/cur - 1)*100:+.2f}%")
                st.caption(f"Range: {p30['lower']:,.2f} – {p30['upper']:,.2f}")
            with c2:
                st.metric(label=f"In 60 days ({p60['date'].date()})",
                          value=f"{p60['projection']:,.2f}",
                          delta=f"{(p60['projection']/cur - 1)*100:+.2f}%")
                st.caption(f"Range: {p60['lower']:,.2f} – {p60['upper']:,.2f}")
            with c3:
                st.metric(label=f"In 90 days ({p90['date'].date()})",
                          value=f"{p90['projection']:,.2f}",
                          delta=f"{(p90['projection']/cur - 1)*100:+.2f}%")
                st.caption(f"Range: {p90['lower']:,.2f} – {p90['upper']:,.2f}")

            # Chart with history + projection
            fig = go.Figure()
            recent_close = best_df["Close"].tail(180)
            fig.add_trace(go.Scatter(x=recent_close.index, y=recent_close.values,
                                     name="Historical", line=dict(color="#38bdf8", width=1.6)))
            fig.add_trace(go.Scatter(x=proj["date"], y=proj["projection"],
                                     name="Projected", line=dict(color="#22c55e", width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=pd.concat([proj["date"], proj["date"][::-1]]),
                                     y=pd.concat([proj["upper"], proj["lower"][::-1]]),
                                     fill="toself", fillcolor="rgba(34,197,94,0.15)",
                                     line=dict(width=0), showlegend=True,
                                     name="95% Confidence"))
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20),
                              hovermode="x unified",
                              title=f"{best['Stock']} — Last 180 Days + 90-Day Projection")
            st.plotly_chart(fig, use_container_width=True)

            # Reasoning
            reasoning = []
            if best["Trend"] == "Bullish":
                reasoning.append("Price is **above the 200-day moving average** — a bullish long-term signal.")
            if best["Sharpe"] > 1.0:
                reasoning.append(f"Sharpe ratio of **{best['Sharpe']:.2f}** indicates strong risk-adjusted returns.")
            if best["20d Momentum %"] > 0:
                reasoning.append(f"Positive 20-day momentum of **{best['20d Momentum %']:+.2f}%** confirms a short-term uptrend.")
            if best["1Y Return %"] > 0:
                reasoning.append(f"1-year return of **{best['1Y Return %']:+.2f}%** demonstrates sustained performance.")

            if reasoning:
                st.subheader("🔍 Why this stock?")
                for r in reasoning:
                    st.markdown(f"- {r}")

            st.warning("⚠️ **Disclaimer** — This is a **statistical projection** based on linear "
                       "regression of recent price history, not financial advice. Markets can "
                       "move unexpectedly due to events not captured by historical data. "
                       "Always do your own research and consult a qualified financial advisor "
                       "before investing.")

        # Full ranking
        st.subheader("📊 Full Ranking — All Stocks Scored")
        rank_df = pd.DataFrame(scored)
        rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))
        st.dataframe(rank_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Built with Python · Streamlit · Plotly · yfinance — live data from Yahoo Finance.")
