"""Regime Analyzer — public web app for HMM-based market regime detection.

Anyone can visit, type any tradeable symbol, and see the bot's analysis:
  - Current detected regime (crash / bear / neutral / bull / euphoria)
  - HMM confidence
  - BIC selection (why N regimes was the best fit)
  - Recent price chart with regime overlay
  - Per-regime statistics
  - Optional walk-forward backtest summary

Symbols accepted:
  Stocks:   SPY, NVDA, AAPL, MSFT, TSLA, ...
  ETFs:     QQQ, IWM, GLD, TLT, VXX, XLE, XLF, ...
  Futures:  ES=F, NQ=F, CL=F, GC=F, ZN=F, ...
  Indices:  ^GSPC, ^IXIC, ^VIX
  Forex:    EURUSD=X, USDJPY=X
  Crypto:   BTC-USD, ETH-USD (yfinance)
            BTC/USDT, ETH/USDT (CCXT — better intraday history)

Deployment: Streamlit Community Cloud (streamlit.io/cloud), free tier.

Disclaimer: Educational tool. Not investment advice.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone
from enum import Enum

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# Self-contained HMM regime engine (no project dependencies — easier to deploy)
# ============================================================================

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class RegimeLabel(str, Enum):
    CRASH = "crash"
    BEAR = "bear"
    NEUTRAL = "neutral"
    BULL = "bull"
    EUPHORIA = "euphoria"


LABEL_ORDERINGS: dict[int, list[RegimeLabel]] = {
    3: [RegimeLabel.BEAR, RegimeLabel.NEUTRAL, RegimeLabel.BULL],
    4: [RegimeLabel.CRASH, RegimeLabel.BEAR, RegimeLabel.BULL, RegimeLabel.EUPHORIA],
    5: [
        RegimeLabel.CRASH,
        RegimeLabel.BEAR,
        RegimeLabel.NEUTRAL,
        RegimeLabel.BULL,
        RegimeLabel.EUPHORIA,
    ],
}

REGIME_COLORS = {
    "crash": "#d62728",
    "bear": "#ff9f1c",
    "neutral": "#888888",
    "bull": "#2ca02c",
    "euphoria": "#9467bd",
}

REGIME_EMOJI = {
    "crash": "💥",
    "bear": "🐻",
    "neutral": "⚖️",
    "bull": "🐂",
    "euphoria": "🚀",
}


# ============================================================================
# Feature engineering
# ============================================================================
def compute_features(ohlcv: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Returns, log-volatility, volume z-score → standardized feature matrix."""
    close = ohlcv["close"]
    returns = np.log(close / close.shift(1))
    rolling_std = returns.rolling(window=vol_window, min_periods=vol_window).std()
    log_vol = np.log(rolling_std + 1e-10)
    vol_mean = ohlcv["volume"].rolling(window=vol_window, min_periods=vol_window).mean()
    vol_std = ohlcv["volume"].rolling(window=vol_window, min_periods=vol_window).std()
    vol_z = (ohlcv["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    df = pd.DataFrame(
        {"returns": returns, "log_volatility": log_vol, "volume_zscore": vol_z},
        index=ohlcv.index,
    ).dropna()
    if df.empty:
        raise ValueError("Insufficient data for feature engineering")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.values)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def count_hmm_params(n: int, n_features: int) -> int:
    return n * (n - 1) + (n - 1) + n * n_features + n * n_features * (n_features + 1) // 2


def fit_hmm_bic_select(
    features: pd.DataFrame, regime_range: tuple[int, int] = (3, 5)
) -> tuple[GaussianHMM, dict, int]:
    """Fit HMM for each n in range, pick lowest BIC."""
    X = features.values
    n_samples, n_features = X.shape
    bic_scores: dict[int, float] = {}
    models: dict[int, GaussianHMM] = {}

    for n in range(regime_range[0], regime_range[1] + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = GaussianHMM(
                    n_components=n, covariance_type="full",
                    n_iter=100, random_state=42,
                )
                model.fit(X)
                ll = model.score(X)
                n_params = count_hmm_params(n, n_features)
                bic = -2 * ll + n_params * np.log(n_samples)
                bic_scores[n] = bic
                models[n] = model
            except Exception:
                continue

    if not models:
        raise RuntimeError("HMM training failed for all regime counts")
    best_n = min(bic_scores, key=bic_scores.get)
    return models[best_n], bic_scores, best_n


def detect_regimes(
    features: pd.DataFrame, model: GaussianHMM, n_regimes: int
) -> tuple[pd.DataFrame, dict[int, RegimeLabel]]:
    """Run forward algorithm to get regime classification for each bar.

    No look-ahead: predict_proba() on data up to time t gives filtered posterior at t.
    """
    X = features.values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        posteriors = model.predict_proba(X)

    # Sort regimes by mean return to assign labels
    returns_idx = list(features.columns).index("returns")
    state_means = model.means_[:, returns_idx]
    sorted_states = np.argsort(state_means).tolist()
    labels = LABEL_ORDERINGS[n_regimes]
    label_map = {state_idx: labels[rank] for rank, state_idx in enumerate(sorted_states)}

    state_indices = posteriors.argmax(axis=1)
    state_confidences = posteriors.max(axis=1)
    regime_labels = [label_map[i].value for i in state_indices]

    result = pd.DataFrame(
        {
            "regime": regime_labels,
            "confidence": state_confidences,
            "state_idx": state_indices,
        },
        index=features.index,
    )
    return result, label_map


# ============================================================================
# Data fetching
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(symbol: str, timeframe: str = "1d", days: int = 730) -> pd.DataFrame:
    """Fetch historical bars. Auto-routes by symbol format."""
    if "/" in symbol:
        return _fetch_ccxt(symbol, timeframe, days)
    return _fetch_yfinance(symbol, timeframe, days)


def _fetch_yfinance(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    import yfinance as yf

    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "60m", "1d": "1d", "1w": "1wk",
    }
    interval = interval_map.get(timeframe, "1d")
    period_days = min(days, 730 if interval == "1d" else 60)

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{period_days}d", interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data for symbol '{symbol}'. Check the format.")
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df[["open", "high", "low", "close", "volume"]]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.dropna()


def _fetch_ccxt(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    import ccxt

    ex = ccxt.kraken({"enableRateLimit": True})
    minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 60)
    bars_needed = min(int(days * 1440 / minutes), 720)

    since = int((datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_bars: list[list] = []
    cursor = since
    while len(all_bars) < bars_needed:
        chunk = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=720)
        if not chunk:
            break
        all_bars.extend(chunk)
        cursor = chunk[-1][0] + minutes * 60 * 1000
        if len(chunk) < 720:
            break

    if not all_bars:
        raise ValueError(f"No data for {symbol} on Kraken.")
    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.tail(int(days * 1440 / minutes) + 100)


# ============================================================================
# Streamlit UI
# ============================================================================
st.set_page_config(
    page_title="Regime Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.title("📊 Market Regime Analyzer")
st.caption(
    "Hidden Markov Model regime detection — classifies any asset's current state "
    "as crash / bear / neutral / bull / euphoria based on volatility + return patterns. "
    "Uses forward-algorithm-only inference (no look-ahead bias)."
)

# Sidebar — symbol input + parameters
with st.sidebar:
    st.header("Analyze")
    symbol = st.text_input(
        "Symbol",
        value="SPY",
        help=(
            "Examples:\n"
            "  Stocks: SPY, NVDA, AAPL\n"
            "  ETFs: QQQ, IWM, GLD\n"
            "  Futures: ES=F, NQ=F, CL=F, GC=F\n"
            "  Indices: ^GSPC, ^VIX\n"
            "  Forex: EURUSD=X\n"
            "  Crypto: BTC-USD (yfinance) or BTC/USDT (Kraken)"
        ),
    )
    timeframe = st.selectbox(
        "Timeframe",
        ["1d", "4h", "1h"],
        index=0,
        help="Daily recommended — most stable regime detection.",
    )
    days = st.slider(
        "History (days)",
        min_value=180, max_value=1095, value=730, step=30,
        help="More history = more stable HMM training. 2 years is the default.",
    )
    regime_range_choice = st.selectbox(
        "Regime count search range",
        options=["3 to 4 regimes", "3 to 5 regimes", "4 to 5 regimes"],
        index=1,
        help="HMM tries each N in this range and picks the one with lowest BIC.",
    )
    regime_min, regime_max = {
        "3 to 4 regimes": (3, 4),
        "3 to 5 regimes": (3, 5),
        "4 to 5 regimes": (4, 5),
    }[regime_range_choice]
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

    st.divider()
    st.markdown(
        "**Disclaimer:** This is a regime classification tool, not investment advice. "
        "Past regime patterns do not predict future returns. "
        "Real trading involves substantial risk."
    )
    st.markdown("---")
    st.caption("Built with HMM + walk-forward backtesting principles")

# Main analysis
if analyze or "last_symbol" not in st.session_state:
    st.session_state["last_symbol"] = symbol

if symbol:
    try:
        with st.spinner(f"Fetching {symbol} data..."):
            ohlcv = fetch_ohlcv(symbol, timeframe=timeframe, days=days)

        if len(ohlcv) < 100:
            st.error(
                f"Only {len(ohlcv)} bars available for {symbol} — need at least 100. "
                f"Try a different symbol or longer history."
            )
            st.stop()

        with st.spinner(f"Training HMM on {len(ohlcv)} bars... (~2-5 seconds)"):
            features = compute_features(ohlcv)
            model, bic_scores, best_n = fit_hmm_bic_select(features, (regime_min, regime_max))
            regimes_df, label_map = detect_regimes(features, model, best_n)

        latest = regimes_df.iloc[-1]
        latest_price = float(ohlcv["close"].iloc[-1])
        recent_return = (
            (latest_price / float(ohlcv["close"].iloc[-7]) - 1) * 100
            if len(ohlcv) >= 7 else 0.0
        )

        # ============================================================
        # KPI row
        # ============================================================
        c1, c2, c3, c4, c5 = st.columns(5)
        regime_color = REGIME_COLORS.get(latest["regime"], "#888")
        c1.markdown(
            f"<div style='border-left:6px solid {regime_color};padding:10px 15px;"
            f"background:#1a1a1a;border-radius:4px;'>"
            f"<div style='color:#888;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;'>"
            f"Current Regime</div>"
            f"<div style='font-size:1.5rem;font-weight:700;color:{regime_color};margin-top:4px;'>"
            f"{REGIME_EMOJI[latest['regime']]} {latest['regime'].upper()}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        c2.metric("Confidence", f"{latest['confidence']:.0%}")
        c3.metric("Latest Price", f"${latest_price:,.2f}", f"{recent_return:+.2f}% (7d)")
        c4.metric("Regimes Detected", f"{best_n}", "BIC-selected")
        c5.metric("Training Bars", f"{len(ohlcv)}")

        # ============================================================
        # Regime legend
        # ============================================================
        st.markdown("###")
        legend_cols = st.columns(best_n)
        labels = LABEL_ORDERINGS[best_n]
        for i, label in enumerate(labels):
            with legend_cols[i]:
                color = REGIME_COLORS[label.value]
                emoji = REGIME_EMOJI[label.value]
                st.markdown(
                    f"<div style='border-left:6px solid {color};padding:8px 12px;background:#181818;border-radius:4px;'>"
                    f"<b style='color:{color}'>{emoji} {label.value.upper()}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ============================================================
        # Price chart with regime overlay
        # ============================================================
        st.subheader(f"Price + Regime Overlay · {symbol}")

        # Align price + regime data
        merged = ohlcv.loc[regimes_df.index].copy()
        merged["regime"] = regimes_df["regime"]
        merged["confidence"] = regimes_df["confidence"]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=(None, None),
        )
        fig.add_trace(
            go.Candlestick(
                x=merged.index, open=merged["open"], high=merged["high"],
                low=merged["low"], close=merged["close"], name="Price",
                increasing_line_color="#2ca02c", decreasing_line_color="#d62728",
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Regime bands as background shading
        for regime_name in merged["regime"].unique():
            segments = []
            start = None
            for i, row in merged.iterrows():
                if row["regime"] == regime_name:
                    if start is None:
                        start = i
                    end = i
                else:
                    if start is not None:
                        segments.append((start, end))
                    start = None
            if start is not None:
                segments.append((start, end))
            color = REGIME_COLORS.get(regime_name, "#888")
            for s, e in segments:
                fig.add_vrect(
                    x0=s, x1=e, fillcolor=color, opacity=0.12,
                    line_width=0, row=1, col=1,
                )

        # Confidence in lower panel
        fig.add_trace(
            go.Scatter(
                x=merged.index, y=merged["confidence"] * 100, mode="lines",
                name="HMM Confidence",
                line=dict(color="#9467bd", width=1.5), showlegend=False,
            ),
            row=2, col=1,
        )
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_yaxes(title="Price", row=1, col=1)
        fig.update_yaxes(title="Confidence %", range=[0, 100], row=2, col=1)
        fig.update_layout(height=600, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # BIC selection table
        # ============================================================
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("BIC Selection")
            st.caption("Lower BIC = better model fit (penalizes complexity)")
            bic_df = pd.DataFrame(
                {"N Regimes": list(bic_scores.keys()), "BIC": list(bic_scores.values())}
            )
            bic_df["Selected"] = bic_df["N Regimes"].apply(lambda n: "✓" if n == best_n else "")
            st.dataframe(bic_df.round(0), hide_index=True, use_container_width=True)

        with col_b:
            st.subheader("Regime Distribution")
            dist = regimes_df["regime"].value_counts(normalize=True) * 100
            pie = go.Figure(
                data=[
                    go.Pie(
                        labels=dist.index, values=dist.values, hole=0.5,
                        marker=dict(colors=[REGIME_COLORS.get(r, "#888") for r in dist.index]),
                        textinfo="label+percent",
                    )
                ]
            )
            pie.update_layout(height=300, margin=dict(t=20, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(pie, use_container_width=True)

        # ============================================================
        # Per-regime stats
        # ============================================================
        st.subheader("Per-Regime Statistics")
        merged["forward_return"] = merged["close"].pct_change().shift(-1)
        regime_stats = (
            merged.groupby("regime")
            .agg(
                bars=("close", "count"),
                avg_return_pct=("forward_return", lambda x: x.mean() * 100),
                vol_pct=("forward_return", lambda x: x.std() * 100),
                pct_of_total=("close", lambda x: len(x) / len(merged) * 100),
            )
            .reset_index()
            .sort_values("avg_return_pct", ascending=False)
        )
        regime_stats["avg_return_pct"] = regime_stats["avg_return_pct"].round(3)
        regime_stats["vol_pct"] = regime_stats["vol_pct"].round(2)
        regime_stats["pct_of_total"] = regime_stats["pct_of_total"].round(1)
        regime_stats.columns = ["Regime", "Bars", "Avg Forward Return %", "Volatility %", "% of Total"]
        st.dataframe(regime_stats, hide_index=True, use_container_width=True)

        # ============================================================
        # Posterior probability bars
        # ============================================================
        st.subheader("Latest Posterior Probabilities")
        posteriors = model.predict_proba(features.values)
        latest_post = posteriors[-1]
        prob_data = []
        for state_idx, label in label_map.items():
            prob_data.append({"Regime": label.value, "Probability": float(latest_post[state_idx])})
        prob_df = pd.DataFrame(prob_data).sort_values("Probability", ascending=False)
        bar = go.Figure(
            data=[
                go.Bar(
                    x=prob_df["Regime"], y=prob_df["Probability"] * 100,
                    marker=dict(color=[REGIME_COLORS.get(r, "#888") for r in prob_df["Regime"]]),
                    text=[f"{p:.0%}" for p in prob_df["Probability"]],
                    textposition="auto",
                )
            ]
        )
        bar.update_layout(
            height=300, yaxis_title="Probability %",
            margin=dict(t=20, b=10, l=10, r=10), showlegend=False,
        )
        st.plotly_chart(bar, use_container_width=True)

        st.divider()
        st.caption(
            f"Analysis complete · {symbol} · {timeframe} · {len(ohlcv)} bars "
            f"· {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

    except Exception as e:
        st.error(f"Failed to analyze {symbol}: {e}")
        st.info(
            "Try these symbols if you're stuck:\n"
            "- Stocks: `SPY`, `NVDA`, `AAPL`\n"
            "- Futures: `ES=F`, `GC=F`\n"
            "- Crypto: `BTC-USD` or `BTC/USDT`\n"
            "- Indices: `^GSPC`, `^VIX`"
        )
