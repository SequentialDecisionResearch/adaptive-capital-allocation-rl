# -*- coding: utf-8 -*-
"""
Stocks as Time-Varying Investment States
Long-Horizon Incremental Allocation RL / Contextual-Bandit Demo
================================================================

Purpose
-------
This single-file research demo accompanies the article:

    "Reinforcement Learning & Investing: Stocks as Time-Varying Investment States"

The central question is not "Which stock is permanently best?" but:

    "Where is the next dollar of capital most useful, given the current
     investment states and the current portfolio state?"

The code intentionally stays interpretable and research-oriented.  It is NOT a
production trading system and is NOT financial advice.

Main ideas implemented
----------------------
1. Yahoo Finance daily adjusted prices for ~30 liquid US-listed stocks.
2. Stocks are represented by time-varying state vectors, not permanent ticker IDs.
3. A pooled empirical-Bayesian linear model learns across all stock-time states.
4. Thompson Sampling provides controlled exploration through posterior uncertainty.
5. A pooled cross-ticker nearest-neighbour state estimator adds the article's
   "same state can matter more than same ticker" idea.
6. Portfolio-aware marginal utility converts opportunity scores into a
   "next-dollar" decision using concentration, sector, correlation, volatility,
   drawdown, and uncertainty penalties.
7. Monthly new capital is split into micro-buys.  After every micro-buy, portfolio
   weights are recomputed before allocating the next small capital unit.
8. CASH is always an action.
9. Strict walk-forward timing:
      - signal uses the PREVIOUS trading day's close,
      - trade executes on the next month's first trading day,
      - model training only uses forward-return labels that are already mature.
10. Baselines use identical cash flows: SPY DCA, QQQ DCA, equal-weight DCA,
    and deterministic (non-Thompson) dynamic-state allocation.
11. Outputs include backtest summary, daily NAV, time-weighted drawdowns,
    decisions, trades, weights, model diagnostics, nearest historical states,
    and publication/GitHub-ready figures.

Required packages
-----------------
    pip install yfinance pandas numpy matplotlib

Designed for
------------
    Spyder / Anaconda / standard Python 3.10+

Default Windows output directory
--------------------------------
    C:\\study_notes\\traval_rec\\Stocks_as_Time_Varying_Investment

Important research limitations
------------------------------
* This demo uses price/market states only.  Point-in-time historical fundamentals
  are deliberately NOT pulled from current Yahoo fundamentals because doing so
  would create historical-data leakage.  A research-grade extension should add
  point-in-time fundamentals from a proper database.
* Yahoo Finance is convenient for education/research, but is not an institutional
  point-in-time market-data feed.
* No claim is made that the proposed method will outperform the benchmarks.
  The purpose is to demonstrate a defensible sequential decision framework.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. CONFIGURATION -- edit these values in Spyder if desired
# =============================================================================

OUTPUT_DIR = Path(r"C:\study_notes\traval_rec\Stocks_as_Time_Varying_Investment")

# About 30 stocks: large platforms, semis, networking/security, data-centre power,
# and a few defensive/health/financial names.  Late IPOs simply enter the dynamic
# universe after enough price history becomes available.
UNIVERSE: List[str] = [
    "NVDA", "MSFT", "GOOG", "AMZN", "META", "AAPL",
    "AVGO", "AMD", "TSM", "ASML", "ANET", "MRVL", "MU", "LRCX", "KLAC", "AMAT",
    "CRWD", "PANW", "ORCL",
    "VRT", "ETN", "GEV", "APH",
    "LLY", "ISRG", "COST", "WMT", "MA", "JPM", "CAT",
]

BENCHMARKS = ["SPY", "QQQ"]

SECTOR_MAP: Dict[str, str] = {
    "NVDA": "ai_semis", "AVGO": "ai_semis", "AMD": "ai_semis", "TSM": "ai_semis",
    "ASML": "ai_semis", "MRVL": "ai_semis", "MU": "ai_semis", "LRCX": "ai_semis",
    "KLAC": "ai_semis", "AMAT": "ai_semis",
    "MSFT": "platforms", "GOOG": "platforms", "AMZN": "platforms", "META": "platforms",
    "AAPL": "platforms", "ORCL": "platforms",
    "ANET": "network_security", "CRWD": "network_security", "PANW": "network_security",
    "APH": "network_security",
    "VRT": "power_infrastructure", "ETN": "power_infrastructure", "GEV": "power_infrastructure",
    "LLY": "defensive_health", "ISRG": "defensive_health",
    "COST": "defensive_consumer", "WMT": "defensive_consumer",
    "MA": "financials", "JPM": "financials",
    "CAT": "industrial",
}

DOWNLOAD_START = "2012-01-01"
BACKTEST_START = "2019-01-01"
DOWNLOAD_END: Optional[str] = None  # None = latest available Yahoo data
FORCE_DOWNLOAD = False              # False reuses cached CSV if it exists

INITIAL_CAPITAL = 20_000.0
MONTHLY_CONTRIBUTION = 2_000.0
TRANSACTION_COST_RATE = 0.0005      # 5 bps all-in illustrative spread/slippage/fee

# State / target construction
FORWARD_HORIZON_DAYS = 63           # approximately 3 trading months
MIN_HISTORY_DAYS = 252              # candidate needs ~1 year of history
TRAIN_LOOKBACK_YEARS = 8
MIN_TRAIN_ROWS = 400
TARGET_CLIP = 1.0                   # clip 3m log-return labels for robustness

# Empirical-Bayesian linear model
PRIOR_VARIANCE = 1.0
RIDGE_FOR_NOISE_ESTIMATE = 1e-3
POSTERIOR_JITTER = 1e-10

# Pooled cross-ticker state matching
KNN_BLEND_WEIGHT = 0.25             # 0 = pure Bayesian linear model
KNN_NEIGHBORS = 30

# Incremental / next-dollar allocation
MICRO_BUYS = 20
IDLE_CASH_REDEPLOY_FRACTION = 0.25  # each month may reconsider some parked cash
MAX_SINGLE_STOCK_WEIGHT = 0.12
MAX_SECTOR_WEIGHT = 0.35
CASH_SCORE = 0.0

# Marginal-utility penalties.  Q is a ~3-month log-return estimate.
LAMBDA_VOL = 0.04
LAMBDA_DRAWDOWN = 0.05
LAMBDA_WEIGHT = 0.18
LAMBDA_SECTOR = 0.04
LAMBDA_CORRELATION = 0.008
LAMBDA_UNCERTAINTY = 0.10
CORRELATION_LOOKBACK_DAYS = 63

RANDOM_SEED = 42
NEIGHBOR_REPORT_K = 12
TOP_HOLDINGS_TO_PLOT = 8
TOP_HEATMAP_TICKERS = 12

# If True, all figures remain open at the end so Spyder displays them.
SHOW_FIGURES = True


# =============================================================================
# 2. GLOBAL FEATURE DEFINITION
# =============================================================================

FEATURE_COLUMNS = [
    "ret_21",
    "ret_63",
    "ret_126",
    "ret_252",
    "vol_21",
    "vol_63",
    "dd_252",
    "ma50_gap",
    "ma200_gap",
    "rel_spy_63",
    "rel_spy_126",
    "market_spy_63",
    "market_spy_252",
    "market_vol_63",
    "market_dd_252",
    "qqq_minus_spy_63",
]


# =============================================================================
# 3. SMALL UTILITIES
# =============================================================================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]
    return out


def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def last_trading_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index=index, data=index)
    vals = s.groupby(index.to_period("M")).max().values
    return pd.DatetimeIndex(vals)


def first_trading_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index=index, data=index)
    vals = s.groupby(index.to_period("M")).min().values
    return pd.DatetimeIndex(vals)


def annualized_volatility(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(252.0))


def time_weighted_returns(nav: pd.Series, flows: pd.Series, initial_capital: float) -> pd.Series:
    """Daily TWR removing external contributions from investment return."""
    nav = nav.astype(float).copy()
    flows = flows.reindex(nav.index).fillna(0.0).astype(float)
    r = pd.Series(index=nav.index, dtype=float)
    if len(nav) == 0:
        return r

    # First close relative to initial capital: captures initial execution friction.
    r.iloc[0] = nav.iloc[0] / initial_capital - 1.0
    for i in range(1, len(nav)):
        prev = nav.iloc[i - 1]
        if not np.isfinite(prev) or prev <= 0:
            r.iloc[i] = np.nan
        else:
            r.iloc[i] = (nav.iloc[i] - prev - flows.iloc[i]) / prev
    return r


def performance_stats(nav: pd.Series, flows: pd.Series, initial_capital: float) -> Dict[str, float]:
    r = time_weighted_returns(nav, flows, initial_capital).dropna()
    if len(r) == 0:
        return {
            "Final Wealth": np.nan, "TWR CAGR": np.nan, "Ann Vol": np.nan,
            "Sharpe": np.nan, "Max Drawdown": np.nan, "Calmar": np.nan,
        }

    twr_index = (1.0 + r).cumprod()
    years = max(len(r) / 252.0, 1.0 / 252.0)
    cagr = float(twr_index.iloc[-1] ** (1.0 / years) - 1.0)
    vol = annualized_volatility(r)
    sharpe = np.nan
    if np.isfinite(vol) and vol > 1e-12:
        sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252.0))

    dd = twr_index / twr_index.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = np.nan if max_dd >= -1e-12 else float(cagr / abs(max_dd))

    return {
        "Final Wealth": float(nav.iloc[-1]),
        "TWR CAGR": cagr,
        "Ann Vol": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
    }


def fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{100.0 * x:,.2f}%"


# =============================================================================
# 4. YAHOO DATA DOWNLOAD / CACHE
# =============================================================================

def download_adjusted_close(symbols: Iterable[str]) -> pd.DataFrame:
    """Download adjusted daily Close via current yfinance API."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Missing package 'yfinance'. In Spyder/Anaconda run:\n"
            "    pip install yfinance\n"
            "Then restart the Python kernel and run this file again."
        ) from exc

    symbols = list(dict.fromkeys([str(s).upper() for s in symbols]))
    print(f"Downloading Yahoo adjusted prices for {len(symbols)} symbols ...")

    raw = yf.download(
        tickers=symbols,
        start=DOWNLOAD_START,
        end=DOWNLOAD_END,
        interval="1d",
        auto_adjust=True,
        repair=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
        multi_level_index=True,
    )

    if raw is None or len(raw) == 0:
        raise RuntimeError("Yahoo/yfinance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(map(str, raw.columns.get_level_values(0)))
        lvl1 = set(map(str, raw.columns.get_level_values(1)))
        if "Close" in lvl0:
            close = raw["Close"].copy()
        elif "Close" in lvl1:
            close = raw.xs("Close", axis=1, level=1).copy()
        else:
            raise RuntimeError("Could not locate Close prices in yfinance MultiIndex columns.")
    else:
        if "Close" not in raw.columns:
            raise RuntimeError("Could not locate Close prices in yfinance output.")
        close = raw[["Close"]].copy()
        if len(symbols) == 1:
            close.columns = symbols

    close = clean_columns(close)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    close = close.loc[~close.index.duplicated(keep="last")]

    # Forward-fill only AFTER a symbol has begun trading; no backward fill.
    close = close.ffill(limit=5)
    return close


def load_or_download_prices() -> pd.DataFrame:
    ensure_output_dir()
    cache_file = OUTPUT_DIR / "yahoo_adjusted_close.csv"

    if cache_file.exists() and not FORCE_DOWNLOAD:
        print(f"Using cached Yahoo data: {cache_file}")
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        prices = clean_columns(prices)
        prices.index = pd.to_datetime(prices.index)
    else:
        prices = download_adjusted_close(UNIVERSE + BENCHMARKS)
        prices.to_csv(cache_file)
        print(f"Saved Yahoo cache: {cache_file}")

    return prices


# =============================================================================
# 5. TIME-VARYING INVESTMENT STATES
# =============================================================================

def build_feature_panel(
    prices: pd.DataFrame,
    universe: List[str],
) -> pd.DataFrame:
    """
    Build one pooled row for each (signal_date, ticker).

    IMPORTANT: Ticker ID is metadata, not a model feature.  The predictive model
    therefore learns across stock-time investment states rather than maintaining
    a separate parameter vector for each ticker.
    """
    px = prices.copy()
    stock_px = px[universe]
    daily_ret = stock_px.pct_change(fill_method=None)

    ret_21 = stock_px / stock_px.shift(21) - 1.0
    ret_63 = stock_px / stock_px.shift(63) - 1.0
    ret_126 = stock_px / stock_px.shift(126) - 1.0
    ret_252 = stock_px / stock_px.shift(252) - 1.0
    vol_21 = daily_ret.rolling(21, min_periods=15).std() * np.sqrt(252.0)
    vol_63 = daily_ret.rolling(63, min_periods=40).std() * np.sqrt(252.0)
    dd_252 = stock_px / stock_px.rolling(252, min_periods=126).max() - 1.0
    ma50_gap = stock_px / stock_px.rolling(50, min_periods=40).mean() - 1.0
    ma200_gap = stock_px / stock_px.rolling(200, min_periods=150).mean() - 1.0

    spy = px["SPY"]
    qqq = px["QQQ"]
    spy_ret = spy.pct_change(fill_method=None)
    spy_63 = spy / spy.shift(63) - 1.0
    spy_126 = spy / spy.shift(126) - 1.0
    spy_252 = spy / spy.shift(252) - 1.0
    spy_vol_63 = spy_ret.rolling(63, min_periods=40).std() * np.sqrt(252.0)
    spy_dd_252 = spy / spy.rolling(252, min_periods=126).max() - 1.0
    qqq_63 = qqq / qqq.shift(63) - 1.0

    rel_spy_63 = ret_63.sub(spy_63, axis=0)
    rel_spy_126 = ret_126.sub(spy_126, axis=0)

    # One target-maturity date for each daily signal date.
    idx_series = pd.Series(prices.index, index=prices.index)
    target_available_date = idx_series.shift(-FORWARD_HORIZON_DAYS)

    signal_dates = last_trading_days(prices.index)
    rows = []

    for ticker in universe:
        future_log_return = np.log(stock_px[ticker].shift(-FORWARD_HORIZON_DAYS) / stock_px[ticker])

        df = pd.DataFrame(index=prices.index)
        df["ret_21"] = ret_21[ticker]
        df["ret_63"] = ret_63[ticker]
        df["ret_126"] = ret_126[ticker]
        df["ret_252"] = ret_252[ticker]
        df["vol_21"] = vol_21[ticker]
        df["vol_63"] = vol_63[ticker]
        df["dd_252"] = dd_252[ticker]
        df["ma50_gap"] = ma50_gap[ticker]
        df["ma200_gap"] = ma200_gap[ticker]
        df["rel_spy_63"] = rel_spy_63[ticker]
        df["rel_spy_126"] = rel_spy_126[ticker]
        df["market_spy_63"] = spy_63
        df["market_spy_252"] = spy_252
        df["market_vol_63"] = spy_vol_63
        df["market_dd_252"] = spy_dd_252
        df["qqq_minus_spy_63"] = qqq_63 - spy_63
        df["target_log_return"] = future_log_return.clip(-TARGET_CLIP, TARGET_CLIP)
        df["target_available_date"] = target_available_date

        # Use month-end states only, which reduces serial redundancy and keeps the
        # demo fast enough for Spyder while preserving a meaningful pooled history.
        df = df.reindex(signal_dates)
        df = df.dropna(subset=FEATURE_COLUMNS, how="any")
        if df.empty:
            continue

        df = df.reset_index().rename(columns={"index": "date"})
        df["ticker"] = ticker
        rows.append(df)

    if not rows:
        raise RuntimeError("No feature rows could be constructed.")

    panel = pd.concat(rows, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["target_available_date"] = pd.to_datetime(panel["target_available_date"])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel


# =============================================================================
# 6. EMPIRICAL-BAYESIAN LINEAR MODEL + THOMPSON SAMPLING
# =============================================================================

@dataclass
class EmpiricalBayesLinear:
    feature_mean: np.ndarray = field(default_factory=lambda: np.empty(0))
    feature_std: np.ndarray = field(default_factory=lambda: np.empty(0))
    posterior_mean: np.ndarray = field(default_factory=lambda: np.empty(0))
    posterior_cov: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    noise_var: float = np.nan
    train_xz: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    train_y: np.ndarray = field(default_factory=lambda: np.empty(0))

    def _standardize_fit(self, x: np.ndarray) -> np.ndarray:
        self.feature_mean = np.nanmean(x, axis=0)
        self.feature_std = np.nanstd(x, axis=0, ddof=0)
        self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
        return (x - self.feature_mean) / self.feature_std

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feature_mean) / self.feature_std

    @staticmethod
    def design(xz: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(xz)), xz])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "EmpiricalBayesLinear":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        xz = self._standardize_fit(x)
        xd = self.design(xz)

        p = xd.shape[1]
        ridge = np.eye(p)
        ridge[0, 0] = 1e-8  # almost no intercept penalty

        xtx = xd.T @ xd
        xty = xd.T @ y
        beta0 = np.linalg.pinv(xtx + RIDGE_FOR_NOISE_ESTIMATE * ridge) @ xty
        resid = y - xd @ beta0
        noise_var = float(np.mean(resid ** 2))
        if not np.isfinite(noise_var) or noise_var < 1e-6:
            noise_var = max(float(np.nanvar(y)) * 0.25, 1e-6)

        prior_prec = np.eye(p) / PRIOR_VARIANCE
        prior_prec[0, 0] = 1e-8

        post_prec = prior_prec + xtx / noise_var
        post_cov = np.linalg.pinv(post_prec)
        post_cov = 0.5 * (post_cov + post_cov.T)
        post_cov += np.eye(p) * POSTERIOR_JITTER
        post_mean = post_cov @ (xty / noise_var)

        self.posterior_mean = post_mean
        self.posterior_cov = post_cov
        self.noise_var = noise_var
        self.train_xz = xz
        self.train_y = y
        return self

    def predict_mean_and_epistemic_std(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xz = self.transform(np.asarray(x, dtype=float))
        xd = self.design(xz)
        mean = xd @ self.posterior_mean
        var = np.einsum("ij,jk,ik->i", xd, self.posterior_cov, xd)
        var = np.maximum(var, 0.0)
        return mean, np.sqrt(var)

    def sample_predict(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Numerical guard: eigenvalue clipping gives a PSD covariance for sampling.
        cov = 0.5 * (self.posterior_cov + self.posterior_cov.T)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 0.0, None)
        cov_psd = (vecs * vals) @ vecs.T
        theta = rng.multivariate_normal(self.posterior_mean, cov_psd + np.eye(len(vals)) * 1e-12)
        xz = self.transform(np.asarray(x, dtype=float))
        return self.design(xz) @ theta


# =============================================================================
# 7. POOLED CROSS-TICKER STATE MATCHING
# =============================================================================

def pooled_knn_estimate(
    model: EmpiricalBayesLinear,
    x_new: np.ndarray,
    k: int = KNN_NEIGHBORS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate expected future return from the nearest historical investment states
    across ALL tickers.  This is intentionally not restricted to ticker identity.
    """
    xz_new = model.transform(np.asarray(x_new, dtype=float))
    train_xz = model.train_xz
    y = model.train_y

    means = np.zeros(len(xz_new), dtype=float)
    ses = np.zeros(len(xz_new), dtype=float)

    if len(train_xz) == 0:
        return means, ses

    kk = min(k, len(train_xz))
    for i, z in enumerate(xz_new):
        dist = np.sqrt(np.sum((train_xz - z) ** 2, axis=1))
        ix = np.argpartition(dist, kk - 1)[:kk]
        d = dist[ix]
        yy = y[ix]
        w = 1.0 / np.maximum(d, 1e-6)
        w = w / w.sum()
        mu = float(np.sum(w * yy))
        var = float(np.sum(w * (yy - mu) ** 2))
        neff = float(1.0 / np.sum(w ** 2))
        means[i] = mu
        ses[i] = math.sqrt(max(var, 0.0) / max(neff, 1.0))

    return means, ses


def nearest_state_report(
    model: EmpiricalBayesLinear,
    train_df: pd.DataFrame,
    current_row: pd.Series,
    k: int = NEIGHBOR_REPORT_K,
) -> pd.DataFrame:
    z = model.transform(current_row[FEATURE_COLUMNS].to_numpy(dtype=float).reshape(1, -1))[0]
    dist = np.sqrt(np.sum((model.train_xz - z) ** 2, axis=1))
    kk = min(k, len(dist))
    ix = np.argsort(dist)[:kk]

    report = train_df.iloc[ix][["date", "ticker", "target_log_return"]].copy().reset_index(drop=True)
    report["distance"] = dist[ix]
    report = report[["date", "ticker", "distance", "target_log_return"]]
    return report


# =============================================================================
# 8. SIMPLE PORTFOLIO OBJECT
# =============================================================================

@dataclass
class Portfolio:
    name: str
    cash: float
    shares: Dict[str, float] = field(default_factory=dict)
    total_external_contributions: float = 0.0

    def add_cash(self, amount: float) -> None:
        if amount > 0:
            self.cash += float(amount)
            self.total_external_contributions += float(amount)

    def position_value(self, ticker: str, price: float) -> float:
        return float(self.shares.get(ticker, 0.0) * price)

    def nav(self, price_row: pd.Series) -> float:
        total = float(self.cash)
        for t, sh in self.shares.items():
            p = safe_float(price_row.get(t, np.nan))
            if np.isfinite(p):
                total += sh * p
        return float(total)

    def weights(self, price_row: pd.Series) -> Dict[str, float]:
        nav = self.nav(price_row)
        if nav <= 0:
            return {}
        out = {}
        for t, sh in self.shares.items():
            p = safe_float(price_row.get(t, np.nan))
            if np.isfinite(p) and sh > 0:
                out[t] = float(sh * p / nav)
        return out

    def buy(self, ticker: str, cash_amount: float, price: float) -> Tuple[float, float]:
        """Spend cash_amount INCLUDING transaction friction. Returns (shares, spend)."""
        if cash_amount <= 0 or self.cash <= 0 or not np.isfinite(price) or price <= 0:
            return 0.0, 0.0
        spend = float(min(cash_amount, self.cash))
        effective_price = price * (1.0 + TRANSACTION_COST_RATE)
        sh = spend / effective_price
        self.shares[ticker] = self.shares.get(ticker, 0.0) + sh
        self.cash -= spend
        return float(sh), spend


# =============================================================================
# 9. PORTFOLIO-AWARE MARGINAL UTILITY / NEXT-DOLLAR POLICY
# =============================================================================

def sector_weights(portfolio: Portfolio, price_row: pd.Series) -> Dict[str, float]:
    w = portfolio.weights(price_row)
    out: Dict[str, float] = {}
    for t, wt in w.items():
        sector = SECTOR_MAP.get(t, "other")
        out[sector] = out.get(sector, 0.0) + wt
    return out


def candidate_correlation_to_portfolio(
    ticker: str,
    signal_date: pd.Timestamp,
    portfolio: Portfolio,
    price_row: pd.Series,
    daily_returns: pd.DataFrame,
) -> float:
    w = portfolio.weights(price_row)
    held = [t for t, wt in w.items() if wt > 1e-6 and t in daily_returns.columns]
    if not held:
        return 0.0

    hist = daily_returns.loc[:signal_date].tail(CORRELATION_LOOKBACK_DAYS)
    if ticker not in hist.columns or len(hist) < 20:
        return 0.0

    held_w = np.array([w[t] for t in held], dtype=float)
    if held_w.sum() <= 0:
        return 0.0
    held_w = held_w / held_w.sum()
    port_r = hist[held].fillna(0.0).to_numpy() @ held_w
    cand = hist[ticker].to_numpy(dtype=float)
    mask = np.isfinite(cand) & np.isfinite(port_r)
    if mask.sum() < 20:
        return 0.0
    c = np.corrcoef(cand[mask], port_r[mask])[0, 1]
    return 0.0 if not np.isfinite(c) else float(c)


def projected_caps_ok(
    ticker: str,
    spend: float,
    portfolio: Portfolio,
    price_row: pd.Series,
) -> bool:
    nav = portfolio.nav(price_row)
    if nav <= 0:
        return False

    w = portfolio.weights(price_row)
    current_w = w.get(ticker, 0.0)
    projected_w = current_w + spend / nav
    if projected_w > MAX_SINGLE_STOCK_WEIGHT + 1e-12:
        return False

    sw = sector_weights(portfolio, price_row)
    sec = SECTOR_MAP.get(ticker, "other")
    projected_sector = sw.get(sec, 0.0) + spend / nav
    if projected_sector > MAX_SECTOR_WEIGHT + 1e-12:
        return False

    return True


def marginal_utility(
    score: float,
    q_unc: float,
    state_row: pd.Series,
    ticker: str,
    portfolio: Portfolio,
    execution_prices: pd.Series,
    signal_date: pd.Timestamp,
    daily_returns: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    w = portfolio.weights(execution_prices)
    current_weight = float(w.get(ticker, 0.0))
    sw = sector_weights(portfolio, execution_prices)
    sec_weight = float(sw.get(SECTOR_MAP.get(ticker, "other"), 0.0))
    corr = candidate_correlation_to_portfolio(
        ticker, signal_date, portfolio, execution_prices, daily_returns
    )

    vol = max(safe_float(state_row["vol_63"], 0.0), 0.0)
    dd = abs(min(safe_float(state_row["dd_252"], 0.0), 0.0))
    positive_corr = max(corr, 0.0)

    penalty = (
        LAMBDA_VOL * vol
        + LAMBDA_DRAWDOWN * dd
        + LAMBDA_WEIGHT * current_weight
        + LAMBDA_SECTOR * sec_weight
        + LAMBDA_CORRELATION * positive_corr
        + LAMBDA_UNCERTAINTY * max(q_unc, 0.0)
    )
    mu = float(score - penalty)

    details = {
        "vol_63": vol,
        "drawdown_252": dd,
        "current_weight": current_weight,
        "sector_weight": sec_weight,
        "corr_to_portfolio": corr,
        "uncertainty": q_unc,
        "total_penalty": penalty,
    }
    return mu, details


def allocate_incrementally(
    portfolio: Portfolio,
    budget: float,
    scores: pd.DataFrame,
    execution_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    execution_prices: pd.Series,
    daily_returns: pd.DataFrame,
    score_column: str,
    strategy_name: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Split a monthly budget into small pieces.  After every buy, recompute portfolio
    penalties before asking where the NEXT capital unit should go.
    """
    if budget <= 1e-10 or portfolio.cash <= 1e-10:
        return [], {"CASH": 0.0}

    budget = float(min(budget, portfolio.cash))
    micro = budget / float(MICRO_BUYS)
    remaining = budget
    trade_log: List[Dict] = []
    allocation: Dict[str, float] = {}

    for step in range(MICRO_BUYS):
        spend = min(micro, remaining, portfolio.cash)
        if spend <= 1e-10:
            break

        best_ticker = None
        best_mu = CASH_SCORE
        best_details = None
        best_row = None

        for _, row in scores.iterrows():
            ticker = row["ticker"]
            price = safe_float(execution_prices.get(ticker, np.nan))
            if not np.isfinite(price) or price <= 0:
                continue
            if not projected_caps_ok(ticker, spend, portfolio, execution_prices):
                continue

            mu, details = marginal_utility(
                score=float(row[score_column]),
                q_unc=float(row["q_unc"]),
                state_row=row,
                ticker=ticker,
                portfolio=portfolio,
                execution_prices=execution_prices,
                signal_date=signal_date,
                daily_returns=daily_returns,
            )
            if mu > best_mu:
                best_mu = mu
                best_ticker = ticker
                best_details = details
                best_row = row

        if best_ticker is None:
            # CASH wins.  Keep this and all remaining unspent budget flexible.
            break

        price = float(execution_prices[best_ticker])
        weight_before = portfolio.weights(execution_prices).get(best_ticker, 0.0)
        shares, actual_spend = portfolio.buy(best_ticker, spend, price)
        weight_after = portfolio.weights(execution_prices).get(best_ticker, 0.0)
        remaining -= actual_spend
        allocation[best_ticker] = allocation.get(best_ticker, 0.0) + actual_spend

        trade_log.append({
            "strategy": strategy_name,
            "execution_date": execution_date,
            "signal_date": signal_date,
            "micro_step": step + 1,
            "ticker": best_ticker,
            "cash_spent": actual_spend,
            "price": price,
            "shares_bought": shares,
            "q_mean": float(best_row["q_mean"]),
            "q_sample": float(best_row["q_sample"]),
            "q_unc": float(best_row["q_unc"]),
            "knn_mean": float(best_row["knn_mean"]),
            "marginal_utility": best_mu,
            "weight_before": weight_before,
            "weight_after": weight_after,
            "sector": SECTOR_MAP.get(best_ticker, "other"),
            "sector_weight_before": float(best_details["sector_weight"]),
            "corr_to_portfolio": float(best_details["corr_to_portfolio"]),
            "vol_63": float(best_details["vol_63"]),
            "drawdown_252": float(best_details["drawdown_252"]),
            "total_penalty": float(best_details["total_penalty"]),
            "cash_after": float(portfolio.cash),
        })

    # The planned budget not spent on assets is explicitly treated as CASH allocation.
    cash_kept = max(remaining, 0.0)
    if cash_kept > 1e-10:
        allocation["CASH"] = allocation.get("CASH", 0.0) + cash_kept

    return trade_log, allocation


# =============================================================================
# 10. WALK-FORWARD MODEL FIT / SCORES
# =============================================================================

def fit_scores_for_signal_date(
    panel: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> Tuple[Optional[pd.DataFrame], Optional[EmpiricalBayesLinear], Optional[pd.DataFrame], Dict]:
    lookback_start = signal_date - pd.DateOffset(years=TRAIN_LOOKBACK_YEARS)

    train = panel[
        (panel["date"] >= lookback_start)
        & (panel["date"] < signal_date)
        & (panel["target_available_date"].notna())
        & (panel["target_available_date"] <= signal_date)
        & (panel["target_log_return"].notna())
    ].copy()

    current = panel[panel["date"] == signal_date].copy()
    current = current.dropna(subset=FEATURE_COLUMNS)

    diagnostics = {
        "signal_date": signal_date,
        "train_rows": len(train),
        "candidate_rows": len(current),
        "status": "ok",
    }

    if len(train) < MIN_TRAIN_ROWS or current.empty:
        diagnostics["status"] = "insufficient_data"
        return None, None, train, diagnostics

    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["target_log_return"].to_numpy(dtype=float)
    x_cur = current[FEATURE_COLUMNS].to_numpy(dtype=float)

    model = EmpiricalBayesLinear().fit(x_train, y_train)
    bayes_mean, bayes_unc = model.predict_mean_and_epistemic_std(x_cur)

    # One common posterior draw per decision date: true Thompson-style ranking.
    date_seed = RANDOM_SEED + int(signal_date.strftime("%Y%m%d")) % 1_000_000
    rng = np.random.default_rng(date_seed)
    bayes_sample = model.sample_predict(x_cur, rng)

    knn_mean, knn_se = pooled_knn_estimate(model, x_cur, KNN_NEIGHBORS)

    w = float(np.clip(KNN_BLEND_WEIGHT, 0.0, 1.0))
    q_mean = (1.0 - w) * bayes_mean + w * knn_mean
    q_sample = (1.0 - w) * bayes_sample + w * knn_mean
    q_unc = (1.0 - w) * bayes_unc + w * knn_se

    out = current[["date", "ticker"] + FEATURE_COLUMNS].copy().reset_index(drop=True)
    out["bayes_mean"] = bayes_mean
    out["bayes_unc"] = bayes_unc
    out["knn_mean"] = knn_mean
    out["knn_se"] = knn_se
    out["q_mean"] = q_mean
    out["q_sample"] = q_sample
    out["q_unc"] = q_unc

    diagnostics.update({
        "noise_std": math.sqrt(model.noise_var),
        "mean_q": float(np.mean(q_mean)),
        "mean_uncertainty": float(np.mean(q_unc)),
        "top_q_ticker": str(out.loc[out["q_mean"].idxmax(), "ticker"]),
        "top_q_mean": float(out["q_mean"].max()),
    })

    return out, model, train.reset_index(drop=True), diagnostics


# =============================================================================
# 11. BACKTEST ENGINE
# =============================================================================

def build_execution_schedule(index: pd.DatetimeIndex) -> Dict[pd.Timestamp, pd.Timestamp]:
    """execution_date -> previous trading day signal_date."""
    first_days = first_trading_days(index)
    index_set = set(index)
    schedule = {}

    for exec_date in first_days:
        if exec_date < pd.Timestamp(BACKTEST_START):
            continue
        pos = index.get_indexer([exec_date])[0]
        if pos <= 0:
            continue
        signal_date = index[pos - 1]
        if signal_date not in index_set:
            continue
        schedule[pd.Timestamp(exec_date)] = pd.Timestamp(signal_date)

    return schedule


def available_universe(prices: pd.DataFrame) -> List[str]:
    valid = []
    for t in UNIVERSE:
        if t not in prices.columns:
            continue
        if prices[t].notna().sum() >= MIN_HISTORY_DAYS:
            valid.append(t)
    return valid


def equal_weight_allocate(
    portfolio: Portfolio,
    budget: float,
    tickers: List[str],
    execution_prices: pd.Series,
) -> Dict[str, float]:
    candidates = [t for t in tickers if np.isfinite(safe_float(execution_prices.get(t, np.nan)))]
    if budget <= 0 or not candidates:
        return {"CASH": max(budget, 0.0)}
    per = min(budget, portfolio.cash) / len(candidates)
    out = {}
    for t in candidates:
        _, spend = portfolio.buy(t, per, float(execution_prices[t]))
        out[t] = out.get(t, 0.0) + spend
    return out


def one_asset_allocate(
    portfolio: Portfolio,
    budget: float,
    ticker: str,
    execution_prices: pd.Series,
) -> Dict[str, float]:
    p = safe_float(execution_prices.get(ticker, np.nan))
    if not np.isfinite(p) or budget <= 0:
        return {"CASH": max(budget, 0.0)}
    _, spend = portfolio.buy(ticker, min(budget, portfolio.cash), p)
    return {ticker: spend}


def record_monthly_weights(
    execution_date: pd.Timestamp,
    portfolios: Dict[str, Portfolio],
    price_row: pd.Series,
) -> List[Dict]:
    rows = []
    for name, p in portfolios.items():
        nav = p.nav(price_row)
        w = p.weights(price_row)
        row = {"date": execution_date, "strategy": name, "NAV": nav}
        for t in UNIVERSE + BENCHMARKS:
            row[t] = w.get(t, 0.0)
        row["CASH"] = 0.0 if nav <= 0 else p.cash / nav
        rows.append(row)
    return rows


def run_backtest(
    prices: pd.DataFrame,
    panel: pd.DataFrame,
    universe: List[str],
) -> Dict[str, pd.DataFrame]:
    all_needed = universe + BENCHMARKS
    px = prices[all_needed].copy()
    px = px.loc[px.index >= pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=10)].copy()
    if px.empty:
        raise RuntimeError("No price data in the requested backtest period.")

    full_daily_returns = prices[universe].pct_change(fill_method=None)
    schedule = build_execution_schedule(px.index)
    execution_dates = sorted(schedule.keys())
    if not execution_dates:
        raise RuntimeError("No monthly execution dates could be constructed.")

    start_date = execution_dates[0]
    px_bt = px.loc[start_date:].copy()

    portfolios: Dict[str, Portfolio] = {
        "Proposed_TS": Portfolio("Proposed_TS", INITIAL_CAPITAL),
        "Greedy_Dynamic": Portfolio("Greedy_Dynamic", INITIAL_CAPITAL),
        "EqualWeight_DCA": Portfolio("EqualWeight_DCA", INITIAL_CAPITAL),
        "SPY_DCA": Portfolio("SPY_DCA", INITIAL_CAPITAL),
        "QQQ_DCA": Portfolio("QQQ_DCA", INITIAL_CAPITAL),
    }

    nav_rows: List[Dict] = []
    flow_rows: List[Dict] = []
    trade_rows: List[Dict] = []
    allocation_rows: List[Dict] = []
    weight_rows: List[Dict] = []
    score_rows: List[Dict] = []
    model_rows: List[Dict] = []

    first_execution = True
    last_model = None
    last_train_df = None
    last_scores = None
    last_signal_date = None

    for date, price_row in px_bt.iterrows():
        monthly_flow = 0.0

        if date in schedule:
            signal_date = schedule[date]
            scores, model, train_df, diagnostics = fit_scores_for_signal_date(panel, signal_date)
            model_rows.append(diagnostics)

            if first_execution:
                contribution = 0.0
            else:
                contribution = MONTHLY_CONTRIBUTION
                monthly_flow = contribution
                for p in portfolios.values():
                    p.add_cash(contribution)

            # Budget design: initial portfolio cash is evaluated immediately; later,
            # each decision can deploy the new contribution plus a fraction of
            # previously parked cash, preserving CASH as future flexibility.
            if first_execution:
                proposed_budget = portfolios["Proposed_TS"].cash
                greedy_budget = portfolios["Greedy_Dynamic"].cash
                ew_budget = portfolios["EqualWeight_DCA"].cash
                spy_budget = portfolios["SPY_DCA"].cash
                qqq_budget = portfolios["QQQ_DCA"].cash
            else:
                proposed_idle = max(portfolios["Proposed_TS"].cash - contribution, 0.0)
                greedy_idle = max(portfolios["Greedy_Dynamic"].cash - contribution, 0.0)
                proposed_budget = min(
                    portfolios["Proposed_TS"].cash,
                    contribution + IDLE_CASH_REDEPLOY_FRACTION * proposed_idle,
                )
                greedy_budget = min(
                    portfolios["Greedy_Dynamic"].cash,
                    contribution + IDLE_CASH_REDEPLOY_FRACTION * greedy_idle,
                )
                ew_budget = min(portfolios["EqualWeight_DCA"].cash, contribution)
                spy_budget = min(portfolios["SPY_DCA"].cash, contribution)
                qqq_budget = min(portfolios["QQQ_DCA"].cash, contribution)

            # Equal-weight baseline uses the SAME currently observable universe as
            # the dynamic models, so a fresh IPO is not bought before enough state
            # history exists.  SPY/QQQ remain simple one-asset DCA benchmarks.
            eligible_now = scores["ticker"].tolist() if scores is not None else []
            ew_alloc = equal_weight_allocate(
                portfolios["EqualWeight_DCA"], ew_budget, eligible_now, price_row
            )
            spy_alloc = one_asset_allocate(portfolios["SPY_DCA"], spy_budget, "SPY", price_row)
            qqq_alloc = one_asset_allocate(portfolios["QQQ_DCA"], qqq_budget, "QQQ", price_row)

            for strat, alloc in [
                ("EqualWeight_DCA", ew_alloc), ("SPY_DCA", spy_alloc), ("QQQ_DCA", qqq_alloc)
            ]:
                for t, amt in alloc.items():
                    allocation_rows.append({
                        "execution_date": date, "signal_date": signal_date,
                        "strategy": strat, "ticker": t, "allocated_cash": amt,
                    })

            if scores is not None and model is not None:
                # Record model scores before portfolio penalties change through micro-buys.
                proposed_portfolio = portfolios["Proposed_TS"]
                greedy_portfolio = portfolios["Greedy_Dynamic"]
                for _, row in scores.iterrows():
                    t = row["ticker"]
                    if t not in universe:
                        continue
                    mu_ts, _ = marginal_utility(
                        float(row["q_sample"]), float(row["q_unc"]), row, t,
                        proposed_portfolio, price_row, signal_date, full_daily_returns,
                    )
                    mu_g, _ = marginal_utility(
                        float(row["q_mean"]), float(row["q_unc"]), row, t,
                        greedy_portfolio, price_row, signal_date, full_daily_returns,
                    )
                    score_rows.append({
                        "execution_date": date,
                        "signal_date": signal_date,
                        "ticker": t,
                        "q_mean": float(row["q_mean"]),
                        "q_sample": float(row["q_sample"]),
                        "q_unc": float(row["q_unc"]),
                        "bayes_mean": float(row["bayes_mean"]),
                        "bayes_unc": float(row["bayes_unc"]),
                        "knn_mean": float(row["knn_mean"]),
                        "knn_se": float(row["knn_se"]),
                        "mu_proposed_initial": mu_ts,
                        "mu_greedy_initial": mu_g,
                    })

                ts_trades, ts_alloc = allocate_incrementally(
                    portfolio=portfolios["Proposed_TS"],
                    budget=proposed_budget,
                    scores=scores,
                    execution_date=date,
                    signal_date=signal_date,
                    execution_prices=price_row,
                    daily_returns=full_daily_returns,
                    score_column="q_sample",
                    strategy_name="Proposed_TS",
                )
                g_trades, g_alloc = allocate_incrementally(
                    portfolio=portfolios["Greedy_Dynamic"],
                    budget=greedy_budget,
                    scores=scores,
                    execution_date=date,
                    signal_date=signal_date,
                    execution_prices=price_row,
                    daily_returns=full_daily_returns,
                    score_column="q_mean",
                    strategy_name="Greedy_Dynamic",
                )
                trade_rows.extend(ts_trades)
                trade_rows.extend(g_trades)

                for strat, alloc in [("Proposed_TS", ts_alloc), ("Greedy_Dynamic", g_alloc)]:
                    for t, amt in alloc.items():
                        allocation_rows.append({
                            "execution_date": date, "signal_date": signal_date,
                            "strategy": strat, "ticker": t, "allocated_cash": amt,
                        })

                last_model = model
                last_train_df = train_df
                last_scores = scores
                last_signal_date = signal_date
            else:
                # If the model is not yet estimable, leave dynamic-strategy cash unspent.
                for strat, budget in [("Proposed_TS", proposed_budget), ("Greedy_Dynamic", greedy_budget)]:
                    allocation_rows.append({
                        "execution_date": date, "signal_date": signal_date,
                        "strategy": strat, "ticker": "CASH", "allocated_cash": budget,
                    })

            weight_rows.extend(record_monthly_weights(date, portfolios, price_row))
            first_execution = False

        nav_row = {"date": date}
        flow_row = {"date": date}
        for name, p in portfolios.items():
            nav_row[name] = p.nav(price_row)
            flow_row[name] = monthly_flow
        nav_rows.append(nav_row)
        flow_rows.append(flow_row)

    nav = pd.DataFrame(nav_rows).set_index("date")
    flows = pd.DataFrame(flow_rows).set_index("date")
    trades = pd.DataFrame(trade_rows)
    allocations = pd.DataFrame(allocation_rows)
    weights = pd.DataFrame(weight_rows)
    scores_df = pd.DataFrame(score_rows)
    model_diag = pd.DataFrame(model_rows)

    # Time-weighted daily returns and drawdowns for apples-to-apples risk comparison.
    twr_ret = pd.DataFrame(index=nav.index)
    twr_idx = pd.DataFrame(index=nav.index)
    drawdown = pd.DataFrame(index=nav.index)
    summary_rows = []

    total_contributions = INITIAL_CAPITAL + MONTHLY_CONTRIBUTION * max(len(execution_dates) - 1, 0)

    for name in portfolios:
        r = time_weighted_returns(nav[name], flows[name], INITIAL_CAPITAL)
        idx = (1.0 + r.fillna(0.0)).cumprod()
        dd = idx / idx.cummax() - 1.0
        twr_ret[name] = r
        twr_idx[name] = idx
        drawdown[name] = dd

        stats = performance_stats(nav[name], flows[name], INITIAL_CAPITAL)
        stats.update({
            "Strategy": name,
            "Initial Capital": INITIAL_CAPITAL,
            "Monthly Contribution": MONTHLY_CONTRIBUTION,
            "Total Contributions": total_contributions,
            "Wealth Minus Contributions": stats["Final Wealth"] - total_contributions,
            "Final Cash": portfolios[name].cash,
        })
        summary_rows.append(stats)

    summary = pd.DataFrame(summary_rows).set_index("Strategy")

    # Final nearest-state report for the highest current expected-return candidate.
    neighbors = pd.DataFrame()
    focus_ticker = None
    if last_model is not None and last_train_df is not None and last_scores is not None and not last_scores.empty:
        focus_idx = last_scores["q_mean"].idxmax()
        focus_row = last_scores.loc[focus_idx]
        focus_ticker = str(focus_row["ticker"])
        current_source = panel[(panel["date"] == last_signal_date) & (panel["ticker"] == focus_ticker)]
        if not current_source.empty:
            neighbors = nearest_state_report(
                last_model, last_train_df, current_source.iloc[0], NEIGHBOR_REPORT_K
            )
            neighbors.insert(0, "focus_ticker", focus_ticker)
            neighbors.insert(1, "focus_signal_date", last_signal_date)

    return {
        "nav": nav,
        "flows": flows,
        "twr_returns": twr_ret,
        "twr_index": twr_idx,
        "drawdown": drawdown,
        "summary": summary,
        "trades": trades,
        "allocations": allocations,
        "weights": weights,
        "scores": scores_df,
        "model_diagnostics": model_diag,
        "neighbors": neighbors,
        "focus_ticker": focus_ticker,
    }


# =============================================================================
# 12. OUTPUT FILES / FIGURES
# =============================================================================

def save_config(universe: List[str]) -> None:
    config = {
        "universe": universe,
        "benchmarks": BENCHMARKS,
        "download_start": DOWNLOAD_START,
        "backtest_start": BACKTEST_START,
        "forward_horizon_days": FORWARD_HORIZON_DAYS,
        "initial_capital": INITIAL_CAPITAL,
        "monthly_contribution": MONTHLY_CONTRIBUTION,
        "transaction_cost_rate": TRANSACTION_COST_RATE,
        "train_lookback_years": TRAIN_LOOKBACK_YEARS,
        "knn_blend_weight": KNN_BLEND_WEIGHT,
        "knn_neighbors": KNN_NEIGHBORS,
        "micro_buys": MICRO_BUYS,
        "max_single_stock_weight": MAX_SINGLE_STOCK_WEIGHT,
        "max_sector_weight": MAX_SECTOR_WEIGHT,
        "cash_score": CASH_SCORE,
        "penalties": {
            "vol": LAMBDA_VOL,
            "drawdown": LAMBDA_DRAWDOWN,
            "weight": LAMBDA_WEIGHT,
            "sector": LAMBDA_SECTOR,
            "correlation": LAMBDA_CORRELATION,
            "uncertainty": LAMBDA_UNCERTAINTY,
        },
        "seed": RANDOM_SEED,
    }
    with open(OUTPUT_DIR / "demo_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def save_data_outputs(panel: pd.DataFrame, result: Dict[str, pd.DataFrame]) -> None:
    panel.to_csv(OUTPUT_DIR / "investment_state_panel.csv", index=False)
    result["nav"].to_csv(OUTPUT_DIR / "daily_nav.csv")
    result["flows"].to_csv(OUTPUT_DIR / "daily_external_cashflows.csv")
    result["twr_returns"].to_csv(OUTPUT_DIR / "daily_time_weighted_returns.csv")
    result["drawdown"].to_csv(OUTPUT_DIR / "daily_time_weighted_drawdown.csv")
    result["summary"].to_csv(OUTPUT_DIR / "backtest_summary.csv")
    result["trades"].to_csv(OUTPUT_DIR / "dynamic_strategy_trades.csv", index=False)
    result["allocations"].to_csv(OUTPUT_DIR / "monthly_allocations.csv", index=False)
    result["weights"].to_csv(OUTPUT_DIR / "monthly_portfolio_weights.csv", index=False)
    result["scores"].to_csv(OUTPUT_DIR / "decision_scores.csv", index=False)
    result["model_diagnostics"].to_csv(OUTPUT_DIR / "model_diagnostics.csv", index=False)
    result["neighbors"].to_csv(OUTPUT_DIR / "nearest_historical_states.csv", index=False)


def plot_results(result: Dict[str, pd.DataFrame]) -> None:
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    nav = result["nav"]
    dd = result["drawdown"]
    weights = result["weights"]
    allocations = result["allocations"]
    scores = result["scores"]
    neighbors = result["neighbors"]

    # Figure 1: Actual wealth paths with identical external cash flows.
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    for c in nav.columns:
        ax.plot(nav.index, nav[c], label=c, linewidth=1.5)
    ax.set_title("Long-Horizon Incremental Allocation: Portfolio Wealth")
    ax.set_ylabel("Portfolio value ($)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_1_portfolio_wealth.png", dpi=180)

    # Figure 2: Time-weighted drawdowns, not raw NAV drawdowns distorted by deposits.
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    for c in dd.columns:
        ax.plot(dd.index, 100.0 * dd[c], label=c, linewidth=1.4)
    ax.set_title("Time-Weighted Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_2_drawdown.png", dpi=180)

    # Figure 3: Proposed portfolio top holdings through time.
    if not weights.empty:
        w = weights[weights["strategy"] == "Proposed_TS"].copy()
        if not w.empty:
            candidate_cols = [c for c in UNIVERSE if c in w.columns]
            mean_w = w[candidate_cols].mean().sort_values(ascending=False)
            top = list(mean_w.head(TOP_HOLDINGS_TO_PLOT).index)
            plot_df = w.set_index("date")[top + ["CASH"]].fillna(0.0)
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.stackplot(plot_df.index, [plot_df[c].values for c in plot_df.columns], labels=plot_df.columns)
            ax.set_title("Proposed Strategy: Top Portfolio Weights + Cash")
            ax.set_ylabel("Portfolio weight")
            ax.set_ylim(0, 1.0)
            ax.legend(loc="upper left", ncol=3, fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / "figure_3_proposed_weights.png", dpi=180)

    # Figure 4: Monthly incremental allocation heatmap.
    if not allocations.empty:
        a = allocations[allocations["strategy"] == "Proposed_TS"].copy()
        if not a.empty:
            totals = a.groupby("ticker")["allocated_cash"].sum().sort_values(ascending=False)
            selected = list(totals.drop(index="CASH", errors="ignore").head(TOP_HEATMAP_TICKERS).index)
            if "CASH" in totals.index:
                selected.append("CASH")
            p = a[a["ticker"].isin(selected)].pivot_table(
                index="ticker", columns="execution_date", values="allocated_cash",
                aggfunc="sum", fill_value=0.0
            )
            fig = plt.figure(figsize=(13, 6))
            ax = fig.add_subplot(111)
            im = ax.imshow(p.values, aspect="auto", interpolation="nearest")
            ax.set_yticks(range(len(p.index)))
            ax.set_yticklabels(p.index, fontsize=8)
            if p.shape[1] > 0:
                step = max(1, p.shape[1] // 10)
                ticks = list(range(0, p.shape[1], step))
                ax.set_xticks(ticks)
                ax.set_xticklabels([pd.Timestamp(p.columns[i]).strftime("%Y-%m") for i in ticks], rotation=45, ha="right")
            ax.set_title("Proposed Strategy: Monthly New-Capital Allocation Heatmap")
            fig.colorbar(im, ax=ax, label="Allocated cash ($)")
            fig.tight_layout()
            fig.savefig(fig_dir / "figure_4_allocation_heatmap.png", dpi=180)

    # Figure 5: Final decision opportunity vs portfolio-aware initial marginal utility.
    if not scores.empty:
        last_date = scores["execution_date"].max()
        s = scores[scores["execution_date"] == last_date].copy()
        if not s.empty:
            s = s.sort_values("q_mean", ascending=False).head(12)
            x = np.arange(len(s))
            width = 0.38
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.bar(x - width / 2, s["q_mean"], width, label="Expected opportunity Q")
            ax.bar(x + width / 2, s["mu_proposed_initial"], width, label="Initial marginal utility")
            ax.axhline(0.0, linewidth=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(s["ticker"], rotation=45, ha="right")
            ax.set_title(f"Opportunity vs Next-Dollar Utility ({pd.Timestamp(last_date).date()})")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(fig_dir / "figure_5_opportunity_vs_marginal_utility.png", dpi=180)

    # Figure 6: "Same ticker, different state" nearest historical stock-time states.
    if not neighbors.empty:
        n = neighbors.copy()
        labels = [f"{t}\n{pd.Timestamp(d).strftime('%Y-%m')}" for t, d in zip(n["ticker"], n["date"])]
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.bar(range(len(n)), n["distance"].values)
        ax.set_xticks(range(len(n)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        focus = n["focus_ticker"].iloc[0]
        ax.set_title(f"Nearest Historical Investment States to Current {focus}")
        ax.set_ylabel("Standardized state distance")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "figure_6_nearest_historical_states.png", dpi=180)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


# =============================================================================
# 13. CONSOLE REPORT
# =============================================================================

def print_console_report(universe: List[str], panel: pd.DataFrame, result: Dict[str, pd.DataFrame]) -> None:
    summary = result["summary"].copy()

    print("\n" + "=" * 94)
    print("STOCKS AS TIME-VARYING INVESTMENT STATES -- DEMO BACKTEST")
    print("=" * 94)
    print(f"Dynamic stock universe used: {len(universe)}")
    print(", ".join(universe))
    print(f"Feature-state rows: {len(panel):,}")
    print(f"Backtest: {result['nav'].index.min().date()} -> {result['nav'].index.max().date()}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Monthly contribution: ${MONTHLY_CONTRIBUTION:,.0f}")
    print("\nImportant: Max Drawdown below is TIME-WEIGHTED, so monthly deposits do not hide losses.\n")

    display_cols = [
        "Final Wealth", "Total Contributions", "Wealth Minus Contributions",
        "TWR CAGR", "Ann Vol", "Sharpe", "Max Drawdown", "Calmar", "Final Cash",
    ]
    printable = summary[display_cols].copy()
    for c in ["Final Wealth", "Total Contributions", "Wealth Minus Contributions", "Final Cash"]:
        printable[c] = printable[c].map(lambda x: f"${x:,.0f}" if np.isfinite(x) else "n/a")
    for c in ["TWR CAGR", "Ann Vol", "Max Drawdown"]:
        printable[c] = printable[c].map(fmt_pct)
    for c in ["Sharpe", "Calmar"]:
        printable[c] = printable[c].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "n/a")
    print(printable.to_string())

    trades = result["trades"]
    if not trades.empty:
        ts = trades[trades["strategy"] == "Proposed_TS"]
        print("\nProposed Thompson / next-dollar trades:")
        print(f"  micro-buys executed: {len(ts):,}")
        print(f"  distinct tickers bought: {ts['ticker'].nunique()}")
        if len(ts) > 0:
            top = ts.groupby("ticker")["cash_spent"].sum().sort_values(ascending=False).head(10)
            print("  largest cumulative allocations:")
            for t, amt in top.items():
                print(f"    {t:6s} ${amt:,.0f}")

    diag = result["model_diagnostics"]
    if not diag.empty:
        ok = diag[diag["status"] == "ok"]
        if not ok.empty:
            last = ok.iloc[-1]
            print("\nLatest walk-forward model:")
            print(f"  signal date: {pd.Timestamp(last['signal_date']).date()}")
            print(f"  training rows: {int(last['train_rows']):,}")
            print(f"  candidates: {int(last['candidate_rows']):,}")
            print(f"  empirical noise std: {safe_float(last.get('noise_std')):.4f}")
            print(f"  top expected opportunity: {last.get('top_q_ticker')} ({safe_float(last.get('top_q_mean')):.4f})")

    neighbors = result["neighbors"]
    if not neighbors.empty:
        focus = neighbors["focus_ticker"].iloc[0]
        print(f"\nNearest historical investment states to current {focus}:")
        show = neighbors[["ticker", "date", "distance", "target_log_return"]].copy()
        show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
        show["target_log_return"] = show["target_log_return"].map(lambda x: f"{x:.4f}")
        show["distance"] = show["distance"].map(lambda x: f"{x:.3f}")
        print(show.to_string(index=False))

    print("\nSaved outputs to:")
    print(f"  {OUTPUT_DIR}")
    print("\nResearch demo only. Historical backtests are not evidence of future performance.")
    print("=" * 94 + "\n")


# =============================================================================
# 14. MAIN
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.random.seed(RANDOM_SEED)
    ensure_output_dir()

    print("\nStocks as Time-Varying Investment States")
    print("Long-Horizon Incremental Allocation RL / Contextual-Bandit Demo")
    print("-" * 76)

    prices = load_or_download_prices()

    missing_bench = [b for b in BENCHMARKS if b not in prices.columns or prices[b].notna().sum() < MIN_HISTORY_DAYS]
    if missing_bench:
        raise RuntimeError(f"Required benchmark data missing or too short: {missing_bench}")

    universe = available_universe(prices)
    if len(universe) < 10:
        raise RuntimeError(
            f"Only {len(universe)} stock symbols have enough Yahoo history. "
            "Check internet/data download or edit UNIVERSE."
        )

    missing = [t for t in UNIVERSE if t not in universe]
    if missing:
        print("Dynamic-universe note: excluded for insufficient/missing history:")
        print("  " + ", ".join(missing))

    # Keep only columns actually used; never backward-fill pre-IPO history.
    keep = universe + BENCHMARKS
    prices = prices[keep].copy()
    prices = prices.sort_index()

    print("Building pooled stock-time investment states ...")
    panel = build_feature_panel(prices, universe)
    print(f"Built {len(panel):,} monthly stock-state rows.")

    save_config(universe)

    print("Running strict walk-forward backtest ...")
    result = run_backtest(prices, panel, universe)

    save_data_outputs(panel, result)
    print_console_report(universe, panel, result)
    plot_results(result)


if __name__ == "__main__":
    main()
