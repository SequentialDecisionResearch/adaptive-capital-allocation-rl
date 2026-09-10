# -*- coding: utf-8 -*-
"""
SSRN validation program for:
    Stocks as Time-Varying Investment States:
    Cross-Ticker Similarity, Dynamic Allocation, and the Limits of Predictive Transfer

Designed to be run directly in Spyder (Run / F5).

This program DOES NOT modify the original research CSV files or the original
backtest program.  It reads the frozen output panel and independently rebuilds
key validation statistics used in the SSRN manuscript, then adds several
robustness checks requested for the SSRN version:

1. Reproduce the geometry statistics (Table 3 style).
2. Reproduce pooled / cross-ticker / same-ticker kNN forecasts for k=5,12,30.
3. Reproduce subperiod MAE results (Table 5 style).
4. Reproduce paired moving-block bootstrap MAE contrasts.
5. Add candidate-count-matched geometry to remove the mechanical advantage of
   the much larger cross-ticker candidate pool.
6. Add naive forecasting baselines (zero return, pooled historical mean,
   same-ticker historical mean).
7. Add moving-block uncertainty intervals for monthly Spearman IC.
8. If decision_scores.csv is present, independently audit the Bayesian / kNN /
   blended score results reported in the manuscript.
9. Write all new results to the SAME directory, using the prefix
   'ssrn_validation_' so existing files are never overwritten.

Required packages:
    numpy, pandas, matplotlib

Default Windows directory:
    C:\\study_notes\\traval_rec\\Stocks_as_Time_Varying_Investment
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. PATHS / CONFIGURATION
# =============================================================================

ROOT_DIR = Path(r"C:\study_notes\traval_rec\Stocks_as_Time_Varying_Investment")
CONFIG_FILE = ROOT_DIR / "ssrn_validation_config.json"

DEFAULT_CONFIG: Dict = {
    "input_panel": "investment_state_panel.csv",
    "decision_scores_file": "decision_scores.csv",
    "backtest_summary_file": "backtest_summary.csv",
    "validation_start": "2019-01-01",
    "train_lookback_years": 8,
    "min_train_rows": 400,
    "knn_k_values": [5, 12, 30],
    "minimum_same_ticker_history": 5,
    "geometry_primary_k": 30,
    "inverse_distance_floor": 1e-6,
    "bootstrap_resamples": 4000,
    "bootstrap_block_length_months": 3,
    "bootstrap_seed": 42,
    "target_clip": 1.0,
    "make_figures": True,
    "output_prefix": "ssrn_validation_",
    "feature_columns": [
        "ret_21", "ret_63", "ret_126", "ret_252",
        "vol_21", "vol_63", "dd_252",
        "ma50_gap", "ma200_gap",
        "rel_spy_63", "rel_spy_126",
        "market_spy_63", "market_spy_252",
        "market_vol_63", "market_dd_252",
        "qqq_minus_spy_63",
    ],
}


def load_config() -> Dict:
    """Load optional JSON config; missing keys fall back to frozen defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    return cfg


def out_path(cfg: Dict, suffix: str) -> Path:
    return ROOT_DIR / f"{cfg['output_prefix']}{suffix}"


# =============================================================================
# 2. SMALL NUMERICAL UTILITIES
# =============================================================================


def spearman_corr(x: Iterable[float], y: Iterable[float]) -> float:
    """Spearman correlation without requiring scipy."""
    x = pd.Series(np.asarray(list(x), dtype=float))
    y = pd.Series(np.asarray(list(y), dtype=float))
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    rx = x.rank(method="average").to_numpy(dtype=float)
    ry = y.rank(method="average").to_numpy(dtype=float)
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def inverse_distance_prediction(
    distances: np.ndarray,
    targets: np.ndarray,
    k: int,
    floor: float,
) -> float:
    """Inverse-distance weighted kNN prediction."""
    if len(distances) == 0:
        return np.nan
    kk = min(int(k), len(distances))
    ix = np.argpartition(distances, kk - 1)[:kk]
    d = distances[ix]
    yy = targets[ix]
    w = 1.0 / np.maximum(d, floor)
    w = w / w.sum()
    return float(np.sum(w * yy))


def exact_matched_cross_closer_probability(
    cross_distances: np.ndarray,
    same_min_distance: float,
    matched_n: int,
) -> float:
    """
    Exact probability that an equal-sized sample drawn WITHOUT replacement from
    the cross-ticker pool contains at least one state closer than the nearest
    same-ticker state.

    If R of N cross-ticker candidates are closer than the same-ticker minimum,
    and n candidates are sampled, then

        P(any closer) = 1 - C(N-R, n) / C(N, n).

    Using lgamma avoids huge combinatorial integers and makes the statistic
    deterministic (no Monte Carlo noise).
    """
    dc = np.asarray(cross_distances, dtype=float)
    n_cross = len(dc)
    if n_cross == 0 or matched_n <= 0:
        return np.nan

    n = int(min(matched_n, n_cross))
    r = int(np.sum(dc < same_min_distance))

    if r <= 0:
        return 0.0
    if n > n_cross - r:
        return 1.0

    # log[ C(N-R,n) / C(N,n) ]
    log_no_success = (
        math.lgamma(n_cross - r + 1)
        - math.lgamma(n + 1)
        - math.lgamma(n_cross - r - n + 1)
        - math.lgamma(n_cross + 1)
        + math.lgamma(n + 1)
        + math.lgamma(n_cross - n + 1)
    )
    p_no_success = math.exp(log_no_success)
    return float(np.clip(1.0 - p_no_success, 0.0, 1.0))


def moving_block_bootstrap_mean(
    values: np.ndarray,
    resamples: int,
    block_length: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Moving-block bootstrap interval for the mean of a time-ordered series."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan

    point = float(np.mean(x))
    if len(x) == 1 or block_length <= 1:
        rng = np.random.default_rng(seed)
        draws = rng.choice(x, size=(resamples, len(x)), replace=True).mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        return point, float(lo), float(hi)

    L = int(min(block_length, len(x)))
    max_start = len(x) - L
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples, dtype=float)

    for b in range(resamples):
        rebuilt: List[float] = []
        while len(rebuilt) < len(x):
            start = int(rng.integers(0, max_start + 1))
            rebuilt.extend(x[start : start + L].tolist())
        boot[b] = float(np.mean(rebuilt[: len(x)]))

    lo, hi = np.quantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)


# =============================================================================
# 3. LOAD AND VALIDATE THE FROZEN PANEL
# =============================================================================


def load_panel(cfg: Dict) -> pd.DataFrame:
    panel_path = ROOT_DIR / cfg["input_panel"]
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n  {panel_path}\n\n"
            "Place ssrn_validation.py and ssrn_validation_config.json in the "
            "same project directory as investment_state_panel.csv."
        )

    panel = pd.read_csv(panel_path)
    required = {
        "date", "ticker", "target_log_return", "target_available_date",
        *cfg["feature_columns"],
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"investment_state_panel.csv is missing columns: {missing}")

    panel["date"] = pd.to_datetime(panel["date"])
    panel["target_available_date"] = pd.to_datetime(
        panel["target_available_date"], errors="coerce"
    )
    panel["ticker"] = panel["ticker"].astype(str)
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel


# =============================================================================
# 4. WALK-FORWARD VALIDATION ENGINE
# =============================================================================


def run_walk_forward_validation(panel: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = list(cfg["feature_columns"])
    ks = sorted({int(k) for k in cfg["knn_k_values"]})
    lookback_years = int(cfg["train_lookback_years"])
    min_train_rows = int(cfg["min_train_rows"])
    min_same_history = int(cfg["minimum_same_ticker_history"])
    primary_k = int(cfg["geometry_primary_k"])
    floor = float(cfg["inverse_distance_floor"])
    validation_start = pd.Timestamp(cfg["validation_start"])

    eval_panel = panel[
        (panel["date"] >= validation_start)
        & panel["target_log_return"].notna()
    ].copy()

    forecast_rows: List[Dict] = []
    geometry_rows: List[Dict] = []

    grouped_eval = list(eval_panel.groupby("date", sort=True))
    print(f"Evaluation dates: {len(grouped_eval)}")
    print(f"Evaluation stock-month rows: {len(eval_panel):,}")

    for date_idx, (signal_date, current) in enumerate(grouped_eval, start=1):
        lookback_start = signal_date - pd.DateOffset(years=lookback_years)
        train = panel[
            (panel["date"] >= lookback_start)
            & (panel["date"] < signal_date)
            & panel["target_available_date"].notna()
            & (panel["target_available_date"] <= signal_date)
            & panel["target_log_return"].notna()
        ].copy()

        if len(train) < min_train_rows:
            print(
                f"Skipping {signal_date.date()}: only {len(train)} mature training rows "
                f"(< {min_train_rows})."
            )
            continue

        x_train = train[features].to_numpy(dtype=float)
        feature_mean = np.nanmean(x_train, axis=0)
        feature_std = np.nanstd(x_train, axis=0, ddof=0)
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
        z_train = (x_train - feature_mean) / feature_std

        y_train = train["target_log_return"].to_numpy(dtype=float)
        train_tickers = train["ticker"].to_numpy(dtype=str)
        pooled_mean = float(np.mean(y_train))

        for _, row in current.iterrows():
            ticker = str(row["ticker"])
            actual = float(row["target_log_return"])
            z = (row[features].to_numpy(dtype=float) - feature_mean) / feature_std
            distances = np.sqrt(np.sum((z_train - z) ** 2, axis=1))

            same_mask = train_tickers == ticker
            cross_mask = ~same_mask
            same_n = int(np.sum(same_mask))
            cross_n = int(np.sum(cross_mask))

            rec: Dict = {
                "date": signal_date,
                "ticker": ticker,
                "actual_return": actual,
                "train_rows": len(train),
                "same_history_rows": same_n,
                "cross_history_rows": cross_n,
                "zero_return_prediction": 0.0,
                "pooled_historical_mean_prediction": pooled_mean,
                "same_ticker_historical_mean_prediction": (
                    float(np.mean(y_train[same_mask])) if same_n >= min_same_history else np.nan
                ),
            }

            pool_specs = {
                "pooled": np.ones(len(train), dtype=bool),
                "cross": cross_mask,
                "same": same_mask,
            }

            for method, mask in pool_specs.items():
                d_method = distances[mask]
                y_method = y_train[mask]

                # Match the manuscript validation protocol: same-ticker forecasts
                # require at least 5 own-history observations; for k>available count,
                # use all available own-history observations.
                same_available = not (
                    method == "same" and same_n < min_same_history
                )
                for k in ks:
                    col = f"pred_{method}_k{k}"
                    rec[col] = (
                        inverse_distance_prediction(d_method, y_method, k, floor)
                        if same_available and len(d_method) > 0
                        else np.nan
                    )

            forecast_rows.append(rec)

            # Geometry diagnostics. A nearest same-ticker state requires >=1 own
            # historical point, so a small number of early-listing cases are not
            # comparable for the nearest-same vs nearest-cross statistic.
            pooled_k_ix = np.argpartition(
                distances, min(primary_k, len(distances)) - 1
            )[: min(primary_k, len(distances))]
            observed_same_share = float(np.mean(same_mask[pooled_k_ix]))
            expected_same_share = float(same_n / len(train))

            geom: Dict = {
                "date": signal_date,
                "ticker": ticker,
                "train_rows": len(train),
                "same_history_rows": same_n,
                "cross_history_rows": cross_n,
                f"observed_same_share_in_pooled_k{primary_k}": observed_same_share,
                "expected_same_share_from_training_pool": expected_same_share,
            }

            if same_n > 0 and cross_n > 0:
                same_dist = distances[same_mask]
                cross_dist = distances[cross_mask]
                same_min = float(np.min(same_dist))
                cross_min = float(np.min(cross_dist))
                closer_cross_count = int(np.sum(cross_dist < same_min))
                matched_n = min(same_n, cross_n)

                geom.update({
                    "nearest_same_distance": same_min,
                    "nearest_cross_distance": cross_min,
                    "raw_cross_closer": float(cross_min < same_min),
                    "cross_candidates_closer_than_same_min": closer_cross_count,
                    "matched_candidate_count": matched_n,
                    "matched_cross_closer_probability": (
                        exact_matched_cross_closer_probability(
                            cross_dist, same_min, matched_n
                        )
                    ),
                })
            else:
                geom.update({
                    "nearest_same_distance": np.nan,
                    "nearest_cross_distance": np.nan,
                    "raw_cross_closer": np.nan,
                    "cross_candidates_closer_than_same_min": np.nan,
                    "matched_candidate_count": np.nan,
                    "matched_cross_closer_probability": np.nan,
                })

            geometry_rows.append(geom)

        if date_idx == 1 or date_idx % 12 == 0 or date_idx == len(grouped_eval):
            print(
                f"Processed {date_idx:>3}/{len(grouped_eval)} dates through "
                f"{signal_date.date()}"
            )

    forecasts = pd.DataFrame(forecast_rows)
    geometry = pd.DataFrame(geometry_rows)
    return forecasts, geometry


# =============================================================================
# 5. SUMMARY TABLES
# =============================================================================


def forecast_metrics(df: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    z = df[["date", "actual_return", pred_col]].dropna().copy()
    if z.empty:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "Mean monthly Spearman IC": np.nan}

    err = z[pred_col] - z["actual_return"]
    monthly_ic = []
    for _, g in z.groupby("date", sort=True):
        ic = spearman_corr(g[pred_col], g["actual_return"])
        if np.isfinite(ic):
            monthly_ic.append(ic)

    return {
        "n": int(len(z)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Mean monthly Spearman IC": float(np.mean(monthly_ic)) if monthly_ic else np.nan,
    }


def make_geometry_summary(geometry: pd.DataFrame, forecasts: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    primary_k = int(cfg["geometry_primary_k"])
    comparable = geometry.dropna(subset=["nearest_same_distance", "nearest_cross_distance"])

    obs_col = f"observed_same_share_in_pooled_k{primary_k}"
    observed = float(geometry[obs_col].mean())
    expected = float(geometry["expected_same_share_from_training_pool"].mean())

    rows = [
        ("Evaluation stock-months", float(len(forecasts))),
        ("Evaluation dates", float(forecasts["date"].nunique())),
        ("Comparable nearest-same/cross cases", float(len(comparable))),
        ("Raw P(nearest cross < nearest same)", float(comparable["raw_cross_closer"].mean())),
        ("Median nearest cross standardized distance", float(comparable["nearest_cross_distance"].median())),
        ("Median nearest same standardized distance", float(comparable["nearest_same_distance"].median())),
        (f"Observed same-ticker share among pooled k={primary_k}", observed),
        ("Expected same-ticker share from eligible history", expected),
        ("Observed / expected same-ticker neighbor ratio", observed / expected if expected > 0 else np.nan),
        (f"Cross-ticker share among pooled k={primary_k}", 1.0 - observed),
        (
            "Candidate-count-matched P(cross closer)",
            float(comparable["matched_cross_closer_probability"].mean()),
        ),
    ]
    return pd.DataFrame(rows, columns=["Diagnostic", "Value"])


def make_table4(forecasts: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    rows = []
    for k in sorted({int(x) for x in cfg["knn_k_values"]}):
        for method, label in [
            ("pooled", "Pooled all tickers"),
            ("cross", "Cross-ticker only"),
            ("same", "Same ticker only"),
        ]:
            pred_col = f"pred_{method}_k{k}"
            metrics = forecast_metrics(forecasts, pred_col)
            rows.append({
                "k": k,
                "History restriction": label,
                **metrics,
            })
    return pd.DataFrame(rows)


def make_subperiod_table(forecasts: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    k = max(int(x) for x in cfg["knn_k_values"])
    periods = [
        ("2019-2021", 2019, 2021),
        ("2022-2024", 2022, 2024),
        ("2025-2026", 2025, 2026),
    ]

    rows = []
    for label, y0, y1 in periods:
        z = forecasts[forecasts["date"].dt.year.between(y0, y1)].copy()
        row = {"Period": label}
        for method, out_name in [
            ("pooled", "Pooled all tickers"),
            ("cross", "Cross-ticker only"),
            ("same", "Same ticker only"),
        ]:
            pred_col = f"pred_{method}_k{k}"
            zz = z[["actual_return", pred_col]].dropna()
            row[out_name] = float(np.mean(np.abs(zz[pred_col] - zz["actual_return"])))
        rows.append(row)
    return pd.DataFrame(rows)


def make_naive_baselines(forecasts: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    k = max(int(x) for x in cfg["knn_k_values"])
    same_col = f"pred_same_k{k}"
    paired = forecasts[forecasts[same_col].notna()].copy()

    specs = [
        ("Full available", forecasts, "Zero return", "zero_return_prediction"),
        ("Full available", forecasts, "Pooled historical mean", "pooled_historical_mean_prediction"),
        ("Same-history eligible", paired, "Same-ticker historical mean", "same_ticker_historical_mean_prediction"),
        ("Same-history eligible", paired, f"Same-ticker kNN k={k}", same_col),
        ("Same-history eligible", paired, f"Cross-ticker kNN k={k}", f"pred_cross_k{k}"),
        ("Same-history eligible", paired, f"Pooled kNN k={k}", f"pred_pooled_k{k}"),
        ("Paired same-history sample", paired, "Zero return", "zero_return_prediction"),
        ("Paired same-history sample", paired, "Pooled historical mean", "pooled_historical_mean_prediction"),
        ("Paired same-history sample", paired, "Same-ticker historical mean", "same_ticker_historical_mean_prediction"),
    ]

    rows = []
    for sample, data, label, col in specs:
        m = forecast_metrics(data, col)
        rows.append({"Sample": sample, "Forecast": label, **m})
    return pd.DataFrame(rows)


def make_bootstrap_mae_contrasts(forecasts: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    B = int(cfg["bootstrap_resamples"])
    L = int(cfg["bootstrap_block_length_months"])
    seed0 = int(cfg["bootstrap_seed"])
    rows = []

    for k in sorted({int(x) for x in cfg["knn_k_values"]}):
        same_col = f"pred_same_k{k}"
        for j, (other, label) in enumerate([
            ("cross", "Same minus cross"),
            ("pooled", "Same minus pooled"),
        ]):
            other_col = f"pred_{other}_k{k}"
            z = forecasts[["date", "actual_return", same_col, other_col]].dropna().copy()
            z["paired_abs_error_difference"] = (
                np.abs(z[same_col] - z["actual_return"])
                - np.abs(z[other_col] - z["actual_return"])
            )
            date_diff = z.groupby("date", sort=True)["paired_abs_error_difference"].mean()
            point, lo, hi = moving_block_bootstrap_mean(
                date_diff.to_numpy(dtype=float),
                resamples=B,
                block_length=L,
                seed=seed0 + 100 * k + j,
            )
            rows.append({
                "k": k,
                "Contrast": label,
                "Paired rows": int(len(z)),
                "Decision dates": int(len(date_diff)),
                "Mean difference": point,
                "95% CI lower": lo,
                "95% CI upper": hi,
            })
    return pd.DataFrame(rows)


def make_ic_intervals(forecasts: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    B = int(cfg["bootstrap_resamples"])
    L = int(cfg["bootstrap_block_length_months"])
    seed0 = int(cfg["bootstrap_seed"])
    rows = []

    for k in sorted({int(x) for x in cfg["knn_k_values"]}):
        for j, (method, label) in enumerate([
            ("pooled", "Pooled all tickers"),
            ("cross", "Cross-ticker only"),
            ("same", "Same ticker only"),
        ]):
            col = f"pred_{method}_k{k}"
            monthly = []
            z = forecasts[["date", "actual_return", col]].dropna()
            for dt, g in z.groupby("date", sort=True):
                ic = spearman_corr(g[col], g["actual_return"])
                if np.isfinite(ic):
                    monthly.append((dt, ic))

            ic_values = np.array([v for _, v in monthly], dtype=float)
            point, lo, hi = moving_block_bootstrap_mean(
                ic_values,
                resamples=B,
                block_length=L,
                seed=seed0 + 1000 + 100 * k + j,
            )
            rows.append({
                "k": k,
                "History restriction": label,
                "Decision dates": len(ic_values),
                "Mean monthly Spearman IC": point,
                "95% CI lower": lo,
                "95% CI upper": hi,
            })
    return pd.DataFrame(rows)


def make_geometry_intervals(geometry: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    B = int(cfg["bootstrap_resamples"])
    L = int(cfg["bootstrap_block_length_months"])
    seed0 = int(cfg["bootstrap_seed"])
    z = geometry.dropna(subset=["raw_cross_closer", "matched_cross_closer_probability"]).copy()

    rows = []
    for j, (col, label) in enumerate([
        ("raw_cross_closer", "Raw P(nearest cross < nearest same)"),
        ("matched_cross_closer_probability", "Candidate-count-matched P(cross closer)"),
    ]):
        date_series = z.groupby("date", sort=True)[col].mean()
        point, lo, hi = moving_block_bootstrap_mean(
            date_series.to_numpy(dtype=float),
            resamples=B,
            block_length=L,
            seed=seed0 + 2000 + j,
        )
        rows.append({
            "Diagnostic": label,
            "Comparable cases": int(len(z)),
            "Decision dates": int(len(date_series)),
            "Mean": point,
            "95% CI lower": lo,
            "95% CI upper": hi,
        })
    return pd.DataFrame(rows)


def make_target_boundary_diagnostic(panel: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    clip = float(cfg["target_clip"])
    y = pd.to_numeric(panel["target_log_return"], errors="coerce").dropna()
    at_boundary = np.isclose(np.abs(y.to_numpy(dtype=float)), clip, atol=1e-12)
    return pd.DataFrame([{
        "Observed non-missing targets": int(len(y)),
        "Target clip boundary": clip,
        "Rows exactly at +/- clip boundary": int(np.sum(at_boundary)),
        "Share at clip boundary": float(np.mean(at_boundary)) if len(y) else np.nan,
        "Note": (
            "The frozen panel stores the clipped target. Rows exactly at the boundary "
            "are a diagnostic, not proof that every such row was changed by clipping."
        ),
    }])


# =============================================================================
# 6. OPTIONAL AUDIT OF decision_scores.csv
# =============================================================================


def audit_decision_scores(panel: pd.DataFrame, cfg: Dict) -> Optional[pd.DataFrame]:
    path = ROOT_DIR / cfg["decision_scores_file"]
    if not path.exists():
        print(f"Optional file not found; skipping blended-score audit: {path.name}")
        return None

    scores = pd.read_csv(path)
    required = {"signal_date", "ticker", "q_mean", "bayes_mean", "knn_mean"}
    if not required.issubset(scores.columns):
        print(f"Skipping blended-score audit: {path.name} lacks required columns.")
        return None

    scores["signal_date"] = pd.to_datetime(scores["signal_date"])
    targets = panel[["date", "ticker", "target_log_return"]].copy()
    merged = scores.merge(
        targets,
        left_on=["signal_date", "ticker"],
        right_on=["date", "ticker"],
        how="left",
    )
    merged = merged[merged["target_log_return"].notna()].copy()
    merged = merged.rename(columns={"signal_date": "date_for_group"})

    rows = []
    for col, label in [
        ("q_mean", "Bayesian + pooled-kNN blend"),
        ("bayes_mean", "Bayesian component"),
        ("knn_mean", "Pooled kNN component"),
    ]:
        err = merged[col] - merged["target_log_return"]
        ics = []
        for _, g in merged.groupby("date_for_group", sort=True):
            ic = spearman_corr(g[col], g["target_log_return"])
            if np.isfinite(ic):
                ics.append(ic)
        rows.append({
            "Score": label,
            "n": int(len(merged)),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "Mean monthly Spearman IC": float(np.mean(ics)) if ics else np.nan,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 7. FIGURES
# =============================================================================


def make_validation_figures(
    geometry_summary: pd.DataFrame,
    naive: pd.DataFrame,
    cfg: Dict,
) -> None:
    if not bool(cfg.get("make_figures", True)):
        return

    # Figure A: raw geometry versus candidate-count-matched geometry.
    value_map = dict(zip(geometry_summary["Diagnostic"], geometry_summary["Value"]))
    raw = value_map["Raw P(nearest cross < nearest same)"]
    matched = value_map["Candidate-count-matched P(cross closer)"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = ["Raw nearest-neighbor\ncomparison", "Candidate-count\nmatched"]
    vals = [100 * raw, 100 * matched]
    bars = ax.bar(labels, vals)
    ax.axhline(50.0, linewidth=1.0, linestyle="--")
    ax.set_ylabel("Cross-ticker closer probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Raw vs Candidate-Count-Matched State Geometry")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(out_path(cfg, "geometry_raw_vs_matched.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure B: k=30 and simple baselines on the paired same-history sample.
    paired = naive[naive["Sample"] == "Paired same-history sample"].copy()
    extra = naive[
        (naive["Sample"] == "Same-history eligible")
        & naive["Forecast"].str.contains("k=", regex=False)
    ].copy()
    plot_df = pd.concat([paired, extra], ignore_index=True)
    plot_df = plot_df.drop_duplicates(subset=["Forecast"], keep="first")

    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        x = np.arange(len(plot_df))
        vals = 100 * plot_df["MAE"].to_numpy(dtype=float)
        bars = ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["Forecast"], rotation=25, ha="right")
        ax.set_ylabel("MAE (percentage points)")
        ax.set_title("Forecast MAE on Comparable Validation Samples")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.08, f"{val:.2f}", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path(cfg, "forecast_mae_baselines.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 8. REPORT WRITER
# =============================================================================


def pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{100.0 * x:.2f}%"


def write_text_report(
    geometry_summary: pd.DataFrame,
    table4: pd.DataFrame,
    subperiod: pd.DataFrame,
    naive: pd.DataFrame,
    bootstrap: pd.DataFrame,
    ic_intervals: pd.DataFrame,
    score_audit: Optional[pd.DataFrame],
    cfg: Dict,
) -> None:
    gm = dict(zip(geometry_summary["Diagnostic"], geometry_summary["Value"]))
    kmax = max(int(x) for x in cfg["knn_k_values"])

    lines = []
    lines.append("SSRN VALIDATION REPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Inputs are treated as frozen research outputs. No original CSV is modified.")
    lines.append("")
    lines.append("KEY GEOMETRY RESULTS")
    lines.append(f"Evaluation stock-months: {int(gm['Evaluation stock-months']):,}")
    lines.append(f"Evaluation dates: {int(gm['Evaluation dates']):,}")
    lines.append(
        "Comparable nearest-same/cross cases: "
        f"{int(gm['Comparable nearest-same/cross cases']):,}"
    )
    lines.append(
        "Raw P(nearest cross < nearest same): "
        f"{pct(gm['Raw P(nearest cross < nearest same)'])}"
    )
    lines.append(
        "Candidate-count-matched P(cross closer): "
        f"{pct(gm['Candidate-count-matched P(cross closer)'])}"
    )
    obs_key = f"Observed same-ticker share among pooled k={cfg['geometry_primary_k']}"
    lines.append(
        "Observed same-ticker share among pooled neighbors: "
        f"{pct(gm[obs_key])}"
    )
    lines.append(
        "Expected same-ticker share from eligible history: "
        f"{pct(gm['Expected same-ticker share from eligible history'])}"
    )
    lines.append("")
    lines.append(f"KNN FORECAST RESULTS (k={kmax})")
    t = table4[table4["k"] == kmax]
    for _, row in t.iterrows():
        lines.append(
            f"{row['History restriction']}: n={int(row['n'])}, "
            f"MAE={pct(row['MAE'])}, RMSE={pct(row['RMSE'])}, "
            f"mean monthly IC={row['Mean monthly Spearman IC']:.4f}"
        )
    lines.append("")
    lines.append("NAIVE BASELINES")
    paired = naive[naive["Sample"] == "Paired same-history sample"]
    for _, row in paired.iterrows():
        lines.append(f"{row['Forecast']}: n={int(row['n'])}, MAE={pct(row['MAE'])}")
    lines.append("")
    lines.append("PAIRED MOVING-BLOCK BOOTSTRAP")
    for _, row in bootstrap.iterrows():
        lines.append(
            f"k={int(row['k'])}, {row['Contrast']}: mean={100*row['Mean difference']:.2f} pp, "
            f"95% CI=[{100*row['95% CI lower']:.2f}, {100*row['95% CI upper']:.2f}] pp"
        )
    lines.append("")
    lines.append("SUBPERIOD RESULTS")
    lines.append(subperiod.to_string(index=False))
    lines.append("")
    lines.append("IC INTERVALS")
    lines.append(ic_intervals.to_string(index=False))

    if score_audit is not None:
        lines.append("")
        lines.append("DECISION-SCORE AUDIT")
        lines.append(score_audit.to_string(index=False))

    lines.append("")
    lines.append("Interpretive note:")
    lines.append(
        "The raw nearest-neighbor comparison and the candidate-count-matched statistic "
        "answer different questions. The matched statistic controls the mechanical "
        "minimum-distance advantage created by the much larger cross-ticker pool."
    )

    with open(out_path(cfg, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# 9. MAIN
# =============================================================================


def main() -> None:
    print("=" * 88)
    print("SSRN VALIDATION: Stocks as Time-Varying Investment States")
    print("=" * 88)
    print(f"Root directory: {ROOT_DIR}")

    if not ROOT_DIR.exists():
        raise FileNotFoundError(
            f"Project directory does not exist:\n  {ROOT_DIR}\n"
            "If your folder moved, edit ROOT_DIR near the top of ssrn_validation.py."
        )

    cfg = load_config()
    panel = load_panel(cfg)
    print(f"Loaded panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers")
    print(f"Panel dates: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print()

    forecasts, geometry = run_walk_forward_validation(panel, cfg)

    geometry_summary = make_geometry_summary(geometry, forecasts, cfg)
    table4 = make_table4(forecasts, cfg)
    subperiod = make_subperiod_table(forecasts, cfg)
    naive = make_naive_baselines(forecasts, cfg)
    bootstrap = make_bootstrap_mae_contrasts(forecasts, cfg)
    ic_intervals = make_ic_intervals(forecasts, cfg)
    geometry_intervals = make_geometry_intervals(geometry, cfg)
    target_boundary = make_target_boundary_diagnostic(panel, cfg)
    score_audit = audit_decision_scores(panel, cfg)

    # Case-level outputs are intentionally retained for transparent replication.
    forecasts.to_csv(out_path(cfg, "forecast_case_level.csv"), index=False)
    geometry.to_csv(out_path(cfg, "geometry_case_level.csv"), index=False)
    geometry_summary.to_csv(out_path(cfg, "table3_geometry.csv"), index=False)
    table4.to_csv(out_path(cfg, "table4_knn_forecasts.csv"), index=False)
    subperiod.to_csv(out_path(cfg, "table5_subperiods.csv"), index=False)
    naive.to_csv(out_path(cfg, "naive_baselines.csv"), index=False)
    bootstrap.to_csv(out_path(cfg, "bootstrap_mae_contrasts.csv"), index=False)
    ic_intervals.to_csv(out_path(cfg, "ic_intervals.csv"), index=False)
    geometry_intervals.to_csv(out_path(cfg, "geometry_intervals.csv"), index=False)
    target_boundary.to_csv(out_path(cfg, "target_boundary_diagnostic.csv"), index=False)
    if score_audit is not None:
        score_audit.to_csv(out_path(cfg, "decision_score_audit.csv"), index=False)

    make_validation_figures(geometry_summary, naive, cfg)
    write_text_report(
        geometry_summary, table4, subperiod, naive, bootstrap,
        ic_intervals, score_audit, cfg,
    )

    print("\n" + "=" * 88)
    print("KEY CHECKS")
    print("=" * 88)
    print(geometry_summary.to_string(index=False))
    print("\nTable 4-style kNN validation:")
    print(table4.to_string(index=False))
    print("\nNaive baselines:")
    print(naive.to_string(index=False))
    print("\nPaired moving-block bootstrap:")
    print(bootstrap.to_string(index=False))

    print("\nFiles written to:")
    print(f"  {ROOT_DIR}")
    print("All new files begin with 'ssrn_validation_'.")
    print("Original research files were not modified.")
    print("=" * 88)


if __name__ == "__main__":
    main()
