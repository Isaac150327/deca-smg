from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

FEATURE_COLUMNS = [
    "return_1d",
    "return_2d",
    "return_5d",
    "return_20d",
    "gap_return",
    "intraday_return",
    "range_pct",
    "close_location",
    "upper_wick_pct",
    "return_zscore_20",
    "volume_ratio_20",
    "log_dollar_volume",
    "realized_vol_20",
    "beta_40",
    "correlation_40",
    "excess_return_1d",
    "distance_from_ma20",
    "momentum_acceleration",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _normalize_columns(prices: pd.DataFrame) -> pd.DataFrame:
    result = prices.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    missing = REQUIRED_PRICE_COLUMNS.difference(result.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")

    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.tz_localize(None)
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for optional in ["market_cap", "short_interest_pct", "execution_close"]:
        if optional in result.columns:
            result[optional] = pd.to_numeric(result[optional], errors="coerce")
    for optional in ["binary_event_risk", "material_catalyst"]:
        if optional in result.columns:
            result[optional] = result[optional].fillna(False).astype(bool)

    result = result.drop_duplicates(["date", "ticker"], keep="last")
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    price_columns = ["open", "high", "low", "close"]
    invalid = result[price_columns].isna().any(axis=1) | (result[price_columns] <= 0).any(axis=1)
    return result.loc[~invalid].copy()


def _single_ticker_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").copy()
    close = frame["close"]
    previous_close = close.shift(1)
    daily_return = close.pct_change(fill_method=None)

    frame["return_1d"] = daily_return
    frame["return_2d"] = close.pct_change(2, fill_method=None)
    frame["return_5d"] = close.pct_change(5, fill_method=None)
    frame["return_20d"] = close.pct_change(20, fill_method=None)
    frame["gap_return"] = _safe_divide(frame["open"], previous_close) - 1
    frame["intraday_return"] = _safe_divide(close, frame["open"]) - 1
    frame["range_pct"] = _safe_divide(frame["high"] - frame["low"], previous_close)

    daily_range = frame["high"] - frame["low"]
    frame["close_location"] = _safe_divide(close - frame["low"], daily_range).clip(0, 1)
    frame["upper_wick_pct"] = _safe_divide(
        frame["high"] - np.maximum(frame["open"], close), previous_close
    ).clip(lower=0)

    trailing_mean = daily_return.rolling(20, min_periods=10).mean()
    trailing_std = daily_return.rolling(20, min_periods=10).std()
    frame["return_zscore_20"] = _safe_divide(daily_return - trailing_mean, trailing_std)
    prior_median_volume = frame["volume"].shift(1).rolling(20, min_periods=10).median()
    frame["volume_ratio_20"] = _safe_divide(frame["volume"], prior_median_volume)
    dollar_volume = close * frame["volume"]
    frame["median_dollar_volume_20"] = close * prior_median_volume
    frame["dollar_volume"] = dollar_volume
    frame["log_dollar_volume"] = np.log1p(dollar_volume.clip(lower=0))
    frame["realized_vol_20"] = daily_return.rolling(20, min_periods=10).std() * np.sqrt(252)
    moving_average = close.shift(1).rolling(20, min_periods=10).mean()
    frame["distance_from_ma20"] = _safe_divide(close, moving_average) - 1
    frame["momentum_acceleration"] = daily_return - frame["return_5d"] / 5

    execution_close = (
        frame["execution_close"].fillna(close)
        if "execution_close" in frame.columns
        else close
    )
    next_close = execution_close.shift(-1)
    frame["previous_execution_close"] = execution_close.shift(1)
    frame["entry_price"] = execution_close
    frame["exit_price"] = next_close
    frame["target_short_return_1d"] = 1 - _safe_divide(next_close, execution_close)
    frame["exit_date"] = frame["date"].shift(-1)
    return frame


def build_feature_panel(prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """Build decision-time features and a next-snapshot short-return target.

    Each input row must represent only information available at the configured
    decision time. For this project that means a 3:40 p.m. ET snapshot, not a
    completed daily bar.
    """

    normalized = _normalize_columns(prices)
    pieces = [_single_ticker_features(group) for _, group in normalized.groupby("ticker")]
    panel = pd.concat(pieces, ignore_index=True)

    benchmark_rows = panel.loc[
        panel["ticker"].eq(benchmark.upper()), ["date", "return_1d"]
    ].rename(columns={"return_1d": "benchmark_return_1d"})
    panel = panel.merge(benchmark_rows, on="date", how="left")

    beta_pieces: list[pd.DataFrame] = []
    for _, group in panel.groupby("ticker", sort=False):
        group = group.sort_values("date").copy()
        covariance = group["return_1d"].rolling(40, min_periods=15).cov(
            group["benchmark_return_1d"]
        )
        benchmark_variance = group["benchmark_return_1d"].rolling(40, min_periods=15).var()
        group["beta_40"] = _safe_divide(covariance, benchmark_variance)
        group["correlation_40"] = group["return_1d"].rolling(40, min_periods=15).corr(
            group["benchmark_return_1d"]
        )
        group["excess_return_1d"] = (
            group["return_1d"] - group["beta_40"] * group["benchmark_return_1d"]
        )
        beta_pieces.append(group)

    result = pd.concat(beta_pieces, ignore_index=True)
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def candidate_mask(panel: pd.DataFrame, config: dict) -> pd.Series:
    rules = config["game_rules"]
    filters = config["candidate_filter"]
    mask = (
        panel["ticker"].ne("SPY")
        & panel["close"].ge(float(rules["minimum_price"]))
        & panel["previous_execution_close"].ge(float(rules["minimum_price"]))
        & panel["close"].le(float(filters["maximum_price"]))
        & panel["return_1d"].ge(float(filters["minimum_day_return"]))
        & panel["return_5d"].ge(float(filters["minimum_five_day_return"]))
        & panel["volume_ratio_20"].ge(float(filters["minimum_volume_ratio"]))
        & panel["dollar_volume"].ge(float(filters["minimum_dollar_volume"]))
        & panel["median_dollar_volume_20"].ge(float(filters["minimum_median_dollar_volume"]))
    )
    if "market_cap" in panel.columns:
        known = panel["market_cap"].notna()
        within = panel["market_cap"].between(
            float(filters["minimum_market_cap"]),
            float(filters["maximum_market_cap"]),
            inclusive="both",
        )
        mask &= ~known | within
    return mask.fillna(False)


def apply_risk_gate(panel: pd.DataFrame, config: dict) -> pd.Series:
    risk = config["risk"]
    allowed = (
        panel["return_1d"].le(float(risk["maximum_day_return"]))
        & panel["close_location"].le(float(risk["maximum_close_location"]))
    )
    if "short_interest_pct" in panel.columns:
        known = panel["short_interest_pct"].notna()
        allowed &= ~known | panel["short_interest_pct"].le(
            float(risk["maximum_short_interest_pct"])
        )
    if risk.get("exclude_binary_event_risk", True) and "binary_event_risk" in panel.columns:
        allowed &= ~panel["binary_event_risk"].fillna(False)
    return allowed.fillna(False)


def add_heuristic_score(panel: pd.DataFrame) -> pd.DataFrame:
    """Add an interpretable exhaustion score used as a cold-start baseline."""

    result = panel.copy()

    def scaled(column: str, low: float, high: float) -> pd.Series:
        return ((result[column] - low) / (high - low)).clip(0, 1).fillna(0)

    result["heuristic_score"] = (
        0.22 * scaled("excess_return_1d", 0.05, 0.60)
        + 0.18 * scaled("return_zscore_20", 1.0, 5.0)
        + 0.15 * scaled("volume_ratio_20", 1.5, 10.0)
        + 0.12 * scaled("return_5d", 0.15, 1.00)
        + 0.13 * scaled("upper_wick_pct", 0.00, 0.20)
        + 0.10 * scaled("realized_vol_20", 0.40, 2.00)
        + 0.05 * scaled("beta_40", 1.0, 4.0)
        + 0.05 * (1 - result["close_location"].clip(0, 1).fillna(1))
    )
    result["continuation_risk"] = (
        (result["close_location"].ge(0.90))
        & (result["return_1d"].ge(0.25))
        & (result["upper_wick_pct"].fillna(0).lt(0.02))
    )
    result["extreme_move_risk"] = (
        result["return_1d"].ge(0.75) | result["range_pct"].ge(0.75)
    )
    return result


def ensure_feature_columns(frame: pd.DataFrame, columns: Iterable[str] = FEATURE_COLUMNS) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Feature data is missing columns: {sorted(missing)}")
