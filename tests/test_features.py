import numpy as np
import pandas as pd
import pytest

from deca_smg.features import build_feature_panel


def test_target_is_next_snapshot_short_return_without_lookahead():
    dates = pd.bdate_range("2026-01-02", periods=45)
    prices = []
    spy = np.linspace(100, 104, len(dates))
    stock = np.linspace(10, 20, len(dates))
    stock[-2:] = [20, 18]
    for ticker, closes in [("SPY", spy), ("PUMP", stock)]:
        for date, close in zip(dates, closes, strict=True):
            prices.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close * 0.98,
                    "high": close * 1.03,
                    "low": close * 0.97,
                    "close": close,
                    "volume": 1_000_000,
                }
            )
    panel = build_feature_panel(pd.DataFrame(prices))
    row = panel.loc[
        panel["ticker"].eq("PUMP") & panel["date"].eq(dates[-2])
    ].iloc[0]
    assert row["target_short_return_1d"] == pytest.approx(0.10)
    assert row["exit_date"] == dates[-1]


def test_features_do_not_cross_ticker_boundaries():
    dates = pd.bdate_range("2026-01-02", periods=25)
    rows = []
    for ticker, close in [("SPY", 100.0), ("AAA", 5.0), ("BBB", 50.0)]:
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 100_000,
                }
            )
    panel = build_feature_panel(pd.DataFrame(rows))
    first = panel.groupby("ticker", as_index=False).head(1)
    assert first["return_1d"].isna().all()
