import numpy as np
import pandas as pd

from deca_smg.backtest import simulate
from deca_smg.config import load_config
from deca_smg.features import (
    add_heuristic_score,
    apply_risk_gate,
    build_feature_panel,
    candidate_mask,
)
from deca_smg.model import walk_forward_score


def _synthetic_pump_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-02", periods=110)
    rows = []
    symbols = ["SPY"] + [f"S{i:02d}" for i in range(30)]
    for index, ticker in enumerate(symbols):
        price = 100.0 if ticker == "SPY" else 8.0 + index / 10
        for day, date in enumerate(dates):
            if ticker == "SPY":
                daily_return = rng.normal(0.0003, 0.006)
                volume = 10_000_000
            else:
                pump_today = day >= 22 and (day + index) % 6 == 0
                pump_yesterday = day >= 23 and (day - 1 + index) % 6 == 0
                if pump_today:
                    daily_return = 0.22 + rng.normal(0, 0.01)
                    volume = 5_000_000
                elif pump_yesterday:
                    daily_return = -0.07 + rng.normal(0, 0.01)
                    volume = 2_500_000
                else:
                    daily_return = rng.normal(0.0005, 0.012)
                    volume = 500_000
            open_price = price * (1 + daily_return * 0.30)
            price = price * (1 + daily_return)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": open_price,
                    "high": max(open_price, price) * 1.03,
                    "low": min(open_price, price) * 0.98,
                    "close": price,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_pipeline_uses_only_prior_training_dates():
    config = load_config("config/default.yaml")
    panel = add_heuristic_score(build_feature_panel(_synthetic_pump_panel()))
    candidates = panel.loc[candidate_mask(panel, config)].copy()
    candidates["risk_allowed"] = apply_risk_gate(candidates, config)
    walk_forward = walk_forward_score(candidates, config)
    assert (walk_forward.scored["training_end_date"] < walk_forward.scored["date"]).all()

    result = simulate(walk_forward.scored, config, slippage_bps_per_side=0)
    assert not result.trades.empty
    assert result.trades.groupby("date").size().max() <= 2
    assert result.daily["gross_exposure"].max() <= 0.40
