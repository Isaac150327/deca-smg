import pandas as pd
import pytest

from deca_smg.backtest import simulate


def _config():
    return {
        "game_rules": {
            "transaction_fee_per_order": 0,
            "slippage_bps_per_side": 0,
        },
        "strategy": {"maximum_open_positions": 2, "position_fraction": 0.20},
        "risk": {"minimum_predicted_edge": 0},
        "backtest": {"initial_equity": 100_000},
    }


def test_backtest_selects_only_two_and_sizes_portfolio_contribution():
    rows = []
    for ticker, predicted, realized in [
        ("AAA", 0.06, 0.10),
        ("BBB", 0.05, 0.05),
        ("CCC", 0.04, 0.50),
    ]:
        rows.append(
            {
                "date": pd.Timestamp("2026-02-02"),
                "ticker": ticker,
                "predicted_short_return": predicted,
                "target_short_return_1d": realized,
                "heuristic_score": predicted,
                "risk_allowed": True,
            }
        )
    result = simulate(pd.DataFrame(rows), _config())
    assert list(result.trades["ticker"]) == ["AAA", "BBB"]
    assert result.daily.iloc[0]["portfolio_return"] == pytest.approx(0.03)
    assert result.daily.iloc[0]["gross_exposure"] == pytest.approx(0.40)


def test_costs_are_charged_on_entry_and_exit():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-02-02"),
                "ticker": "AAA",
                "predicted_short_return": 0.02,
                "target_short_return_1d": 0.02,
                "heuristic_score": 1.0,
                "risk_allowed": True,
            }
        ]
    )
    config = _config()
    config["game_rules"]["transaction_fee_per_order"] = 5
    result = simulate(frame, config, slippage_bps_per_side=0)
    assert result.trades.iloc[0]["round_trip_cost"] == pytest.approx(0.0005)
    assert result.trades.iloc[0]["net_short_return"] == pytest.approx(0.0195)


def test_drawdown_includes_loss_from_initial_equity():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-02-02"),
                "ticker": "AAA",
                "predicted_short_return": 0.02,
                "target_short_return_1d": -0.50,
                "heuristic_score": 1.0,
                "risk_allowed": True,
            }
        ]
    )
    result = simulate(frame, _config())
    assert result.metrics["total_return"] == pytest.approx(-0.10)
    assert result.metrics["max_drawdown"] == pytest.approx(-0.10)
