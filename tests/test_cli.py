from argparse import Namespace

import joblib
import pandas as pd

import deca_smg.cli as cli
from deca_smg.config import load_config


def test_backtest_uses_heuristic_artifact_during_cold_start(tmp_path, monkeypatch):
    candidates = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-09-08") + pd.offsets.BDay(index),
                "ticker": f"PUMP{index}",
                "target_short_return_1d": 0.03,
                "heuristic_score": 0.5,
                "risk_allowed": True,
            }
            for index in range(3)
        ]
    )
    monkeypatch.setattr(cli, "read_prices", lambda _path: pd.DataFrame())
    monkeypatch.setattr(cli, "_candidate_panel", lambda _prices, _config: candidates)
    output_dir = tmp_path / "backtest"

    cli.command_backtest(
        Namespace(
            input="unused.csv",
            config="config/default.yaml",
            output_dir=str(output_dir),
        )
    )

    artifact = joblib.load(output_dir / "model.joblib")
    assert artifact["method"] == "fixed_heuristic"
    assert artifact["model"] is None
    assert artifact["selection_enabled"] is False
    assert artifact["validation_status"] == "HEURISTIC_RESEARCH_ONLY"
    assert (output_dir / "metrics.json").exists()


def test_model_validation_requires_enough_profitable_oos_trades():
    config = load_config("config/default.yaml")
    metrics = {
        "trades": 29,
        "total_return": 0.10,
        "average_trade_return": 0.02,
        "max_drawdown": -0.05,
    }
    assert cli._validation_status(metrics, config) == (False, "INSUFFICIENT_OOS_TRADES")

    metrics["trades"] = 30
    assert cli._validation_status(metrics, config) == (True, "VALIDATED_OOS_MODEL")

    metrics["total_return"] = -0.01
    assert cli._validation_status(metrics, config) == (False, "NEGATIVE_OOS_RETURN")
