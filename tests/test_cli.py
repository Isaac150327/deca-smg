from argparse import Namespace

import joblib
import pandas as pd

import deca_smg.cli as cli


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
    assert (output_dir / "metrics.json").exists()
