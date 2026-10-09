from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "game_rules",
        "strategy",
        "decision",
        "candidate_filter",
        "risk",
        "model",
        "backtest",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    rules = config["game_rules"]
    strategy = config["strategy"]
    fraction = float(strategy["position_fraction"])
    positions = int(strategy["maximum_open_positions"])
    if not 0 < fraction <= 1:
        raise ValueError("strategy.position_fraction must be in (0, 1]")
    if positions < 1:
        raise ValueError("maximum_open_positions must be positive")
    if fraction > float(rules["maximum_security_fraction_of_equity"]):
        raise ValueError("Strategy position exceeds the official per-security maximum")
    if fraction * positions > 1:
        raise ValueError("Configured positions deploy more than 100% of equity")
