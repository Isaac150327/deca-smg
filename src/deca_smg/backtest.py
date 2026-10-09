from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily: pd.DataFrame
    metrics: dict[str, float | int]


def _max_drawdown(equity: pd.Series, initial_equity: float) -> float:
    path = pd.concat(
        [pd.Series([initial_equity], dtype=float), equity.reset_index(drop=True)],
        ignore_index=True,
    )
    peak = path.cummax()
    return float((path / peak - 1).min())


def _metrics(trades: pd.DataFrame, daily: pd.DataFrame, initial_equity: float) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "trading_dates": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "average_trade_return": 0.0,
            "sharpe": 0.0,
        }
    daily_std = daily["portfolio_return"].std()
    sharpe = 0.0 if not daily_std or np.isnan(daily_std) else (
        daily["portfolio_return"].mean() / daily_std * np.sqrt(252)
    )
    return {
        "trades": int(len(trades)),
        "trading_dates": int(daily["date"].nunique()),
        "total_return": float(daily["equity"].iloc[-1] / initial_equity - 1),
        "max_drawdown": _max_drawdown(daily["equity"], initial_equity),
        "win_rate": float(trades["net_short_return"].gt(0).mean()),
        "average_trade_return": float(trades["net_short_return"].mean()),
        "average_gross_exposure": float(daily["gross_exposure"].mean()),
        "sharpe": float(sharpe),
    }


def simulate(
    scored: pd.DataFrame,
    config: dict,
    slippage_bps_per_side: float | None = None,
) -> BacktestResult:
    rules = config["game_rules"]
    strategy = config["strategy"]
    risk = config["risk"]
    initial_equity = float(config["backtest"]["initial_equity"])
    max_positions = int(strategy["maximum_open_positions"])
    position_fraction = float(strategy["position_fraction"])
    transaction_fee = float(rules["transaction_fee_per_order"])
    slippage_bps = (
        float(rules["slippage_bps_per_side"])
        if slippage_bps_per_side is None
        else float(slippage_bps_per_side)
    )
    round_trip_slippage = 2 * slippage_bps / 10_000

    eligible = scored.loc[
        scored["risk_allowed"].fillna(False)
        & scored["predicted_short_return"].ge(float(risk["minimum_predicted_edge"]))
        & scored["target_short_return_1d"].notna()
    ].copy()
    eligible = eligible.sort_values(
        ["date", "predicted_short_return", "heuristic_score"],
        ascending=[True, False, False],
    )
    selected = eligible.groupby("date", group_keys=False).head(max_positions).copy()
    selected["rank"] = selected.groupby("date")["predicted_short_return"].rank(
        method="first", ascending=False
    ).astype(int)

    if selected.empty:
        trades = selected
        daily = pd.DataFrame(columns=["date", "portfolio_return", "gross_exposure", "equity"])
        return BacktestResult(trades, daily, _metrics(trades, daily, initial_equity))

    equity = initial_equity
    trade_frames = []
    daily_rows = []
    for trade_date, date_trades in selected.groupby("date", sort=True):
        date_trades = date_trades.copy()
        position_notional = equity * position_fraction
        fixed_fee_return = 2 * transaction_fee / position_notional
        date_trades["gross_short_return"] = date_trades["target_short_return_1d"]
        date_trades["round_trip_cost"] = round_trip_slippage + fixed_fee_return
        date_trades["net_short_return"] = (
            date_trades["gross_short_return"] - date_trades["round_trip_cost"]
        )
        date_trades["position_notional"] = position_notional
        date_trades["portfolio_contribution"] = (
            position_fraction * date_trades["net_short_return"]
        )
        portfolio_return = float(date_trades["portfolio_contribution"].sum())
        equity *= 1 + portfolio_return
        daily_rows.append(
            {
                "date": trade_date,
                "portfolio_return": portfolio_return,
                "positions": len(date_trades),
                "gross_exposure": len(date_trades) * position_fraction,
                "equity": equity,
            }
        )
        trade_frames.append(date_trades)

    trades = pd.concat(trade_frames, ignore_index=True)
    daily = pd.DataFrame(daily_rows)
    return BacktestResult(trades, daily, _metrics(trades, daily, initial_equity))


def cost_sensitivity(scored: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for cost in config["backtest"]["slippage_sensitivity_bps_per_side"]:
        result = simulate(scored, config, slippage_bps_per_side=float(cost))
        rows.append({"slippage_bps_per_side": float(cost), **result.metrics})
    return pd.DataFrame(rows)
