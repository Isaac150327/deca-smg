from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SHORTLIST_COLUMNS = [
    "scan_status",
    "date",
    "rank",
    "selected",
    "risk_allowed",
    "ticker",
    "company_name",
    "exchange",
    "close",
    "market_cap",
    "average_daily_volume_3m",
    "return_1d_pct",
    "return_5d_pct",
    "volume_ratio_20",
    "beta_40",
    "realized_vol_20",
    "upper_wick_pct",
    "predicted_short_return_pct",
    "scoring_method",
    "validation_status",
    "selection_reason",
    "heuristic_score",
    "continuation_risk",
    "extreme_move_risk",
    "short_interest_pct",
    "short_ratio",
    "catalyst_signal",
    "catalyst_keywords",
    "recent_headlines",
    "headline_urls",
    "rule_status",
    "catalyst_review",
    "short_interest_review",
    "team_notes",
]


def write_backtest_outputs(
    result,
    sensitivity: pd.DataFrame,
    output_dir: str | Path,
    heuristic_result=None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(output / "trades.csv", index=False)
    result.daily.to_csv(output / "equity_curve.csv", index=False)
    sensitivity.to_csv(output / "cost_sensitivity.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    if heuristic_result is not None:
        heuristic_result.trades.to_csv(output / "heuristic_trades.csv", index=False)
        heuristic_result.daily.to_csv(output / "heuristic_equity_curve.csv", index=False)
        (output / "heuristic_metrics.json").write_text(
            json.dumps(heuristic_result.metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        comparison = pd.DataFrame(
            [
                {"strategy": "walk_forward_model", **result.metrics},
                {"strategy": "fixed_heuristic", **heuristic_result.metrics},
            ]
        )
        comparison.to_csv(output / "model_vs_heuristic.csv", index=False)


def shortlist_table(
    scored: pd.DataFrame,
    config: dict,
    as_of_date=None,
    selection_enabled: bool = True,
    validation_status: str = "NOT_EVALUATED",
) -> pd.DataFrame:
    if scored.empty:
        row = {column: pd.NA for column in SHORTLIST_COLUMNS}
        row.update(
            {
                "scan_status": "NO_QUALIFYING_CANDIDATES",
                "date": pd.to_datetime(as_of_date) if as_of_date is not None else pd.NaT,
                "selected": False,
                "risk_allowed": False,
                "validation_status": validation_status,
                "selection_reason": "NO_QUALIFYING_CANDIDATES",
                "team_notes": "No stocks passed today's configured candidate filters.",
            }
        )
        return pd.DataFrame([row], columns=SHORTLIST_COLUMNS)
    latest_date = scored["date"].max()
    latest = scored.loc[scored["date"].eq(latest_date)].copy()
    latest["scan_status"] = "CANDIDATE"
    latest = latest.sort_values(
        ["predicted_short_return", "heuristic_score"], ascending=[False, False]
    )
    latest["rank"] = range(1, len(latest) + 1)
    max_positions = int(config["strategy"]["maximum_open_positions"])
    latest["eligible_rank"] = pd.NA
    minimum_edge = float(config["risk"]["minimum_predicted_edge"])
    allowed = latest["risk_allowed"].fillna(False) & latest["predicted_short_return"].ge(
        minimum_edge
    )
    latest.loc[allowed, "eligible_rank"] = range(1, int(allowed.sum()) + 1)
    within_position_limit = latest["eligible_rank"].le(max_positions).fillna(False)
    latest["selected"] = selection_enabled & allowed & within_position_limit
    latest["validation_status"] = validation_status
    latest["selection_reason"] = "BELOW_MODEL_EDGE"
    latest.loc[~latest["risk_allowed"].fillna(False), "selection_reason"] = "RISK_GATE_FAILED"
    latest.loc[allowed & ~within_position_limit, "selection_reason"] = "BELOW_TOP_POSITION_LIMIT"
    latest.loc[latest["selected"], "selection_reason"] = "SELECTED"
    if not selection_enabled:
        latest["selection_reason"] = validation_status
    latest["return_1d_pct"] = 100 * latest["return_1d"]
    latest["return_5d_pct"] = 100 * latest["return_5d"]
    scoring_method = (
        latest["scoring_method"]
        if "scoring_method" in latest
        else pd.Series("walk_forward_model", index=latest.index)
    )
    latest["scoring_method"] = scoring_method
    latest["predicted_short_return_pct"] = (
        100 * latest["predicted_short_return"]
    ).where(scoring_method.eq("walk_forward_model"))
    latest["rule_status"] = "CHECK_SMG_SECURITY_TABLE_AND_LOCAL_RULES"
    latest["catalyst_review"] = "REQUIRED"
    latest["short_interest_review"] = "REQUIRED"
    latest["team_notes"] = ""
    if "market_cap" not in latest:
        latest["market_cap"] = pd.NA
    for column in [
        "company_name",
        "exchange",
        "average_daily_volume_3m",
        "short_interest_pct",
        "short_ratio",
        "catalyst_signal",
        "catalyst_keywords",
        "recent_headlines",
        "headline_urls",
    ]:
        if column not in latest:
            latest[column] = pd.NA
    return latest.reindex(columns=SHORTLIST_COLUMNS)
