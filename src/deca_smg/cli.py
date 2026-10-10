from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .backtest import cost_sensitivity, simulate
from .config import load_config
from .enrichment import enrich_live_candidates
from .features import add_heuristic_score, apply_risk_gate, build_feature_panel, candidate_mask
from .market_data import (
    discover_universe,
    fetch_snapshots,
    read_prices,
    read_tickers,
    write_prices,
)
from .model import FEATURE_COLUMNS, score_live, walk_forward_score
from .reporting import shortlist_table, write_backtest_outputs


def _candidate_panel(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    panel = add_heuristic_score(build_feature_panel(prices))
    candidates = panel.loc[candidate_mask(panel, config)].copy()
    candidates["risk_allowed"] = apply_risk_gate(candidates, config)
    if candidates.empty:
        raise ValueError("No rows passed the configured pump-candidate filters")
    return candidates


def command_fetch(args: argparse.Namespace) -> None:
    tickers = read_tickers(args.tickers)
    prices = fetch_snapshots(
        tickers,
        period=args.period,
        interval=args.interval,
        cutoff=args.cutoff,
    )
    write_prices(prices, args.output)
    print(f"Saved {len(prices):,} snapshot rows for {prices['ticker'].nunique():,} tickers")


def command_discover(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    filters = config["candidate_filter"]
    rules = config["game_rules"]
    universe = discover_universe(
        minimum_price=float(rules["minimum_price"]),
        maximum_price=float(filters["maximum_price"]),
        minimum_market_cap=float(filters["minimum_market_cap"]),
        maximum_market_cap=float(filters["maximum_market_cap"]),
        minimum_volume=args.minimum_volume,
        maximum_tickers=args.maximum_tickers,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output, index=False)
    print(f"Saved {len(universe):,} current U.S. symbols to {output}")


def command_backtest(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    candidates = _candidate_panel(read_prices(args.input), config)
    walk_forward = walk_forward_score(candidates, config)
    result = simulate(walk_forward.scored, config)
    heuristic_scored = walk_forward.scored.copy()
    heuristic_scored["predicted_short_return"] = heuristic_scored["heuristic_score"]
    heuristic_result = simulate(heuristic_scored, config)
    sensitivity = cost_sensitivity(walk_forward.scored, config)
    write_backtest_outputs(result, sensitivity, args.output_dir, heuristic_result)
    model_path = Path(args.output_dir) / "model.joblib"
    joblib.dump(
        {
            "model": walk_forward.final_model,
            "features": FEATURE_COLUMNS,
            "trained_through": str(candidates.loc[candidates["target_short_return_1d"].notna(), "date"].max()),
        },
        model_path,
    )
    print(pd.Series(result.metrics).to_string())
    print(f"Saved backtest artifacts to {args.output_dir}")


def command_scan(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = read_prices(args.input)
    latest_market_date = pd.to_datetime(prices["date"]).dt.tz_localize(None).max()
    candidates = _candidate_panel(prices, config)
    candidates = candidates.loc[candidates["date"].eq(latest_market_date)].copy()
    universe_path = Path(args.universe)
    if not candidates.empty and universe_path.exists():
        universe = pd.read_csv(universe_path).rename(
            columns={
                "symbol": "ticker",
                "shortName": "company_name",
                "marketCap": "market_cap",
                "averageDailyVolume3Month": "average_daily_volume_3m",
            }
        )
        metadata_columns = [
            column
            for column in [
                "ticker",
                "company_name",
                "exchange",
                "market_cap",
                "average_daily_volume_3m",
            ]
            if column in universe.columns
        ]
        candidates = candidates.drop(
            columns=[column for column in metadata_columns if column != "ticker" and column in candidates],
            errors="ignore",
        ).merge(universe[metadata_columns], on="ticker", how="left")
    if not candidates.empty and not args.skip_live_enrichment:
        candidates = enrich_live_candidates(candidates)
        candidates["risk_allowed"] = apply_risk_gate(candidates, config)
    artifact = joblib.load(args.model)
    scored = score_live(artifact["model"], candidates) if not candidates.empty else candidates
    shortlist = shortlist_table(scored, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(output, index=False)
    print(shortlist.to_string(index=False))
    print(f"Saved {len(shortlist):,} candidates to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deca-smg")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser(
        "discover", help="Build a current U.S. micro/small-cap universe"
    )
    discover_parser.add_argument("--config", default="config/default.yaml")
    discover_parser.add_argument("--output", default="data/universe.csv")
    discover_parser.add_argument("--minimum-volume", type=int, default=100_000)
    discover_parser.add_argument("--maximum-tickers", type=int, default=1_000)
    discover_parser.set_defaults(function=command_discover)

    fetch_parser = subparsers.add_parser("fetch", help="Download 3:40 ET Yahoo snapshots")
    fetch_parser.add_argument("--tickers", required=True, help="One ticker per line")
    fetch_parser.add_argument("--output", default="data/snapshots.csv")
    fetch_parser.add_argument("--period", default="60d")
    fetch_parser.add_argument("--interval", default="5m")
    fetch_parser.add_argument("--cutoff", default="15:40")
    fetch_parser.set_defaults(function=command_fetch)

    backtest_parser = subparsers.add_parser("backtest", help="Run a walk-forward backtest")
    backtest_parser.add_argument("--input", required=True)
    backtest_parser.add_argument("--config", default="config/default.yaml")
    backtest_parser.add_argument("--output-dir", default="outputs/backtest")
    backtest_parser.set_defaults(function=command_backtest)

    scan_parser = subparsers.add_parser("scan", help="Write the latest ranked CSV")
    scan_parser.add_argument("--input", required=True)
    scan_parser.add_argument("--model", default="outputs/backtest/model.joblib")
    scan_parser.add_argument("--universe", default="data/universe.csv")
    scan_parser.add_argument(
        "--skip-live-enrichment",
        action="store_true",
        help="Skip current Yahoo headlines and short-interest fields",
    )
    scan_parser.add_argument("--config", default="config/default.yaml")
    scan_parser.add_argument("--output", default="outputs/shortlist.csv")
    scan_parser.set_defaults(function=command_scan)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
