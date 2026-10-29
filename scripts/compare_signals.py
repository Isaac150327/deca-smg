from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from deca_smg.features import add_heuristic_score, build_feature_panel
from deca_smg.market_data import fetch_snapshots


NEW_YORK = ZoneInfo("America/New_York")


def compare(signals: pd.DataFrame) -> pd.DataFrame:
    tickers = list(dict.fromkeys(signals["ticker"].astype(str).str.upper()))
    snapshots = fetch_snapshots(tickers, period="60d", interval="5m")
    snapshots["date"] = pd.to_datetime(snapshots["date"])
    panel = add_heuristic_score(build_feature_panel(snapshots))
    evaluated_at = datetime.now(NEW_YORK)
    rows = []
    for signal in signals.itertuples(index=False):
        frame = snapshots.loc[snapshots["ticker"].eq(signal.ticker)].sort_values("date")
        if frame.empty:
            rows.append({**signal._asdict(), "outcome_status": "NO_PRICE_DATA"})
            continue
        signal_date = pd.Timestamp(signal.date).normalize()
        entry_rows = frame.loc[frame["date"].eq(signal_date)]
        later_rows = frame.loc[frame["date"].gt(signal_date)]
        if entry_rows.empty or later_rows.empty:
            rows.append({**signal._asdict(), "outcome_status": "PENDING"})
            continue

        exit_date = later_rows["date"].iloc[0]
        exit_rows = later_rows.loc[later_rows["date"].eq(exit_date)]
        entry_price = float(entry_rows["execution_close"].iloc[-1])
        exit_price = float(exit_rows["execution_close"].iloc[-1])
        gross_return = 1 - exit_price / entry_price
        feature_rows = panel.loc[
            panel["ticker"].eq(signal.ticker) & panel["date"].eq(signal_date)
        ]
        feature_values = {}
        if not feature_rows.empty:
            feature = feature_rows.iloc[-1]
            feature_values = {
                "signal_return_1d_pct": 100 * feature["return_1d"],
                "signal_return_5d_pct": 100 * feature["return_5d"],
                "signal_volume_ratio": feature["volume_ratio_20"],
                "signal_close_location": feature["close_location"],
                "signal_upper_wick_pct": 100 * feature["upper_wick_pct"],
                "signal_heuristic_score": feature["heuristic_score"],
                "signal_continuation_risk": feature["continuation_risk"],
                "signal_extreme_move_risk": feature["extreme_move_risk"],
            }
        complete = evaluated_at.date() > exit_date.date() or evaluated_at.time().hour >= 16
        rows.append(
            {
                **signal._asdict(),
                **feature_values,
                "entry_price": entry_price,
                "exit_date": exit_date.date().isoformat(),
                "exit_price": exit_price,
                "gross_short_return_pct": 100 * gross_return,
                "outcome_status": "FINAL" if complete else "PROVISIONAL_INTRADAY",
                "evaluated_at_et": evaluated_at.isoformat(timespec="seconds"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    signals = pd.read_csv(args.signals)
    result = compare(signals)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
