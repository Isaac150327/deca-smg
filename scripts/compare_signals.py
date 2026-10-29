from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


NEW_YORK = ZoneInfo("America/New_York")


def _ticker_frame(download: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(download.columns, pd.MultiIndex):
        return download.copy()
    if ticker in download.columns.get_level_values(0):
        return download[ticker].copy()
    if ticker in download.columns.get_level_values(1):
        return download.xs(ticker, axis=1, level=1).copy()
    return pd.DataFrame()


def compare(signals: pd.DataFrame) -> pd.DataFrame:
    tickers = list(dict.fromkeys(signals["ticker"].astype(str).str.upper()))
    download = yf.download(
        tickers,
        period="5d",
        interval="5m",
        auto_adjust=False,
        prepost=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    evaluated_at = datetime.now(NEW_YORK)
    rows = []
    for signal in signals.itertuples(index=False):
        frame = _ticker_frame(download, signal.ticker).dropna(subset=["Close"])
        if frame.empty:
            rows.append({**signal._asdict(), "outcome_status": "NO_PRICE_DATA"})
            continue
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        frame.index = frame.index.tz_convert(NEW_YORK)
        signal_date = pd.Timestamp(signal.date).date()
        entry_rows = frame[frame.index.date == signal_date]
        later_rows = frame[frame.index.date > signal_date]
        if entry_rows.empty or later_rows.empty:
            rows.append({**signal._asdict(), "outcome_status": "PENDING"})
            continue

        exit_date = later_rows.index[0].date()
        exit_rows = later_rows[later_rows.index.date == exit_date]
        entry_price = float(entry_rows["Close"].iloc[-1])
        exit_price = float(exit_rows["Close"].iloc[-1])
        gross_return = 1 - exit_price / entry_price
        complete = evaluated_at.date() > exit_date or evaluated_at.time().hour >= 16
        rows.append(
            {
                **signal._asdict(),
                "entry_price": entry_price,
                "exit_date": exit_date.isoformat(),
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
