from __future__ import annotations

from datetime import time
from pathlib import Path
import warnings

import pandas as pd
import yfinance as yf


warnings.filterwarnings(
    "ignore",
    message=r"The 'generic' unit for NumPy timedelta is deprecated.*",
    category=DeprecationWarning,
    module=r"yfinance\..*",
)


def read_tickers(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        for column in ["symbol", "ticker"]:
            if column in frame.columns:
                values = frame[column].dropna().astype(str).str.upper().str.strip()
                return list(dict.fromkeys(value.replace(".", "-") for value in values if value))
        raise ValueError("Universe CSV must contain a symbol or ticker column")

    tickers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip().upper()
        if value and not value.startswith("#"):
            tickers.append(value.replace(".", "-"))
    return list(dict.fromkeys(tickers))


def discover_universe(
    minimum_price: float,
    maximum_price: float,
    minimum_market_cap: float,
    maximum_market_cap: float,
    minimum_volume: int = 100_000,
    maximum_tickers: int = 1_000,
) -> pd.DataFrame:
    """Discover a current U.S. micro/small-cap research universe from Yahoo.

    The result is point-in-time only. Saving it daily is important because using
    today's survivor list for old dates creates survivorship bias.
    """

    query = yf.EquityQuery(
        "and",
        [
            yf.EquityQuery("eq", ["region", "us"]),
            yf.EquityQuery("is-in", ["exchange", "NMS", "NGM", "NCM", "NYQ"]),
            yf.EquityQuery("gte", ["intradayprice", minimum_price]),
            yf.EquityQuery("lte", ["intradayprice", maximum_price]),
            yf.EquityQuery("gte", ["intradaymarketcap", minimum_market_cap]),
            yf.EquityQuery("lte", ["intradaymarketcap", maximum_market_cap]),
            yf.EquityQuery("gte", ["dayvolume", minimum_volume]),
        ],
    )
    quotes: list[dict] = []
    offset = 0
    while offset < maximum_tickers:
        page_size = min(250, maximum_tickers - offset)
        response = yf.screen(
            query,
            offset=offset,
            size=page_size,
            sortField="intradaymarketcap",
            sortAsc=True,
        )
        page = response.get("quotes", [])
        if not page:
            break
        quotes.extend(page)
        offset += len(page)
        total = int(response.get("total", len(quotes)))
        if offset >= total or len(page) < page_size:
            break

    if not quotes:
        raise RuntimeError("Yahoo returned no symbols for the configured universe")
    frame = pd.DataFrame(quotes)
    if "symbol" not in frame.columns:
        raise RuntimeError("Yahoo's screener response did not contain symbols")
    keep = [
        column
        for column in [
            "symbol",
            "shortName",
            "exchange",
            "regularMarketPrice",
            "regularMarketChangePercent",
            "regularMarketVolume",
            "averageDailyVolume3Month",
            "marketCap",
        ]
        if column in frame.columns
    ]
    frame = frame[keep].drop_duplicates("symbol").copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    return frame.sort_values("symbol").reset_index(drop=True)


def _extract_ticker_frame(download: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(download.columns, pd.MultiIndex):
        return download.copy()
    if ticker in download.columns.get_level_values(0):
        return download[ticker].copy()
    if ticker in download.columns.get_level_values(1):
        return download.xs(ticker, axis=1, level=1).copy()
    return pd.DataFrame()


def fetch_snapshots(
    tickers: list[str],
    period: str = "60d",
    interval: str = "5m",
    cutoff: str = "15:40",
    timezone: str = "America/New_York",
    batch_size: int = 100,
) -> pd.DataFrame:
    """Download intraday bars and aggregate information available by cutoff."""

    symbols = list(dict.fromkeys([symbol.upper() for symbol in tickers] + ["SPY"]))
    cutoff_time = time.fromisoformat(cutoff)
    bar_duration = pd.Timedelta(interval)
    pieces = []
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        raw = yf.download(
            batch,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            prepost=False,
            threads=True,
            progress=False,
        )
        if raw.empty:
            continue
        for symbol in batch:
            frame = _extract_ticker_frame(raw, symbol)
            if frame.empty:
                continue
            frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
            frame = frame.rename(columns={"adj_close": "adjusted_close"})
            if frame.index.tz is None:
                frame.index = frame.index.tz_localize("UTC")
            frame.index = frame.index.tz_convert(timezone)
            frame["date"] = frame.index.tz_localize(None).normalize()
            execution = frame.groupby("date", as_index=False).agg(
                execution_close=("close", "last")
            )
            completed_by_cutoff = [
                (timestamp + bar_duration).time() <= cutoff_time for timestamp in frame.index
            ]
            snapshot = frame.loc[completed_by_cutoff].copy()
            if snapshot.empty:
                continue
            grouped = snapshot.groupby("date", as_index=False).agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            grouped = grouped.merge(execution, on="date", how="left")
            grouped["ticker"] = symbol
            pieces.append(grouped)
    if not pieces:
        raise RuntimeError("No bars were available before the configured cutoff")
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "ticker"])


def read_prices(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_prices(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
