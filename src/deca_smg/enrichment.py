from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


MATERIAL_CATALYST_TERMS = {
    "acquisition",
    "merger",
    "buyout",
    "earnings",
    "revenue",
    "guidance",
    "fda",
    "clinical trial",
    "phase 1",
    "phase 2",
    "phase 3",
    "contract",
    "purchase order",
    "partnership",
    "financing",
    "offering",
    "bankruptcy",
    "settlement",
    "court",
    "patent",
    "dividend",
    "stock split",
}


def classify_headlines(headlines: list[str]) -> tuple[str, str]:
    if not headlines:
        return "NO_RECENT_HEADLINES_FOUND", ""
    joined = " ".join(headlines).lower()
    matches = sorted(term for term in MATERIAL_CATALYST_TERMS if term in joined)
    if matches:
        return "MATERIAL_CATALYST_POSSIBLE", ", ".join(matches)
    return "NO_OBVIOUS_CATALYST_IN_HEADLINES", ""


def _headline_fields(item: dict) -> tuple[str, str]:
    content = item.get("content", item)
    title = str(content.get("title") or "").strip()
    canonical = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
    url = canonical.get("url", "") if isinstance(canonical, dict) else str(canonical)
    return title, url


def _enrich_one(ticker: str, headline_count: int) -> dict:
    result = {
        "ticker": ticker,
        "recent_headlines": "",
        "headline_urls": "",
        "catalyst_signal": "DATA_UNAVAILABLE",
        "catalyst_keywords": "",
        "short_interest_pct": pd.NA,
        "short_ratio": pd.NA,
        "enrichment_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    stock = yf.Ticker(ticker)
    try:
        parsed = [_headline_fields(item) for item in stock.get_news(count=headline_count)]
        parsed = [(title, url) for title, url in parsed if title]
        titles = [title for title, _ in parsed]
        result["recent_headlines"] = " | ".join(titles)
        result["headline_urls"] = " | ".join(url for _, url in parsed if url)
        result["catalyst_signal"], result["catalyst_keywords"] = classify_headlines(titles)
    except Exception:
        pass
    try:
        info = stock.get_info()
        short_fraction = info.get("shortPercentOfFloat")
        if short_fraction is not None:
            result["short_interest_pct"] = 100 * float(short_fraction)
        if info.get("shortRatio") is not None:
            result["short_ratio"] = float(info["shortRatio"])
    except Exception:
        pass
    return result


def enrich_live_candidates(
    candidates: pd.DataFrame, headline_count: int = 5, workers: int = 4
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    tickers = list(dict.fromkeys(candidates["ticker"].astype(str)))
    records = []
    with ThreadPoolExecutor(max_workers=min(workers, len(tickers))) as executor:
        futures = {
            executor.submit(_enrich_one, ticker, headline_count): ticker for ticker in tickers
        }
        for future in as_completed(futures):
            records.append(future.result())
    enrichment = pd.DataFrame(records)
    return candidates.drop(
        columns=[column for column in enrichment.columns if column != "ticker" and column in candidates],
        errors="ignore",
    ).merge(enrichment, on="ticker", how="left")
