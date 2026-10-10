from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def publish_csv(
    input_path: str | Path,
    webhook_url: str,
    token: str,
    worksheet: str = "Shortlist",
    timeout: int = 30,
) -> dict:
    """Replace a Google Sheet tab through the project's Apps Script webhook."""

    path = Path(input_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")

    payload = json.dumps(
        {"token": token, "worksheet": worksheet, "rows": rows}
    ).encode("utf-8")
    request = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Google Sheets webhook request failed: {error}") from error

    try:
        result = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError("Google Sheets webhook returned an invalid response") from error
    if not result.get("ok"):
        raise RuntimeError(
            f"Google Sheets webhook rejected the update: {result.get('error', 'unknown error')}"
        )
    return result
