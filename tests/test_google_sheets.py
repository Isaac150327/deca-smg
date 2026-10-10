import json

from deca_smg.google_sheets import publish_csv


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return b'{"ok": true, "worksheet": "Shortlist", "rows": 1}'


def test_publish_csv_sends_rows_and_token(tmp_path, monkeypatch):
    csv_path = tmp_path / "shortlist.csv"
    csv_path.write_text("ticker,score\nABC,0.5\n", encoding="utf-8")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("deca_smg.google_sheets.urlopen", fake_urlopen)
    result = publish_csv(csv_path, "https://example.test/exec", "private-token")

    assert captured["payload"] == {
        "token": "private-token",
        "worksheet": "Shortlist",
        "rows": [["ticker", "score"], ["ABC", "0.5"]],
    }
    assert captured["timeout"] == 30
    assert result["rows"] == 1
