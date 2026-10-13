import pandas as pd

from deca_smg.config import load_config
from deca_smg.reporting import shortlist_table


def test_empty_shortlist_contains_dated_no_candidates_message():
    result = shortlist_table(
        pd.DataFrame(),
        load_config("config/default.yaml"),
        as_of_date=pd.Timestamp("2026-10-13"),
    )

    assert len(result) == 1
    assert result.loc[0, "scan_status"] == "NO_QUALIFYING_CANDIDATES"
    assert result.loc[0, "date"] == pd.Timestamp("2026-10-13")
    assert not result.loc[0, "selected"]
    assert "No stocks passed" in result.loc[0, "team_notes"]
