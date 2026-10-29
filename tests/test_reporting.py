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


def test_unvalidated_scoring_cannot_emit_selected_true():
    scored = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-10-29"),
                "ticker": "PUMP",
                "risk_allowed": True,
                "predicted_short_return": 0.90,
                "heuristic_score": 0.90,
                "return_1d": 0.20,
                "return_5d": 0.30,
                "scoring_method": "fixed_heuristic",
            }
        ]
    )

    result = shortlist_table(
        scored,
        load_config("config/default.yaml"),
        selection_enabled=False,
        validation_status="HEURISTIC_RESEARCH_ONLY",
    )

    assert not result.loc[0, "selected"]
    assert result.loc[0, "selection_reason"] == "HEURISTIC_RESEARCH_ONLY"
