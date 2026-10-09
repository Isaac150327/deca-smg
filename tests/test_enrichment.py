from deca_smg.enrichment import classify_headlines


def test_headline_classifier_flags_material_catalyst_terms():
    signal, terms = classify_headlines(
        ["Example Bio reports positive Phase 2 clinical trial results"]
    )
    assert signal == "MATERIAL_CATALYST_POSSIBLE"
    assert "clinical trial" in terms
    assert "phase 2" in terms


def test_headline_classifier_does_not_claim_no_catalyst_when_data_is_absent():
    signal, terms = classify_headlines([])
    assert signal == "NO_RECENT_HEADLINES_FOUND"
    assert terms == ""
