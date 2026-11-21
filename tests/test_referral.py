from datetime import datetime, timezone

from careproof.referral import (BLOCKING_STATUSES, STATUSES, availability_label, checklist, next_alternative,
                                referral_summary)

R = [{"facility_id": f"f{i}", "facility_name": f"Hospital {i}", "distance_km": float(i), "evidence_label": "Moderate documented evidence",
      "evidence_field": "procedure", "evidence_excerpt": f"Dialysis unit {i}", "address": "Patna", "source_url": None,
      "identity_warnings": [], "location_warnings": [], "missing_data_warnings": [], "pipeline_version": "v"} for i in range(1, 4)]


def test_checklists_are_service_specific():
    assert any("anaesthetist" in q for q in checklist("c_section"))
    assert any("NICU bed" in q for q in checklist("nicu"))
    assert all(q in checklist("mri") for q in checklist("unknown_service"))


def test_next_alternative_skips_blocked():
    assert next_alternative(R, {})["facility_id"] == "f1"
    assert next_alternative(R, {"f1": "Unreachable", "f2": "Service contradicted"})["facility_id"] == "f3"
    assert next_alternative(R, {f["facility_id"]: "Not accepting referrals" for f in R}) is None
    assert next_alternative(R, {}, exclude="f1")["facility_id"] == "f2"


def test_statuses_and_labels():
    assert set(BLOCKING_STATUSES) < set(STATUSES)
    assert availability_label("Not contacted") == "Current availability not verified"
    assert "session only" in availability_label("Confirmed")


def test_summary_contents_and_no_patient_fields():
    txt = referral_summary("Dialysis", "PIN 800001", R[0], R, "Confirmed", {q: i == 0 for i, q in enumerate(checklist("dialysis"))},
                           now=datetime(2026, 11, 21, 10, 0, tzinfo=timezone.utc))
    for s in ("Dialysis", "PIN 800001", "Hospital 1", '"Dialysis unit 1"', "CALL-AHEAD CHECKLIST", "[x]", "Hospital 2",
              "2026-11-21 10:00 UTC", "does not provide medical advice", "not road distance"):
        assert s in txt
    assert "patient name" not in txt.lower() and "age:" not in txt.lower()
