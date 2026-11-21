from types import SimpleNamespace

import pandas as pd

from careproof.credibility import (description_subject_conflict, other_organisations, same_entity,
                                   third_party_language)
from careproof.pipeline import facility_findings, identity_confidence, location_confidence


def row(**kw):
    base = dict(name="City Hospital", description="City Hospital is a 50-bed hospital.", capability="null",
                address_city="Patna", address_stateOrRegion="Bihar", facility_to_pin_km=1.0, pincode_norm="800001",
                state_conflict=False, pin_states_json='["BIHAR"]', location_source="facility_coordinates",
                location_lat=25.61, location_lon=85.14, facility_lat=25.61, facility_lon=85.14,
                pin_n_states=1, pin_n_districts=1, possible_duplicate=False,
                source_urls_list_json='["https://x.org"]', officialWebsite="x.org")
    base.update(kw)
    return SimpleNamespace(**base)


def codes(r, pin_xy=None, coord_names=None):
    return {(f["code"], f["severity"]) for f in facility_findings(r, pin_xy or {}, coord_names or {})}


def test_clean_record_has_no_findings():
    assert codes(row()) == set()


def test_multiple_facility_identities():
    assert ("MULTIPLE_FACILITY_IDENTITIES", "high") in codes(
        row(name="Saifee Hospital Mumbai India", description="Prince Aly Khan Hospital is located in Mumbai, India."))


def test_name_variants_are_not_flagged():
    for own, desc in [("Life Line Hospital", "Lifeline Hospital is a multi-speciality hospital."),
                      ("Ortho Plus Hospital", "Orthoplus Hospital in Najafgarh"),
                      ("Udyan Hospital", "Udayan Hospital is a trauma centre.")]:
        assert description_subject_conflict(desc, own) is None


def test_coordinate_conflict_severity_by_distance():
    assert ("COORDINATE_LOCATION_CONFLICT", "medium") in codes(row(facility_to_pin_km=60.0))
    assert ("COORDINATE_LOCATION_CONFLICT", "high") in codes(row(facility_to_pin_km=600.0))


def test_state_conflict_is_incompatible_location():
    assert ("MULTIPLE_INCOMPATIBLE_LOCATIONS", "high") in codes(row(state_conflict=True))


def test_distant_pin_mentioned_in_text():
    c = codes(row(description="Branch at Pune 411001 and City Hospital Patna"), pin_xy={"411001": (18.52, 73.85)})
    assert ("MULTIPLE_INCOMPATIBLE_LOCATIONS", "medium") in c


def test_ambiguous_pin_and_duplicate_and_unresolved():
    assert ("AMBIGUOUS_PIN", "medium") in codes(row(pin_n_states=2))
    assert ("AMBIGUOUS_PIN", "low") in codes(row(pin_n_districts=3))
    assert ("POSSIBLE_DUPLICATE", "low") in codes(row(possible_duplicate=True))
    assert ("UNRESOLVED_LOCATION", "high") in codes(row(location_source="none", location_lat=None))


def test_shared_placeholder_coordinate():
    cn = {(25.61, 85.14): 4}
    assert ("SHARED_PLACEHOLDER_COORDINATE", "medium") in codes(row(), coord_names=cn)


def test_missing_provenance():
    assert ("MISSING_SOURCE_PROVENANCE", "medium") in codes(row(source_urls_list_json="[]", officialWebsite=None))


def test_confidence_scores_respond_to_findings():
    hi = facility_findings(row(name="Saifee Hospital", description="Prince Aly Khan Hospital is in Mumbai."), {}, {})
    assert identity_confidence(hi) < 0.5 and identity_confidence([]) == 1.0
    loc = facility_findings(row(facility_to_pin_km=600.0), {}, {})
    assert location_confidence(1.0, loc) <= 0.2


def test_attribution_helpers():
    assert same_entity("Hero DMC Heart Institute", "DMC Hospital")
    assert not same_entity("Srp Hospital", "SPMM Hospital")
    assert other_organisations("Offers dialysis services at SPMM Hospital.", "Srp Hospital") == ["SPMM Hospital"]
    assert third_party_language("cases are referred to the district hospital")
    assert third_party_language("Tie-Up Hospital (ESIC) providing dialysis") is None
