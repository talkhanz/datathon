"""End-to-end tests on a tiny synthetic dataset: silver -> gold -> recommend (same path as Databricks)."""
import json

import pandas as pd
import pytest

from careproof import pipeline as P
from careproof.geography import ResolvedLocation
from careproof.service_matching import count_distinct_statements

COLS = ["unique_id", "name", "description", "source_urls", "specialties", "procedure", "equipment", "capability",
        "address_line1", "address_city", "address_stateOrRegion", "address_zipOrPostcode", "latitude", "longitude",
        "facilityTypeId", "officialWebsite", "websites"]


def fac(i, name, lat, lon, procedure="null", pin="800001", description="General hospital.", specialties="null",
        capability="null"):
    return {"unique_id": f"00000000-0000-0000-0000-{i:012d}", "name": name, "description": description,
            "source_urls": '["https://example.org/a"]', "specialties": specialties, "procedure": procedure,
            "equipment": "null", "capability": capability, "address_line1": "Road", "address_city": "Patna",
            "address_stateOrRegion": "Bihar", "address_zipOrPostcode": pin, "latitude": str(lat),
            "longitude": str(lon), "facilityTypeId": "hospital", "officialWebsite": "null", "websites": "[]"}


def make_synthetic():
    """Tiny synthetic dataset -> (silver, rejected_rows, gold). Shared with tests/test_inference.py."""
    pins = pd.DataFrame([{"circlename": "", "regionname": "", "divisionname": "", "officename": f"O{i}",
                          "pincode": "800001", "officetype": "PO", "delivery": "Delivery", "district": "PATNA",
                          "statename": "BIHAR", "latitude": str(25.61 + i * 0.001), "longitude": "85.14"}
                         for i in range(5)])
    pin_silver = P.build_pincode_silver(pins)
    rows = [
        fac(1, "Near Weak Clinic", 25.612, 85.141, procedure='["Dialysis"]'),
        fac(2, "Far Strong Dialysis Centre", 25.70, 85.20, procedure='["Haemodialysis unit with 10 machines, 24x7"]',
            description="Dedicated dialysis centre offering haemodialysis.", specialties='["nephrology"]'),
        fac(3, "Twin Hospital", 25.62, 85.15, procedure='["Dialysis unit"]'),
        fac(4, "Twin Hospital", 25.6205, 85.1505, procedure='["Dialysis services"]'),
        fac(5, "Mid Hospital", 25.65, 85.17, capability='["Dialysis available"]', procedure='["Dialysis"]'),
        fac(6, "Negated Hospital", 25.611, 85.140, capability='["Dialysis not available"]'),
        fac(7, "Tag Only Hospital", 25.613, 85.142, specialties='["nephrology"]'),
        {**fac(8, "Broken", 0, 0), "unique_id": "| **fragment**"},
    ]
    raw = pd.DataFrame(rows)[COLS]
    silver, quarantine, rep = P.build_facility_silver(raw, pin_silver)
    gold = P.build_facility_service_gold(silver, ["dialysis"])
    return silver, quarantine, gold


@pytest.fixture(scope="module")
def built():
    return make_synthetic()


LOC = ResolvedLocation(25.61, 85.14, "test", "user_coordinates")


def test_quarantine_and_gold_levels(built):
    silver, quarantine, gold = built
    assert len(quarantine) == 1 and len(silver) == 7
    lv = dict(zip(gold["name"] + "|" + gold["facility_id"].str[-1], gold["evidence_level"]))
    assert lv["Negated Hospital|6"] == "negated_only" and lv["Tag Only Hospital|7"] == "specialty_tag_only"


def test_radius_only_filters_never_reorders(built):
    _, _, gold = built
    small, _ = P.recommend(gold, LOC, "dialysis", 5, 50)
    large, _ = P.recommend(gold, LOC, "dialysis", 100, 50)
    small_ids = [r["facility_id"] for r in small]
    large_within = [r["facility_id"] for r in large if r["distance_km"] <= 5]
    assert small_ids == large_within
    for a in small:
        b = next(x for x in large if x["facility_id"] == a["facility_id"])
        assert a["final_ranking_score"] == b["final_ranking_score"]


def test_excluded_levels_and_abstention(built):
    _, _, gold = built
    res, summ = P.recommend(gold, LOC, "dialysis", 100, 10)
    names = {r["facility_name"] for r in res}
    assert "Negated Hospital" not in names and "Tag Only Hospital" not in names
    assert summ["status"] == P.ANSWERED
    none, summ2 = P.recommend(gold, ResolvedLocation(20.0, 80.0, "far", "user_coordinates"), "dialysis", 10, 5)
    assert none == [] and summ2["status"] == P.ABSTAIN_NO_MENTIONS and summ2["message"]


def test_duplicates_collapsed(built):
    _, _, gold = built
    res, _ = P.recommend(gold, LOC, "dialysis", 100, 10)
    twins = [r for r in res if r["facility_name"] == "Twin Hospital"]
    assert len(twins) == 1 and len(twins[0]["duplicate_records_collapsed"]) == 1
    assert any("near-identical record" in w for w in twins[0]["missing_data_warnings"])


def test_every_result_cites_exact_text(built):
    silver, _, gold = built
    by_id = silver.set_index("facility_id").to_dict("index")
    from careproof.evaluation import excerpt_is_exact
    res, _ = P.recommend(gold, LOC, "dialysis", 100, 10)
    assert res and all(excerpt_is_exact(by_id[r["facility_id"]], r["evidence_field"], r["evidence_excerpt"]) for r in res)


def test_ranking_deterministic(built):
    _, _, gold = built
    a, _ = P.recommend(gold, LOC, "dialysis", 100, 10)
    b, _ = P.recommend(gold.sample(frac=1, random_state=7), LOC, "dialysis", 100, 10)
    assert [r["facility_id"] for r in a] == [r["facility_id"] for r in b]


def test_distinct_statements():
    assert count_distinct_statements(["NICU", "NICU.", "Neonatal unit"]) == 2
    assert count_distinct_statements(["Neonatal Intensive Care Unit (NICU)",
                                      "Neonatal Intensive Care Unit (NICU) with incubators"]) == 1
    assert count_distinct_statements([]) == 0


def test_abstains_when_only_ineligible_mentions(built):
    """Records mention the service nearby but none pass the gates -> explicit abstention, never a weak fallback."""
    _, _, gold = built
    g = gold.copy()
    g["eligible"] = False
    res, summ = P.recommend(g, LOC, "dialysis", 100, 5)
    assert res == [] and summ["status"] == P.ABSTAIN_NO_ELIGIBLE
    assert summ["message"] == P.ABSTENTION_MESSAGE and summ["held_back"]


def test_distance_never_rescues_ineligible(built):
    """The nearest record (Near Weak Clinic) is made ineligible; it must not appear however close it is."""
    _, _, gold = built
    g = gold.copy()
    near_id = g[g["name"] == "Near Weak Clinic"]["facility_id"].iloc[0]
    g.loc[g["facility_id"] == near_id, "eligible"] = False
    res, summ = P.recommend(g, LOC, "dialysis", 100, 10)
    assert near_id not in [r["facility_id"] for r in res]
    assert near_id in [h["facility_id"] for h in summ["held_back"]]


def test_alternatives_returned(built):
    _, _, gold = built
    res, summ = P.recommend(gold, LOC, "dialysis", 100, 2)
    assert len(res) == 2 and summ["additional_eligible_beyond_top_k"] >= 1


def test_quarantine_table(built):
    silver, rejected, gold = built
    q = P.build_quarantine(silver, gold, rejected)
    assert {"source_row_id", "reason_code", "severity", "evidence", "human_review_required", "pipeline_version"} <= set(q.columns)
    assert "CORRUPTED_SOURCE_ROW" in set(q.reason_code)
    assert "SERVICE_CONTRADICTION" in set(q.reason_code)  # the "Dialysis not available" record


def test_exact_quotation_preserved(built):
    silver, _, gold = built
    res, _ = P.recommend(gold, LOC, "dialysis", 100, 10)
    for r in res:
        src = silver.set_index("facility_id").loc[r["facility_id"]]
        assert r["evidence_excerpt"] in (src[r["evidence_field"]] or "")


def test_spark_values_use_null_not_nan():
    """Regression: NaN must become None (SQL NULL); found when Databricks stored NaN DOUBLEs."""
    import numpy as np
    kind, vals = P.spark_column_values(pd.Series([1.5, np.nan, None]))
    assert kind == "double" and vals == [1.5, None, None]
    kind, vals = P.spark_column_values(pd.Series([True, None, False], dtype=object))
    assert kind == "boolean" and vals == [True, None, False]
    kind, vals = P.spark_column_values(pd.Series(["a", None, np.nan], dtype=object))
    assert kind == "string" and vals == ["a", None, None]
    assert P.spark_column_values(pd.Series([1, 2]))[0] == "long"


def test_unrecognised_state_is_not_a_state_conflict():
    """Regression: a city in the state field (state_norm missing -> NaN after join) must not count as a conflict."""
    pins = pd.DataFrame([{"circlename": "", "regionname": "", "divisionname": "", "officename": f"O{i}",
                          "pincode": "400103", "officetype": "PO", "delivery": "Delivery", "district": "MUMBAI",
                          "statename": "MAHARASHTRA", "latitude": "19.20", "longitude": "72.85"} for i in range(3)])
    rows = [{**fac(1, "Karuna Hospital", 19.201, 72.851, procedure='["Dialysis"]', pin="400103"),
             "address_stateOrRegion": "Mumbai"},
            {**fac(2, "Real Conflict Hospital", 19.202, 72.852, procedure='["Dialysis"]', pin="400103"),
             "address_stateOrRegion": "Gujarat"}]
    silver, _, _ = P.build_facility_silver(pd.DataFrame(rows)[COLS], P.build_pincode_silver(pins))
    sc = dict(zip(silver["name"], silver["state_conflict"]))
    assert sc["Karuna Hospital"] is False or sc["Karuna Hospital"] == False  # noqa: E712
    assert bool(sc["Real Conflict Hospital"]) is True
