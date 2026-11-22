"""External PM-JAY reference: precision-first matching, speciality mapping, reference assembly, agreement maths."""
import pandas as pd
import pytest

from careproof import external as E


def test_core_name_strips_legal_suffix_and_generic_words():
    assert E.core_name("Nidaan Hospital A Unit Of Narsingh Hosp Heart Ins Pvt Ltd") == "nidaan"
    assert E.core_name("Supreme Hospital Private Limited") == "supreme"
    assert E.core_name("Civil Hospital", {"rohtak"}) == ""          # generic names never match


@pytest.mark.parametrize("a,b,ok", [
    ("Aggarwal Heart And Surgical Hospital", "Aggarwal Heart & Surgical Hospital", True),
    ("PRADEEP KANSAL MULTISPECIALITY HOSPITAL", "Pardeep Kansal Multi Speciality Hospital", True),
    ("Shivam Hospital Unit Of R R Hospitals Pvt Ltd", "Shiva Hospital", False),          # reviewed false match
    ("Meenakshi Jain Hospital", "Jain Free Eye Hospital, Jind", False),
    ("National Institute Of Medical Sciences", "Asian Institute of Medical Sciences", False),
    ("Civil Hospital Rohtak", "Civil Hospital Rohtak", False),                           # generic: empty core
])
def test_names_match_is_precision_first(a, b, ok):
    assert E.names_match(a, b, {"rohtak", "jind"}) is ok


def test_specialty_mapping():
    assert E.empanelled_for("General Medicine, Neo- Natal Care Packages", "nicu")
    assert E.empanelled_for("Polytrauma, Urology", "trauma_emergency")
    assert E.empanelled_for("Treatment Procedure Burns And Plastic Surgery", "burn_care")
    assert not E.empanelled_for("Cardiology, General Surgery", "cardiac_surgery")       # cardiology != surgery
    assert E.empanelled_for("Cardio-Thoracic & Vascular Surgery", "cardiac_surgery")
    assert not E.empanelled_for("Primary Care", "mri") and E.empanelled_for("X - Ray, Mri", "mri")


def test_build_reference_merges_lists_and_status():
    spec = pd.DataFrame({"hospital_id": ["H1", "H2"], "hospital_name": ["A", "B"], "hospital_type": "Private",
                         "district": "X", "specialities_raw": ["Polytrauma", "General Medicine"],
                         "list_date": "26/06/2025", "list_type": "empanelled_specialities"})
    cur = pd.DataFrame({"hospital_id": ["H1", "H3"], "hospital_name": ["A", "C"], "hospital_type": "Private",
                        "district": "X", "specialities_raw": None, "list_date": "03/08/2026", "list_type": "empanelled_current"})
    sus = pd.DataFrame({"hospital_id": ["H2"], "hospital_name": [None], "hospital_type": "Private", "district": "X",
                        "specialities_raw": None, "list_date": "03/08/2026", "list_type": "suspended"})
    ref = E.build_reference({"a": spec, "b": cur, "c": sus}).set_index("hospital_id")
    assert set(ref.index) == {"H1", "H2", "H3"}
    assert ref.loc["H1", "on_current_list"] and not ref.loc["H2", "on_current_list"]
    assert ref.loc["H2", "adverse_status"] == "suspended" and not ref.loc["H3", "has_specialities"]


def test_wilson_interval_bounds():
    p, lo, hi, n = E._rate(pd.Series([True, False, True, True]))
    assert n == 4 and 0 <= lo < p < hi <= 1
