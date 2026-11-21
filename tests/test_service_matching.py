import pytest
from careproof.service_matching import match_service, resolve_service, keyword_hit


def rec(**kw):
    base = {"name": "City Hospital", "description": None, "procedure": "null", "equipment": "null",
            "capability": "null", "specialties": "null", "facilityTypeId": "hospital"}
    base.update(kw)
    return base


@pytest.mark.parametrize("q,key", [
    ("Haemodialysis", "dialysis"), ("kidney dialysis", "dialysis"), ("emergency C-section", "c_section"),
    ("Caesarean", "c_section"), ("LSCS", "c_section"), ("neonatal intensive care", "nicu"), ("NICU", "nicu"),
    ("magnetic resonance imaging", "mri"), ("open heart surgery", "cardiac_surgery"), ("CABG", "cardiac_surgery"),
    ("burns", "burn_care"), ("trauma care", "trauma_emergency"), ("casualty", "trauma_emergency"),
])
def test_service_synonyms(q, key):
    assert resolve_service(q).key == key


def test_unknown_service():
    with pytest.raises(ValueError, match="not a supported service"):
        resolve_service("acupuncture")


def test_exact_excerpt_extraction():
    desc = "Founded in 1990. We run a 10-bed haemodialysis unit open 24x7. Parking available."
    r = rec(description=desc, procedure='["Hemodialysis"]', specialties='["nephrology"]')
    m = match_service(r, "dialysis")
    assert m.evidence_level == "explicit_text" and m.is_candidate
    assert m.primary_excerpt_field == "description"
    assert m.primary_excerpt in desc  # exact substring
    assert m.primary_excerpt == "We run a 10-bed haemodialysis unit open 24x7."
    assert set(m.matched_fields) == {"description", "procedure", "specialties"}
    assert "haemodialysis" in m.matched_terms and "nephrology" in m.specialty_tags
    assert m.readiness_score == 1.0 and m.evidence_confidence_score >= 0.75


def test_long_excerpt_is_bounded_exact_substring():
    desc = ("x " * 400) + "dialysis available" + (" y" * 400)
    m = match_service(rec(description=desc), "dialysis")
    assert len(m.primary_excerpt) <= 300 and m.primary_excerpt in desc and "dialysis" in m.primary_excerpt


def test_specialty_tag_alone_is_not_a_candidate():
    m = match_service(rec(specialties='["gynecologyAndObstetrics"]'), "c_section")
    assert m.evidence_level == "specialty_tag_only" and not m.is_candidate and m.primary_excerpt is None


def test_negation_is_a_conflict_not_support():
    m = match_service(rec(capability='["MRI not available on site"]'), "mri")
    assert m.evidence_level == "negated_only" and not m.is_candidate
    m = match_service(rec(procedure='["Without dialysis facility"]'), "dialysis")
    assert m.evidence_level == "negated_only"


@pytest.mark.parametrize("text,ok", [
    ("Dedicated Burn Unit", True), ("Burns management", True), ("Treats acute burns", True),
    ("Burns and plastic surgery department", True), ("Post-burn reconstruction", False),
    ("Burns-related hair restoration is offered.", False), ("Fat burn programme", False),
    ("Sun Burn Treatment", False), ("Burn Reconstruction", False), ("Has three isolated burn beds.", True),
    ("Burns & Trauma care", True), ("Has SICU, NICU, PICU, burns units", True),
])
def test_burn_care_requires_care_context(text, ok):
    assert match_service(rec(procedure=f'["{text}"]'), "burn_care").is_candidate is ok


def test_planned_services_are_not_evidence():
    m = match_service(rec(capability='["300-bed trauma care hospital under construction"]'), "trauma_emergency")
    assert m.evidence_level == "negated_only" and not m.is_candidate
    m = match_service(rec(description="A dialysis unit will be started next year. We offer OPD."), "dialysis")
    assert not m.is_candidate


def test_false_positive_exclusions():
    assert match_service(rec(procedure='["Gastric Bypass Surgery"]'), "cardiac_surgery").evidence_level == "none"
    assert match_service(rec(specialties='["eyeTraumaAndEmergencyEyeCare"]'), "trauma_emergency").evidence_level == "none"
    assert match_service(rec(capability='["Eye trauma care"]'), "trauma_emergency").evidence_level == "none"
    assert match_service(rec(procedure='["Coronary artery bypass grafting (CABG)"]'), "cardiac_surgery").is_candidate


def test_specialty_mismatch_conflict():
    m = match_service(rec(name="Aravind Eye Hospital", procedure='["Dialysis unit with 10 machines"]'), "dialysis")
    assert m.is_candidate and "specialty_mismatch" in m.conflicts
    m = match_service(rec(name="Smile Care", facilityTypeId="dentist", procedure='["MRI"]'), "mri")
    assert "specialty_mismatch" in m.conflicts


def test_baseline_keyword_is_naive():
    r = rec(capability='["MRI not available"]')
    assert keyword_hit(r, "mri")  # baseline counts the negated mention
    assert not keyword_hit(rec(name="Amritaa Diagnostics"), "mri")  # word boundaries, not substrings
