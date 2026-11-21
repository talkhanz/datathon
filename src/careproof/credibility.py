"""Identity, location and attribution credibility checks + quarantine reason codes.

Every finding is a dict: {code, severity, evidence, human_review_required}. Severity drives the hard
eligibility gates in pipeline.py (high -> facility cannot be a normal recommendation for any service).
Rules are transparent and were derived from observed records (see docs/quarantine_rules.md).
"""
from __future__ import annotations

import difflib
import re
from typing import Iterable, Optional

from .normalization import normalize_name

PIPELINE_VERSION = "careproof-0.2.0"

HIGH, MEDIUM, LOW = "high", "medium", "low"

# --- Reason codes ---------------------------------------------------------------------------------------------
MULTIPLE_FACILITY_IDENTITIES = "MULTIPLE_FACILITY_IDENTITIES"
MULTIPLE_INCOMPATIBLE_LOCATIONS = "MULTIPLE_INCOMPATIBLE_LOCATIONS"
COORDINATE_LOCATION_CONFLICT = "COORDINATE_LOCATION_CONFLICT"
AMBIGUOUS_PIN = "AMBIGUOUS_PIN"
POSSIBLE_DUPLICATE = "POSSIBLE_DUPLICATE"
WEAK_SERVICE_EVIDENCE = "WEAK_SERVICE_EVIDENCE"
MISSING_EVIDENCE_EXCERPT = "MISSING_EVIDENCE_EXCERPT"
MISSING_SOURCE_PROVENANCE = "MISSING_SOURCE_PROVENANCE"
SERVICE_CONTRADICTION = "SERVICE_CONTRADICTION"
UNRESOLVED_LOCATION = "UNRESOLVED_LOCATION"
SHARED_PLACEHOLDER_COORDINATE = "SHARED_PLACEHOLDER_COORDINATE"
SPECIALTY_PROFILE_MISMATCH = "SPECIALTY_PROFILE_MISMATCH"
CORRUPTED_SOURCE_ROW = "CORRUPTED_SOURCE_ROW"


def finding(code: str, severity: str, evidence: str, human_review: Optional[bool] = None) -> dict:
    return {"code": code, "severity": severity, "evidence": (evidence or "")[:400],
            "human_review_required": severity in (HIGH, MEDIUM) if human_review is None else human_review}


# --- Organisation-name attribution ------------------------------------------------------------------------------
ORG_RX = re.compile(
    r"\b((?:[A-Z][A-Za-z&'’.-]*\s+){0,4}(?:Hospitals?|Medical College|Nursing Home|Medical Cent(?:re|er)|Clinic))\b")
_GENERIC = {
    "hospital", "hospitals", "the", "and", "of", "pvt", "ltd", "private", "limited", "multispeciality",
    "multispecialty", "multi", "speciality", "specialty", "super", "superspeciality", "superspecialty", "clinic",
    "clinics", "centre", "center", "medical", "care", "health", "healthcare", "nursing", "home", "institute",
    "research", "college", "dr", "sri", "shri", "st", "general", "government", "govt", "district", "civil", "city",
    "new", "children", "childrens", "womens", "women", "mother", "child", "eye", "dental", "heart", "cancer",
    "maternity", "memorial", "charitable", "trust", "mission", "surgical", "ortho", "neuro", "kidney", "best", "top",
    "leading", "infertility", "fertility", "ivf", "spine", "dialysis", "empanelled", "ayushman", "card", "network",
    "listed", "nearby", "local", "main", "rural", "community", "primary", "public", "referral", "tertiary", "sub",
    "divisional", "regional", "state", "national", "ent", "skin", "diagnostic", "diagnostics", "orthopaedic",
    "orthopedic", "cardiac", "chest", "a", "an", "this", "its", "our", "in", "at", "is", "for", "with", "based",
    "plus", "life", "line", "esic", "cghs", "pmjay",
}


def _core_tokens(name: Optional[str], extra_ignore: Iterable[str] = ()) -> list[str]:
    ign = set(extra_ignore)
    return [t for t in (normalize_name(name) or "").split() if t not in _GENERIC and t not in ign and len(t) > 1]


def same_entity(own_name: Optional[str], other_name: str, ignore_tokens: Iterable[str] = ()) -> bool:
    """True if ``other_name`` plausibly refers to the same facility as ``own_name`` (or cannot be told apart).

    Conservative toward *not* flagging: generic names, spelling variants (ratio >= 0.85), spacing variants
    ("Life Line" / "Lifeline") and acronyms ("GAIMS", "DMC") count as the same entity.
    """
    own = _core_tokens(own_name)
    oth = _core_tokens(other_name, ignore_tokens)
    if not oth or not own:
        return True
    own_cat, oth_cat = "".join(own), "".join(oth)
    if own_cat in oth_cat or oth_cat in own_cat:
        return True
    own_all = [t for t in (normalize_name(own_name) or "").split() if len(t) > 0]
    acronym = "".join(t[0] for t in own_all if t not in {"and", "of", "the"})
    if any(len(t) >= 3 and (t in acronym or acronym.startswith(t)) for t in oth):
        return True
    for t in oth:
        if any(difflib.SequenceMatcher(None, t, u).ratio() >= 0.85 for u in own):
            return True
    return False


def other_organisations(text: Optional[str], own_name: Optional[str], ignore_tokens: Iterable[str] = ()) -> list[str]:
    """Named organisations in ``text`` that are NOT the facility itself."""
    out = []
    for m in ORG_RX.finditer(text or ""):
        cand = m.group(1).strip()
        if not same_entity(own_name, cand, ignore_tokens) and cand not in out:
            out.append(cand)
    return out


_SUBJECT_RX = re.compile(
    r"^\s*((?:[A-Z][A-Za-z&'’.-]*\s+){0,5}(?:Hospitals?|Medical College|Nursing Home|Medical Cent(?:re|er)|Clinic))\b")


def description_subject_conflict(description: Optional[str], own_name: Optional[str],
                                 ignore_tokens: Iterable[str] = ()) -> Optional[str]:
    """If the description's grammatical subject is a *different* named facility, return that name."""
    m = _SUBJECT_RX.match(description or "")
    if m and not same_entity(own_name, m.group(1), ignore_tokens):
        return m.group(1).strip()
    return None


# --- Third-party / referral provision (claim is not on-site provision by this facility) ------------------------
THIRD_PARTY_RX = re.compile(
    r"\b(?:(?:are|is|were|was|being|get|gets)\s+referred\s+(?:to|out)|referred\s+to|refers?\s+(?:patients|cases)|"
    r"tie[\s-]?up\s+with|through\s+(?:a|an)?\s*(?:tie[\s-]?up|partner\w*)|in\s+(?:collaboration|partnership|"
    r"association|coordination)\s+with|outsourced\s+to|formerly|previously\s+(?:offered|provided)|"
    r"no\s+longer|discontinued|near(?:by)?\s+(?:a\s+|the\s+)?hospital\s+(?:offering|providing|with))\b", re.I)


def third_party_language(unit: str) -> Optional[str]:
    m = THIRD_PARTY_RX.search(unit or "")
    return m.group(0) if m else None


def mentions_self(text: Optional[str], own_name: Optional[str], ignore_tokens: Iterable[str] = ()) -> bool:
    """True if the text names this facility (a named organisation with distinctive tokens matching own name)."""
    for m in ORG_RX.finditer(text or ""):
        cand = m.group(1)
        if _core_tokens(cand, ignore_tokens) and same_entity(own_name, cand, ignore_tokens):
            return True
    own = "".join(_core_tokens(own_name))
    return bool(own) and len(own) >= 4 and own in (normalize_name(text) or "").replace(" ", "")


def attribution(unit: str, own_name: Optional[str], ignore_tokens: Iterable[str] = ()) -> tuple[bool, list[str]]:
    """(attributable, other_orgs). Not attributable when the text names another facility and not this one."""
    others = other_organisations(unit, own_name, ignore_tokens)
    if not others:
        return True, []
    return mentions_self(unit, own_name, ignore_tokens), others
