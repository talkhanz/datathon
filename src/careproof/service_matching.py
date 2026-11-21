"""Deterministic, interpretable service matching with exact evidence excerpts.

Design rules
------------
* A service is only *claimed* when an explicit service phrase appears in free
  text (procedure / equipment / capability / description / name).
* Specialty tags (e.g. ``nephrology``) are recorded as *supporting context*
  only; they never make a facility recommendable on their own.
* Mentions that are negated ("MRI not available", "without dialysis") are
  recorded as contradictions, not support.
* Every excerpt is an exact substring of the original field value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Optional

from .credibility import attribution, third_party_language
from .normalization import clean_str, normalize_name, parse_array

# Field weights for service_match_score: how directly a field states a service.
FIELD_STRENGTH = {
    "procedure": 1.0,
    "equipment": 1.0,
    "capability": 0.9,
    "description": 0.9,
    "name": 0.7,
}
SPECIALTY_ONLY_STRENGTH = 0.35
ARRAY_FIELDS = ("procedure", "equipment", "capability")
STRING_FIELDS = ("description", "name")
# Excerpt preference: description first (most human-readable), then most specific.
EXCERPT_FIELD_PRIORITY = ("description", "procedure", "equipment", "capability", "name")
MAX_EXCERPT_CHARS = 300


@dataclass(frozen=True)
class ServiceDef:
    key: str
    label: str
    query_aliases: tuple[str, ...]
    strong_patterns: tuple[str, ...]
    specialty_tag_patterns: tuple[str, ...]
    profile_exempt: tuple[str, ...] = ()  # name-profile categories that do NOT conflict
    nfhs_context: tuple[str, ...] = ()


TAXONOMY: dict[str, ServiceDef] = {s.key: s for s in [
    ServiceDef(
        "dialysis", "Dialysis / haemodialysis",
        ("dialysis", "haemodialysis", "hemodialysis", "kidney dialysis", "renal dialysis", "crrt",
         "peritoneal dialysis"),
        (r"\b(?:haemo|hemo|peritoneal\s)?dialy(?:sis|ses|zers?|sers?)\b", r"\bdialysis\b",
         r"\brenal replacement therapy\b", r"\bCRRT\b", r"\bCAPD\b"),
        (r"^nephrology", r"dialysis", r"^renalCare$", r"^pediatricNephrology$"),
        nfhs_context=("w15_plus_with_high_or_very_high_gt_140_mg_dl_blood_sugar_or_pct",
                      "w15_plus_with_high_bp_sys_gte_140_mmhg_and_or_dia_gte_90_mm_pct"),
    ),
    ServiceDef(
        "c_section", "Caesarean section / emergency obstetric surgery",
        ("c-section", "c section", "csection", "caesarean", "cesarean", "caesarian", "cesarian",
         "emergency c-section", "emergency caesarean", "lscs", "emergency obstetric surgery",
         "emergency obstetric care", "cemonc"),
        (r"\bc[\s\-]?sections?\b", r"\bcaesarean\b", r"\bcesarean\b", r"\bcaesarian\b", r"\bcesarian\b",
         r"\bLSCS\b", r"\bCEmONC\b", r"\bcomprehensive emergency obstetric",
         r"\bemergency obstetric (?:surgery|care|services?)\b"),
        (r"obstetric", r"maternalFetal", r"^gynecologyAndObstetrics$"),
        profile_exempt=("fertility",),
        nfhs_context=("births_delivered_by_csection_5y_pct", "institutional_birth_5y_pct",
                      "births_in_a_public_fac_that_were_delivered_by_csection_5y_pct"),
    ),
    ServiceDef(
        "nicu", "Neonatal intensive care (NICU)",
        ("nicu", "neonatal intensive care", "neonatal icu", "newborn icu", "sncu", "neonatal care"),
        (r"\bNICU\b", r"\bneonatal intensive care\b", r"\bneonatal ICU\b", r"\bSNCU\b",
         r"\bspecial newborn care unit\b", r"\bnewborn intensive care\b", r"\blevel[\s\-]*(?:II|III|2|3)\s+neonatal\b"),
        (r"neonat",),
        profile_exempt=("fertility",),
        nfhs_context=("institutional_birth_5y_pct",),
    ),
    ServiceDef(
        "mri", "MRI (magnetic resonance imaging)",
        ("mri", "magnetic resonance imaging", "magnetic resonance", "mri scan"),
        (r"\bMRI\b", r"\bmagnetic resonance\b", r"\bMR (?:imaging|angiography|scan)\b"),
        (r"radiolog", r"imaging"),
        profile_exempt=("diagnostic",),
    ),
    ServiceDef(
        "cardiac_surgery", "Cardiac surgery",
        ("cardiac surgery", "heart surgery", "open heart surgery", "cabg", "bypass surgery",
         "cardiothoracic surgery", "ctvs", "valve replacement"),
        (r"\bcardiac surger(?:y|ies)\b", r"\bheart surger(?:y|ies)\b", r"\bopen[\s\-]heart\b", r"\bCABG\b",
         r"\bcoronary artery bypass\b", r"\bcardio[\s\-]?thoracic (?:and vascular )?surg", r"\bCTVS\b",
         r"\b(?:heart |cardiac |aortic |mitral )valve (?:replacement|repair)\b",
         r"(?<!gastric )(?<!gastric-)(?<!gastric\s)\b(?:coronary )?bypass surger(?:y|ies)\b"),
        (r"^cardiacSurgery$", r"cardiothoracicSurgery", r"cardioThoracicSurgery", r"Cardio Thoracic Surgery"),
    ),
    ServiceDef(
        "burn_care", "Burn care",
        ("burn", "burns", "burn care", "burn unit", "burns unit", "burn treatment", "burns ward"),
        # Broad match on burn(s), excluding observed non-acute contexts (post-burn reconstruction, sunburn,
        # fat burning, scars/marks, burns-related hair restoration). Negation is handled separately.
        (r"(?<!post-)(?<!post )(?<!sun )(?<!sun-)(?<!fat )(?<!calorie )(?<!following )\bburns?\b"
         r"(?![\s\-]*(?:scars?|marks?|related|reconstruct\w*|contractures?|deformit\w*|out\b))",),
        (r"^burn", r"^burns"),
    ),
    ServiceDef(
        "trauma_emergency", "Trauma and emergency care",
        ("trauma", "trauma care", "emergency", "emergency care", "casualty", "accident", "accident and emergency",
         "a&e", "er", "emergency room", "polytrauma"),
        (r"(?<!eye )(?<!dental )(?<!ocular )(?<!psychological )\btrauma (?:care|cent(?:re|er)|units?|surgery|team|ward|services?|icu)\b",
         r"\bpolytrauma\b", r"\bcasualty\b", r"\baccident (?:and|&) emergency\b", r"\bA\s?&\s?E\b",
         r"\bemergency (?:department|room|ward|services?|care|units?|medicine)\b",
         r"\b24\s*[x×/*]\s*7 emergency\b",
         r"(?<!eye )(?<!dental )(?<!ocular )(?<!psychological )\btrauma\b(?=.{0,30}\b(?:surgery|care|management)\b)"),
        (r"^emergencyMedicine$", r"^accidentAndEmergency$", r"^polytrauma$", r"^traumaSurgery$",
         r"criticalCare", r"^orthopedicTrauma"),
    ),
]}

# Negation cues inside the same sentence/array element.
_NEG_BEFORE = re.compile(
    r"\b(?:no|not|without|non[\s\-]?availab\w*|unavailable|does not|doesn't|do not|don't|lacks?|lacking|"
    r"absence of|outsourced|refers? (?:out|patients)|referred (?:out|to|elsewhere))\b[^.;:]{0,50}$", re.I)
_NEG_AFTER = re.compile(
    r"^[^.;:]{0,25}\b(?:not (?:available|provided|offered|specified|listed|mentioned|present)|unavailable|"
    r"is not|are not|outsourced)\b", re.I)

# Planned / not-yet-operational services anywhere in the same sentence or array element.
_NOT_OPERATIONAL = re.compile(
    r"\b(?:under construction|being (?:built|constructed|set up)|proposed|planned|upcoming|coming soon|"
    r"will be (?:started|set up|established|opened|launched|available)|to be (?:started|set up|established|opened|launched)|"
    r"yet to (?:start|be)|in the pipeline|foundation stone)\b", re.I)

# Facility-name profiles that are implausible providers of the supported services.
PROFILE_PATTERNS = {
    "eye": r"\b(?:eye|eyes|netralaya|nethralaya|netra|vision|optical|ophthalm\w*|lasik|retina)\b",
    "dental": r"\b(?:dental|dentist\w*|dentistry|tooth|teeth|orthodont\w*|smile)\b",
    "skin_hair": r"\b(?:skin|derma\w*|hair|cosmetic|aesthetic\w*|laser clinic)\b",
    "alternative": r"\b(?:ayurved\w*|homoeo\w*|homeo\w*|unani|siddha|naturopathy|panchakarma)\b",
    "physio": r"\b(?:physio\w*|rehab\w*)\b",
    "fertility": r"\b(?:ivf|fertility|test tube baby)\b",
    "diagnostic": r"\b(?:diagnostic\w*|path\s?lab\w*|x[\s\-]?ray|imaging cent(?:re|er)|scan cent(?:re|er)|laborator\w*)\b",
}
INCOMPATIBLE_FACILITY_TYPES = {"dentist", "pharmacy", "farmacy"}

_DETAIL = re.compile(
    r"\b(?:\d+\s*(?:-|\s)?(?:bed(?:ded|s)?|machines?|units?|stations?|chairs?|tesla|t)\b|\d(?:\.\d)?\s*t(?:esla)?\b|"
    r"units?|department|dept|cent(?:re|er)|ward|theatre|theater|icu|machines?|level\s*(?:ii|iii|2|3))", re.I)
_24X7 = re.compile(r"\b24\s*[x×/*]\s*7\b|\b24\s*hours?\b|\b24 hrs\b|\bround[\s\-]the[\s\-]clock\b|\ball day and night\b", re.I)


def _rx(patterns) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.I)


_COMPILED = {k: (_rx(s.strong_patterns), _rx(s.specialty_tag_patterns)) for k, s in TAXONOMY.items()}
_PROFILE_RX = {k: re.compile(p, re.I) for k, p in PROFILE_PATTERNS.items()}


# ---------------------------------------------------------------------------
# Service request parsing
# ---------------------------------------------------------------------------
def _norm_q(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&]+", " ", s.lower())).strip()


def resolve_service(requested: str) -> ServiceDef:
    """Map free-text user input to a supported service. Raises ValueError with suggestions."""
    q = _norm_q(requested or "")
    if not q:
        raise ValueError("Please enter a service, e.g. dialysis, c-section, NICU, MRI, cardiac surgery, burn care, trauma.")
    if q.replace(" ", "_") in TAXONOMY:
        return TAXONOMY[q.replace(" ", "_")]
    best, best_len = None, 0
    for s in TAXONOMY.values():
        for a in s.query_aliases:
            an = _norm_q(a)
            if q == an:
                return s
            if re.search(rf"\b{re.escape(an)}\b", q) and len(an) > best_len:
                best, best_len = s, len(an)
    if best:
        return best
    raise ValueError(f"'{requested}' is not a supported service yet. Supported: "
                     + ", ".join(s.label for s in TAXONOMY.values()) + ".")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
@dataclass
class Hit:
    field: str
    term: str          # the exact matched text
    excerpt: str       # exact substring of the original field value
    negated: bool      # negated, planned, historical, referral or third-party provision
    has_detail: bool
    has_24x7: bool
    attributable: bool = True          # False if the passage is about another named facility
    other_orgs: tuple = ()
    negation_kind: str = ""


@dataclass
class ServiceMatch:
    service: str
    evidence_level: str                     # explicit_text | specialty_tag_only | negated_only | none
    service_match_score: float
    evidence_confidence_score: float
    readiness_score: float
    matched_fields: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    primary_excerpt: Optional[str] = None
    primary_excerpt_field: Optional[str] = None
    excerpts: list[dict] = field(default_factory=list)
    specialty_tags: list[str] = field(default_factory=list)
    negated_excerpts: list[dict] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    missing_corroboration: list[str] = field(default_factory=list)
    has_description_statement: bool = False
    n_distinct_statements: int = 0

    @property
    def is_candidate(self) -> bool:
        return self.evidence_level == "explicit_text"


def _units(text: str) -> list[tuple[int, int]]:
    """Sentence-ish spans (start, end) inside a free-text field."""
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?;])\s+|\n+", text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if b > a]


def _window(text: str, a: int, b: int, ms: int, me: int) -> str:
    """Exact substring of text[a:b] around match [ms, me), at most MAX_EXCERPT_CHARS long."""
    if b - a <= MAX_EXCERPT_CHARS:
        return text[a:b].strip()
    half = (MAX_EXCERPT_CHARS - (me - ms)) // 2
    lo, hi = max(a, ms - half), min(b, me + half)
    return text[lo:hi].strip()


def _scan(text: str, fld: str, rx: re.Pattern, own_name: Optional[str] = None, ignore=()) -> list[Hit]:
    if not rx.search(text):  # fast path: most items never mention the service
        return []
    hits = []
    for a, b in _units(text):
        unit = text[a:b]
        for m in rx.finditer(unit):
            before, after = unit[: m.start()], unit[m.end():]
            kind = ("negated" if (_NEG_BEFORE.search(before) or _NEG_AFTER.search(after)) else
                    "planned" if _NOT_OPERATIONAL.search(unit) else
                    "third_party_or_historical" if third_party_language(unit) else "")
            ex = _window(text, a, b, a + m.start(), a + m.end())
            attr, others = attribution(unit, own_name, ignore) if own_name else (True, [])
            hits.append(Hit(fld, m.group(0), ex, bool(kind), bool(_DETAIL.search(unit)), bool(_24X7.search(unit)),
                            attr, tuple(others), kind))
    return hits


def _stmt_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def count_distinct_statements(excerpts) -> int:
    """Number of distinct statements after normalisation; a statement contained in another counts once."""
    kept: list[str] = []
    for k in sorted({_stmt_key(e) for e in excerpts if e}, key=len, reverse=True):
        if k and not any(k in d for d in kept):
            kept.append(k)
    return len(kept)


def name_profiles(name: Optional[str]) -> list[str]:
    n = name or ""
    return [k for k, rx in _PROFILE_RX.items() if rx.search(n)]


@dataclass(frozen=True)
class ParsedRecord:
    """A facility's text fields parsed once and reused for every service (inference optimisation)."""
    arrays: dict
    strings: dict
    specialties: tuple
    name: Optional[str]
    facility_type: str
    any_24x7: bool
    keyword_blob: str
    place_tokens: tuple = ()


def parse_record(record: Mapping) -> ParsedRecord:
    if isinstance(record, ParsedRecord):
        return record
    arrays = {f: tuple(parse_array(record.get(f))) for f in ARRAY_FIELDS}
    strings = {f: clean_str(record.get(f)) for f in STRING_FIELDS}
    specialties = tuple(parse_array(record.get("specialties")))
    any24 = any(_24X7.search(i) for f in ARRAY_FIELDS for i in arrays[f]) or bool(_24X7.search(strings["description"] or ""))
    blob = " ".join(
        [" ".join(parse_array(record.get(f), dedupe=False)) for f in (*ARRAY_FIELDS, "specialties")]
        + [strings[f] or "" for f in STRING_FIELDS]).lower()
    place = tuple(t for f in ("address_city", "address_stateOrRegion")
                  for t in (normalize_name(record.get(f)) or "").split())
    return ParsedRecord(arrays, strings, specialties, strings["name"],
                        (clean_str(record.get("facilityTypeId")) or "").lower(), any24, blob, place)


def match_service(record, service_key: str) -> ServiceMatch:
    """Match one facility record (raw/silver mapping, or a ParsedRecord) against one service."""
    rec = parse_record(record)
    sdef = TAXONOMY[service_key]
    strong_rx, tag_rx = _COMPILED[service_key]
    hits: list[Hit] = []
    for f in ARRAY_FIELDS:
        for item in rec.arrays[f]:
            hits.extend(_scan(item, f, strong_rx, rec.name, rec.place_tokens))
    for f in STRING_FIELDS:
        s = rec.strings[f]
        if s:
            hits.extend(_scan(s, f, strong_rx, rec.name if f != "name" else None, rec.place_tokens))
    tags = [t for t in rec.specialties if tag_rx.search(t)]
    # Exclude eye-trauma style tags explicitly
    tags = [t for t in tags if not re.search(r"^eye|eyeTrauma|emergencyEye|dent", t, re.I)]

    pos = [h for h in hits if not h.negated and h.attributable]
    neg = [h for h in hits if h.negated]
    misattributed = [h for h in hits if not h.negated and not h.attributable]
    fields_pos = sorted({h.field for h in pos}, key=EXCERPT_FIELD_PRIORITY.index)

    conflicts, missing = [], []
    if neg:
        conflicts.append("negated_mention")
    if misattributed:
        conflicts.append("other_facility_attribution")
    profiles = [p for p in name_profiles(rec.name) if p not in sdef.profile_exempt]
    ftype = rec.facility_type
    if pos and (profiles or ftype in INCOMPATIBLE_FACILITY_TYPES):
        conflicts.append("specialty_mismatch")

    if pos:
        level = "explicit_text"
        smatch = max(FIELD_STRENGTH[h.field] for h in pos)
        n = len(fields_pos)
        # Corroboration counts DISTINCT supporting statements, not fields: the source pipeline extracted
        # procedure/capability/equipment with an LLM from the same pages, so the same sentence often recurs
        # across fields (observed in ~10% of multi-field claims). Repeats are not independent evidence.
        n_stmt = count_distinct_statements([h.excerpt for h in pos])
        conf = {1: 0.5, 2: 0.75}.get(n_stmt, 0.9)
        if tags:
            conf += 0.1
        if any(h.has_detail for h in pos):
            conf += 0.05
        conf = min(1.0, conf)
        if n_stmt == 1:
            missing.append("only one distinct supporting statement" +
                           (f" (repeated across {n} fields)" if n > 1 else f" (in '{fields_pos[0]}')"))
        if not tags:
            missing.append("no related specialty tag")
        in_unit_24 = any(h.has_24x7 for h in pos)
        detail = any(h.has_detail for h in pos)
        facility_24 = rec.any_24x7
        readiness = 0.5 * (1.0 if in_unit_24 else 0.5 if facility_24 else 0.0) + 0.5 * (1.0 if detail else 0.0)
    elif tags:
        level, smatch, conf, readiness = "specialty_tag_only", SPECIALTY_ONLY_STRENGTH, 0.2, 0.0
        missing.append("no explicit text mention of the service")
    elif neg:
        level, smatch, conf, readiness = "negated_only", 0.0, 0.0, 0.0
    elif misattributed:
        level, smatch, conf, readiness = "other_facility_only", 0.0, 0.0, 0.0
    else:
        return ServiceMatch(service_key, "none", 0.0, 0.0, 0.0)

    by_field = {}
    for h in pos:
        by_field.setdefault(h.field, h)
    primary = next((by_field[f] for f in EXCERPT_FIELD_PRIORITY if f in by_field), None)
    excerpts = []
    seen = set()
    for h in sorted(pos, key=lambda h: EXCERPT_FIELD_PRIORITY.index(h.field)):
        if h.excerpt not in seen:
            seen.add(h.excerpt)
            excerpts.append({"field": h.field, "term": h.term, "text": h.excerpt})
        if len(excerpts) >= 5:
            break
    return ServiceMatch(
        service=service_key,
        evidence_level=level,
        service_match_score=round(smatch, 4),
        evidence_confidence_score=round(conf, 4),
        readiness_score=round(readiness, 4),
        matched_fields=fields_pos + (["specialties"] if tags else []),
        matched_terms=sorted({h.term.lower() for h in pos}) + sorted(set(tags)),
        primary_excerpt=primary.excerpt if primary else None,
        primary_excerpt_field=primary.field if primary else None,
        excerpts=excerpts,
        specialty_tags=sorted(set(tags)),
        negated_excerpts=[{"field": h.field, "text": h.excerpt, "kind": h.negation_kind} for h in neg[:3]]
        + [{"field": h.field, "text": h.excerpt, "kind": "other_facility:" + "; ".join(h.other_orgs[:2])}
           for h in misattributed[:3]],
        has_description_statement=any(h.field == "description" for h in pos),
        n_distinct_statements=count_distinct_statements([h.excerpt for h in pos]),
        conflicts=conflicts,
        missing_corroboration=missing,
    )


_KEYWORD_RX = {k: [re.compile(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])")
                   for w in sorted({a.lower() for a in sdef.query_aliases if len(a) >= 3})]
               for k, sdef in TAXONOMY.items()}


EXACT_TERM = {"dialysis": "dialysis", "c_section": "c-section", "nicu": "nicu", "mri": "mri",
              "cardiac_surgery": "cardiac surgery", "burn_care": "burn", "trauma_emergency": "trauma"}
_EXACT_RX = {k: re.compile(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])") for k, t in EXACT_TERM.items()}


def exact_term_hit(record, service_key: str) -> bool:
    """Baseline B2: the single canonical service word appears anywhere (no synonyms, no negation handling)."""
    return bool(_EXACT_RX[service_key].search(parse_record(record).keyword_blob))


def keyword_hit(record, service_key: str) -> bool:
    """Naive baseline: any service alias as a whole word in any text field or specialty tag; no negation handling."""
    blob = parse_record(record).keyword_blob
    return any(rx.search(blob) for rx in _KEYWORD_RX[service_key])
