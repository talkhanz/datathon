"""External, independent reference data: Ayushman Bharat PM-JAY hospital empanelment (Haryana State Health Agency).

WHY THIS IS A FAIR CHECK
    Hospitals are empanelled for specific specialities after the State Health Agency checks their infrastructure
    and staff against national empanelment criteria. The lists are official government publications, produced
    completely independently of the Virtue Foundation records and of CareProof's rules.

WHAT IT CAN AND CANNOT SAY
    * "Empanelled for speciality X" is strong *positive* support that the hospital is equipped for X (as of the list date).
    * "Not empanelled for X" is weak *negative* evidence: a hospital can offer a service without empanelling it.
      Metrics are therefore reported as external SUPPORT RATES per system and per evidence level, not as accuracy.

PRIVACY
    The PDFs also list nodal officers' phone numbers and e-mails. Those columns are dropped at parse time and never
    stored. Raw PDFs live in data/external/ (git-ignored); the fetch script records URLs and list dates.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .normalization import normalize_district, normalize_name

# PM-JAY speciality -> CareProof service. Strength says how directly the speciality implies the service.
SPECIALTY_MAP = {
    "nicu": ((r"neo-?natal", r"neonatology"), "direct: neonatal care packages / neonatology"),
    "burn_care": ((r"\bburns?\b",), "direct: burns management"),
    "mri": ((r"\bmri\b",), "direct: MRI"),
    "trauma_emergency": ((r"polytrauma", r"\bemergency\b"), "direct: polytrauma / emergency packages"),
    "cardiac_surgery": ((r"cardio-?thoracic", r"\bctvs\b"), "direct: cardio-thoracic & vascular surgery"),
    "c_section": ((r"obstetrics",), "broad: obstetrics & gynaecology empanelment includes caesarean packages"),
    "dialysis": ((r"nephrology",), "partial: nephrology empanelment (haemodialysis is also billable under other specialities)"),
}
KEEP = ("hospital_id", "hospital_name", "hospital_type", "district", "specialities_raw")
_HEADER = {"hospital id": "hospital_id", "hospital name": "hospital_name", "hospital type": "hospital_type",
           "empanelled specialities": "specialities_raw", "district": "district"}


def _norm_header(c) -> str:
    return re.sub(r"\s+", " ", str(c or "").replace("\n", " ")).strip().lower()


def parse_pmjay_pdf(path: str, district_hint: str | None = None) -> pd.DataFrame:
    """Parse one district list. Only KEEP columns survive (no contact data)."""
    import pdfplumber
    rows, cols, title = [], None, ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for r in table:
                    cells = [re.sub(r"\s+", " ", (c or "").replace("\n", " ")).strip() for c in r]
                    heads = [_norm_header(c) for c in r]
                    if any(h in ("hospital id",) for h in heads):
                        cols = {i: _HEADER[h] for i, h in enumerate(heads) if h in _HEADER}
                        continue
                    if not title and any("district" in c.lower() for c in cells if c):
                        title = next(c for c in cells if c)
                    if cols is None or not any(re.match(r"^HOSP\w+$", c) for c in cells):
                        continue
                    rows.append({name: cells[i] for i, name in cols.items() if i < len(cells)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    m = re.search(r"District\s+([A-Za-z ]+?)\s*(?:\(|Empanelled|$)", title, re.I)
    dist = district_hint or (m.group(1) if m else None)
    if "district" not in df or df["district"].isna().all():
        df["district"] = dist
    date = re.search(r"[Aa]s [Oo]n\s*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})", title)
    df["list_date"] = date.group(1).replace("-", "/") if date else None
    for c in KEEP:
        if c not in df:
            df[c] = None
    return df[list(KEEP) + ["list_date"]]


STATUS_LISTS = ("suspended", "de_empanelled", "withdrawn", "inactive")


def build_reference(parsed: dict) -> pd.DataFrame:
    """One row per PM-JAY hospital ID: specialities (from the speciality-wise lists), whether it is on the current
    empanelled list, and any adverse status (suspended / de-empanelled / withdrawn / inactive)."""
    frames = list(parsed.values())
    allr = pd.concat(frames, ignore_index=True)
    spec = allr[allr.list_type == "empanelled_specialities"].drop_duplicates("hospital_id", keep="last")
    cur = allr[allr.list_type == "empanelled_current"].drop_duplicates("hospital_id")
    ids = pd.Index(pd.concat([spec.hospital_id, cur.hospital_id]).unique(), name="hospital_id")
    ref = pd.DataFrame(index=ids)
    for c in ("hospital_name", "hospital_type", "district"):
        ref[c] = spec.set_index("hospital_id")[c].reindex(ids).fillna(cur.set_index("hospital_id")[c].reindex(ids))
    ref["specialities_raw"] = spec.set_index("hospital_id").specialities_raw.reindex(ids)
    ref["specialities_list_date"] = spec.set_index("hospital_id").list_date.reindex(ids)
    ref["has_specialities"] = ref.specialities_raw.notna()
    ref["on_current_list"] = ids.isin(cur.hospital_id)
    ref["current_list_date"] = cur.list_date.iloc[0] if len(cur) else None
    status = allr[allr.list_type.isin(STATUS_LISTS)].drop_duplicates("hospital_id", keep="last").set_index("hospital_id")
    ref["adverse_status"] = status.list_type.reindex(ids)
    ref["adverse_status_date"] = status.list_date.reindex(ids)
    return ref.reset_index()


def specialities(raw) -> list[str]:
    return [s.strip() for s in re.split(r",", str(raw or "")) if s.strip() and s.strip().lower() != "na"]


def empanelled_for(raw, service: str) -> bool:
    low = re.sub(r"\s*-\s*", "-", str(raw or "").lower())
    return any(re.search(k, low) for k in SPECIALTY_MAP[service][0])


# ------------------------------------------------------------------------------------------------ matching
GENERIC = {"hospital", "hospitals", "hosptal", "civil", "chc", "phc", "community", "health", "centre", "center", "primary",
           "sub", "divisional", "district", "general", "nursing", "home", "multispeciality", "multispecialty", "multi",
           "super", "speciality", "specialty", "and", "the", "of", "medical", "clinic", "private", "limited", "pvt", "ltd",
           "care", "institute", "sdh", "uphc", "urban", "research", "trust", "unit", "haryana", "india", "delhi", "ncr",
           "memorial", "charitable", "society", "s", "dr"}
_CUT = re.compile(r"\b(a unit of|unit of|run by|managed by|near|opp|opposite)\b.*$|,.*$|\(.*$", re.I)


def core_name(name: str, places: set[str] = frozenset()) -> str:
    """Name with legal suffixes, locations and generic hospital words removed: 'Nidaan Hospital A Unit Of X Pvt Ltd'
    -> 'nidaan'. Two records match only if their cores are (almost) identical."""
    base = _CUT.sub("", str(name or ""))
    toks = [t for t in normalize_name(base).split() if t not in GENERIC and t not in places]
    return " ".join(toks)


def names_match(a: str, b: str, places: set[str] = frozenset()) -> bool:
    from difflib import SequenceMatcher
    ca, cb = core_name(a, places), core_name(b, places)
    if len(ca) < 4 or len(cb) < 4:
        return False
    return ca == cb or SequenceMatcher(None, ca, cb).ratio() >= 0.92


def match_facilities(ext: pd.DataFrame, silver: pd.DataFrame, state: str = "HARYANA") -> pd.DataFrame:
    """One-to-one, precision-first name match within the same district (or city). Generic names ("Civil Hospital",
    "PHC X") reduce to an empty core and never match; several candidates -> 'ambiguous' and dropped."""
    s = silver[silver.state_norm == state].copy()
    nd = lambda v: normalize_district(v) if isinstance(v, str) else None  # noqa: E731
    s["district_norm"], s["city_norm2"] = s.pin_primary_district.map(nd), s.address_city.map(nd)
    e = ext.assign(district_norm=ext.district.map(nd))
    places = {t for v in pd.concat([s.district_norm, s.city_norm2, e.district_norm]).dropna() for t in str(v).lower().split()}
    out = []
    for r in e.itertuples():
        pool = s[(s.district_norm == r.district_norm) | (s.city_norm2 == r.district_norm)]
        hits = [f for f in pool.itertuples() if names_match(r.hospital_name, f.name, places)]
        rec = {"hospital_id": r.hospital_id, "hospital_name": r.hospital_name, "district": r.district,
               "core_name": core_name(r.hospital_name, places)}
        if len(hits) == 1:
            out.append({**rec, "facility_id": hits[0].facility_id, "facility_name": hits[0].name, "match": "unique"})
        elif len(hits) > 1:
            out.append({**rec, "facility_id": None, "facility_name": "; ".join(h.name for h in hits[:3]), "match": "ambiguous"})
    m = pd.DataFrame(out)
    if m.empty:
        return m
    uniq = m[m.match == "unique"]
    dup = uniq.facility_id.duplicated(keep=False)  # one CareProof record claimed by several PM-JAY rows -> drop all
    m.loc[uniq.index[dup], "match"] = "ambiguous"
    return m


# ------------------------------------------------------------------------------------------------ evaluation
def _rate(x: pd.Series) -> tuple[float, float, float, int]:
    """Share and Wilson 95% interval."""
    n = int(len(x))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    p, z = float(x.mean()), 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h), n


def external_agreement(matched: pd.DataFrame, ext: pd.DataFrame, gold: pd.DataFrame, scores: pd.DataFrame | None = None):
    """Per service: how often facilities flagged by each system are PM-JAY-empanelled for the matching speciality."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    mm = matched[matched.match == "unique"].merge(ext[["hospital_id", "specialities_raw", "hospital_type"]], on="hospital_id")
    rows, detail = [], []
    for svc in SPECIALTY_MAP:
        d = mm.assign(empanelled=mm.specialities_raw.map(lambda v: empanelled_for(v, svc)))
        g = gold[gold.service == svc].set_index("facility_id")
        d["in_gold"] = d.facility_id.isin(g.index)
        get = lambda col, default: d.facility_id.map(g[col]).where(d.in_gold, default)  # noqa: E731
        d["careproof_eligible"] = get("eligible", False).astype(bool)
        d["evidence_category"] = get("evidence_category", "insufficient").fillna("insufficient")
        d["keyword_hit"] = get("baseline_keyword_hit", False).astype(bool)
        d["exact_term_hit"] = get("baseline_exact_term_hit", False).astype(bool)
        d["tag_rule"] = get("specialty_tags_json", "[]").fillna("[]").map(lambda v: v not in ("[]", "", None))
        d["held_back"] = d.keyword_hit & ~d.careproof_eligible
        if scores is not None:
            sc = scores[scores.service == svc].set_index("facility_id").model_score
            d["model_score"] = d.facility_id.map(sc)
        base = _rate(d.empanelled)
        for system, mask in (("all matched hospitals", pd.Series(True, index=d.index)),
                             ("B2 exact-word match", d.exact_term_hit), ("B3 keyword/synonym match", d.keyword_hit),
                             ("specialty-tag rule", d.tag_rule), ("CareProof eligible", d.careproof_eligible),
                             ("CareProof held back (keyword hit, failed checks)", d.held_back),
                             ("CareProof strong", d.evidence_category == "strong"),
                             ("CareProof moderate", d.evidence_category == "moderate"),
                             ("CareProof weak", d.evidence_category == "weak"),
                             ("model profile fit >= 0.5", d.get("model_score", pd.Series(np.nan, index=d.index)) >= 0.5)):
            p, lo, hi, n = _rate(d.loc[mask, "empanelled"])
            rows.append({"service": svc, "mapping": SPECIALTY_MAP[svc][1], "system": system, "n_flagged": n,
                         "empanelled_rate": p, "ci95_low": lo, "ci95_high": hi, "base_rate": base[0],
                         "recall_of_empanelled": float(mask[d.empanelled].mean()) if d.empanelled.any() else float("nan")})
        if scores is not None and d.empanelled.nunique() == 2 and d.model_score.notna().all():
            rows.append({"service": svc, "mapping": SPECIALTY_MAP[svc][1], "system": "model score (ranking)",
                         "n_flagged": len(d), "empanelled_rate": float("nan"), "ci95_low": float("nan"),
                         "ci95_high": float("nan"), "base_rate": base[0], "recall_of_empanelled": float("nan"),
                         "roc_auc": float(roc_auc_score(d.empanelled, d.model_score)),
                         "average_precision": float(average_precision_score(d.empanelled, d.model_score))})
        detail.append(d.assign(service=svc))
    return pd.DataFrame(rows), pd.concat(detail, ignore_index=True)


def matches_table(matched: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Unique matches joined to the reference: what the app shows as an independent corroboration badge."""
    u = matched[matched.match == "unique"][["facility_id", "hospital_id", "hospital_name", "facility_name"]]
    out = u.merge(ref.drop(columns=["hospital_name"]), on="hospital_id", how="left")
    for svc in SPECIALTY_MAP:
        out[f"empanelled_{svc}"] = out.specialities_raw.map(lambda v, s=svc: empanelled_for(v, s)).where(out.has_specialities)
    return out


def run(ref: pd.DataFrame, silver: pd.DataFrame, gold: pd.DataFrame, scores: pd.DataFrame | None = None) -> dict:
    """Match, then measure agreement on hospitals whose speciality list is known."""
    m = match_facilities(ref, silver)
    spec = ref[ref.has_specialities]
    agg, det = external_agreement(m[m.hospital_id.isin(spec.hospital_id)], spec, gold, scores)
    return {"external_matches": matches_table(m, ref), "external_validation": agg,
            "external_validation_detail": det.drop(columns=["specialities_raw"], errors="ignore"),
            "match_summary": m.match.value_counts().to_dict()}
