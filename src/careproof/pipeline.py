"""Table-building and recommendation logic on pandas DataFrames.

Used identically by the local runner (CSV input) and the Databricks notebooks
(Marketplace tables -> toPandas). All inputs are expected as *strings or None*
per cell; the notebooks cast complex Spark types to JSON strings first.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_WEIGHTS, LOCATION_CONFLICT_KM, ScoringWeights, VERIFICATION_WARNING
from .credibility import (
    AMBIGUOUS_PIN, COORDINATE_LOCATION_CONFLICT, HIGH, LOW, MEDIUM, MISSING_SOURCE_PROVENANCE,
    MULTIPLE_FACILITY_IDENTITIES, MULTIPLE_INCOMPATIBLE_LOCATIONS, PIPELINE_VERSION, POSSIBLE_DUPLICATE,
    SHARED_PLACEHOLDER_COORDINATE, UNRESOLVED_LOCATION, description_subject_conflict, finding,
)
from .explanations import explain, warnings_for
from .geography import EARTH_RADIUS_KM, PlaceIndex, ResolvedLocation, aggregate_pin, haversine_km
from .normalization import (
    clean_str, is_valid_uuid, normalize_district, normalize_name, normalize_pin, normalize_state,
    parse_array, parse_nfhs_value, validate_coordinates,
)
from .scoring import (
    COMPONENTS, eligibility, evidence_category, evidence_strength_score, location_quality_score, rank_key,
    score_candidate,
)
from .service_matching import TAXONOMY, exact_term_hit, keyword_hit, match_service, parse_record

ORIGINAL_FACILITY_COLUMNS_REQUIRED = [
    "unique_id", "name", "description", "source_urls", "specialties", "procedure", "equipment", "capability",
    "address_line1", "address_city", "address_stateOrRegion", "address_zipOrPostcode", "latitude", "longitude",
]
SUSPICIOUS_URL_PATTERNS = ("creditcard", "login", "signin", "sign-in", "payment", "tgsouthernpower")
ARRAY_CAP = 50


class FanoutError(RuntimeError):
    pass


@dataclass
class JoinReport:
    name: str
    left_rows: int
    right_rows: int
    output_rows: int
    matched_rows: int
    right_duplicate_keys: int
    match_rate: float = field(init=False)
    duplicate_key_rate: float = field(init=False)
    fanout_rows: int = field(init=False)

    def __post_init__(self):
        self.match_rate = round(self.matched_rows / self.left_rows, 4) if self.left_rows else 0.0
        self.duplicate_key_rate = round(self.right_duplicate_keys / self.right_rows, 4) if self.right_rows else 0.0
        self.fanout_rows = self.output_rows - self.left_rows

    def as_dict(self):
        return dict(self.__dict__)


def checked_left_join(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str, name: str):
    """Left join that raises FanoutError if any left row is multiplied."""
    dup = int(right[right_on].duplicated(keep=False).sum()) if len(right) else 0
    out = left.merge(right, how="left", left_on=left_on, right_on=right_on, indicator=True)
    matched = int((out["_merge"] == "both").sum())
    out = out.drop(columns="_merge")
    rep = JoinReport(name, len(left), len(right), len(out), matched, dup)
    print(f"[join] {name}: left={rep.left_rows} right={rep.right_rows} out={rep.output_rows} "
          f"match_rate={rep.match_rate:.2%} dup_key_rate={rep.duplicate_key_rate:.2%} fanout={rep.fanout_rows}")
    if rep.output_rows != rep.left_rows:
        raise FanoutError(f"Join '{name}' changed row count {rep.left_rows} -> {rep.output_rows}; "
                          f"right side has {dup} rows with duplicate keys.")
    return out, rep


def _row_hash(row: pd.Series) -> str:
    payload = "\x1f".join("" if v is None else str(v) for v in row.values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False)


# ---------------------------------------------------------------------------
# PIN silver
# ---------------------------------------------------------------------------
def build_pincode_silver(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["_pin"] = df["pincode"].map(lambda x: normalize_pin(x).pin)
    bad = int(df["_pin"].isna().sum())
    if bad:
        print(f"[pincode] {bad} post-office rows have an invalid PIN and are excluded from PIN aggregation")
    rows = []
    for pin, g in df[df["_pin"].notna()].groupby("_pin", sort=True):
        s = aggregate_pin(pin, g.to_dict("records"))
        rows.append({
            "pincode": s.pincode,
            "post_office_count": s.post_office_count,
            "n_districts": len(s.districts),
            "n_states": len(s.states),
            "districts_json": _dumps(s.districts),
            "states_json": _dumps(s.states),
            "primary_district": s.primary_district,
            "primary_state": s.primary_state,
            "centroid_lat": s.centroid_lat,
            "centroid_lon": s.centroid_lon,
            "spread_km": s.spread_km,
            "n_offices_with_valid_coords": s.n_offices_with_valid_coords,
            "n_offices_coord_outliers": s.n_offices_coord_outliers,
            "coord_issue_counts_json": _dumps(s.coord_issue_counts),
            "coords_defensible": bool(s.coords_defensible),
            "is_ambiguous": bool(s.is_ambiguous),
            "missing_coordinates": bool(s.missing_coordinates),
        })
    out = pd.DataFrame(rows)
    assert out["pincode"].is_unique, "PIN silver must be unique per pincode"
    return out


def place_index_from_pin_silver(pin_silver: pd.DataFrame) -> PlaceIndex:
    recs = []
    for r in pin_silver.to_dict("records"):
        r = dict(r)
        r["districts"] = json.loads(r["districts_json"])
        r["states"] = json.loads(r["states_json"])
        for k in ("coords_defensible", "is_ambiguous"):
            r[k] = bool(r[k]) if r[k] is not None else False
        recs.append(r)
    return PlaceIndex.from_pin_rows(recs)


# ---------------------------------------------------------------------------
# NFHS silver (long format)
# ---------------------------------------------------------------------------
NFHS_KEY_COLUMNS = ("district_name", "state_ut")


def build_nfhs_silver(raw: pd.DataFrame) -> pd.DataFrame:
    recs = []
    ind_cols = [c for c in raw.columns if c not in NFHS_KEY_COLUMNS]
    for r in raw.to_dict("records"):
        d_raw, s_raw = r["district_name"], r["state_ut"]
        for c in ind_cols:
            v = parse_nfhs_value(r[c])
            recs.append({
                "district_name_raw": d_raw, "state_ut_raw": s_raw,
                "district_norm": normalize_district(d_raw), "state_norm": normalize_state(s_raw),
                "indicator": c, "raw_value": v.raw, "value": v.value,
                "is_suppressed": v.suppressed, "is_small_sample": v.small_sample, "parse_error": v.parse_error,
            })
    out = pd.DataFrame(recs)
    dup = out.duplicated(["district_norm", "state_norm", "indicator"]).sum()
    if dup:
        raise FanoutError(f"NFHS silver has {dup} duplicate (district, state, indicator) keys")
    return out


# ---------------------------------------------------------------------------
# Facility silver
# ---------------------------------------------------------------------------
def build_facility_silver(raw: pd.DataFrame, pin_silver: pd.DataFrame):
    """Return (silver, quarantine, reports)."""
    missing_cols = [c for c in ORIGINAL_FACILITY_COLUMNS_REQUIRED if c not in raw.columns]
    if missing_cols:
        raise KeyError(f"Facilities input is missing expected columns: {missing_cols}")
    df = raw.copy()
    df.insert(0, "source_row_id", df.apply(_row_hash, axis=1))
    reports = {"input_rows": len(df)}

    # 1. Quarantine structurally corrupted rows (seen in the local CSV export).
    reason = np.where(~df["unique_id"].map(is_valid_uuid), "invalid_unique_id",
                      np.where(df["name"].map(clean_str).isna(), "missing_name", ""))
    df["quarantine_reason"] = reason
    quarantine = df[df["quarantine_reason"] != ""].copy()
    df = df[df["quarantine_reason"] == ""].drop(columns="quarantine_reason")
    reports["quarantined_rows"] = len(quarantine)

    # 2. Exact duplicate source rows -> keep one, count copies.
    counts = df.groupby("source_row_id").size()
    df = df.drop_duplicates("source_row_id").copy()
    df["n_exact_source_copies"] = df["source_row_id"].map(counts).astype(int)
    reports["exact_duplicate_rows_removed"] = int(counts.sum() - len(counts))

    # 3. Stable facility_id. If one unique_id has differing content, disambiguate with the row hash.
    uid_dup = df["unique_id"].duplicated(keep=False)
    df["facility_id"] = np.where(uid_dup, df["unique_id"] + "#" + df["source_row_id"].str[:8], df["unique_id"])
    reports["conflicting_unique_id_rows"] = int(uid_dup.sum())
    assert df["facility_id"].is_unique

    # 4. Normalized fields (originals are kept untouched).
    pins = df["address_zipOrPostcode"].map(normalize_pin)
    df["pincode_norm"] = pins.map(lambda p: p.pin)
    df["pincode_issue"] = pins.map(lambda p: p.issue)
    df["facility_name_norm"] = df["name"].map(normalize_name)
    df["city_norm"] = df["address_city"].map(normalize_district)
    df["state_norm"] = df["address_stateOrRegion"].map(normalize_state)
    coords = [validate_coordinates(a, b) for a, b in zip(df["latitude"], df["longitude"])]
    df["facility_lat"] = [c.lat for c in coords]
    df["facility_lon"] = [c.lon for c in coords]
    df["coord_issue"] = [c.issue for c in coords]
    for c in ("specialties", "procedure", "equipment", "capability", "source_urls", "websites"):
        lists = df[c].map(parse_array)
        df[f"{c}_list_json"] = lists.map(_dumps)
        df[f"{c}_n_items"] = lists.map(len)
    raw_lens = {c: df[c].map(lambda x: len(parse_array(x, dedupe=False))) for c in
                ("specialties", "procedure", "equipment", "capability", "source_urls")}
    truncated = pd.concat(raw_lens, axis=1).ge(ARRAY_CAP).any(axis=1)

    # 5. Join PIN reference (PIN grain -> no fanout possible; checked anyway).
    pin_ref = pin_silver[["pincode", "centroid_lat", "centroid_lon", "coords_defensible", "states_json",
                          "primary_district", "is_ambiguous", "n_states", "n_districts"]].rename(columns={
        "n_states": "pin_n_states", "n_districts": "pin_n_districts",
        "pincode": "pin_ref_pincode", "centroid_lat": "pin_centroid_lat", "centroid_lon": "pin_centroid_lon",
        "coords_defensible": "pin_coords_defensible", "states_json": "pin_states_json",
        "primary_district": "pin_primary_district", "is_ambiguous": "pin_is_ambiguous"})
    df, rep = checked_left_join(df, pin_ref, "pincode_norm", "pin_ref_pincode", "facilities_silver x pincode_silver")
    reports["join_pincode"] = rep.as_dict()
    in_dir = df["pin_ref_pincode"].notna()
    defens = df["pin_coords_defensible"].fillna(False).astype(bool)

    # 6. Location cross-checks.
    def _pin_dist(r):
        if r.facility_lat is None or pd.isna(r.facility_lat) or not r.pin_coords_defensible_b:
            return None
        return round(haversine_km(r.facility_lat, r.facility_lon, r.pin_centroid_lat, r.pin_centroid_lon), 3)
    df["pin_coords_defensible_b"] = defens
    df["facility_to_pin_km"] = df.apply(_pin_dist, axis=1)
    df = df.drop(columns="pin_coords_defensible_b")

    def _state_conflict(r):
        # NB: after the join a missing state is NaN, which is truthy - test with isinstance, not truthiness.
        if not isinstance(r.state_norm, str) or not r.state_norm or not isinstance(r.pin_states_json, str):
            return False
        states = json.loads(r.pin_states_json)
        return bool(states) and r.state_norm not in states
    df["state_conflict"] = df.apply(_state_conflict, axis=1)

    has_fc = df["facility_lat"].notna()
    df["location_source"] = np.where(has_fc, "facility_coordinates", np.where(defens, "pin_centroid", "none"))
    df["location_lat"] = np.where(has_fc, df["facility_lat"], np.where(defens, df["pin_centroid_lat"], np.nan))
    df["location_lon"] = np.where(has_fc, df["facility_lon"], np.where(defens, df["pin_centroid_lon"], np.nan))
    df["location_lat"] = pd.to_numeric(df["location_lat"], errors="coerce")
    df["location_lon"] = pd.to_numeric(df["location_lon"], errors="coerce")
    df["location_quality_score"] = [
        location_quality_score(s, None if pd.isna(d) else d, bool(sc))
        for s, d, sc in zip(df["location_source"], df["facility_to_pin_km"], df["state_conflict"])]

    # 7. Possible duplicates: same normalized name within 1 km.
    df["possible_duplicate"] = False
    for _, g in df[has_fc & df["facility_name_norm"].notna()].groupby("facility_name_norm"):
        if len(g) < 2:
            continue
        idx = g.index.tolist()
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = df.loc[idx[i]], df.loc[idx[j]]
                if haversine_km(a.facility_lat, a.facility_lon, b.facility_lat, b.facility_lon) <= 1.0:
                    df.loc[[idx[i], idx[j]], "possible_duplicate"] = True

    # 8. Flags.
    susp = df["source_urls_list_json"].map(lambda s: any(p in u.lower() for u in json.loads(s) for p in SUSPICIOUS_URL_PATTERNS))
    flags = []
    for i, r in enumerate(df.itertuples(index=False)):
        f = []
        if r.facility_to_pin_km is not None and not pd.isna(r.facility_to_pin_km) and r.facility_to_pin_km > LOCATION_CONFLICT_KM:
            f.append("location_conflict")
        if r.state_conflict:
            f.append("state_conflict")
        if r.location_source == "pin_centroid":
            f.append("coordinates_from_pin")
        if r.pincode_norm is None:
            f.append("pin_missing_or_invalid")
        elif not in_dir.iloc[i]:
            f.append("pin_not_in_directory")
        elif not defens.iloc[i]:
            f.append("pin_centroid_unreliable")
        if truncated.iloc[i]:
            f.append("evidence_list_truncated")
        if susp.iloc[i]:
            f.append("suspicious_source_url")
        if r.possible_duplicate:
            f.append("possible_duplicate")
        if r.state_norm is None:
            f.append("state_unrecognised")
        flags.append(_dumps(f))
    df["facility_flags_json"] = flags
    df["primary_source_url"] = [_primary_url(w, s) for w, s in zip(df["officialWebsite"] if "officialWebsite" in df else [None] * len(df),
                                                                 df["source_urls_list_json"])]
    # 9. Identity / location credibility findings (reason codes + severity) and confidence scores.
    pin_xy = {r.pincode: (r.centroid_lat, r.centroid_lon) for r in pin_silver.itertuples()
              if bool(r.coords_defensible) and r.centroid_lat is not None and not pd.isna(r.centroid_lat)}
    coord_names = df[df["facility_lat"].notna()].groupby(["facility_lat", "facility_lon"])["facility_name_norm"].nunique().to_dict()
    findings_col, idc, locc = [], [], []
    for r in df.itertuples(index=False):
        fnd = facility_findings(r, pin_xy, coord_names)
        findings_col.append(_dumps(fnd))
        idc.append(identity_confidence(fnd))
        locc.append(location_confidence(r.location_quality_score, fnd))
    df["facility_findings_json"] = findings_col
    df["identity_confidence_score"] = idc
    df["location_confidence_score"] = locc
    df["pipeline_version"] = PIPELINE_VERSION
    reports["silver_rows"] = len(df)
    reports["location_source_counts"] = df["location_source"].value_counts().to_dict()
    print(f"[facilities] input={reports['input_rows']} quarantined={reports['quarantined_rows']} "
          f"exact_dups_removed={reports['exact_duplicate_rows_removed']} silver={len(df)}")
    return df.reset_index(drop=True), quarantine.reset_index(drop=True), reports


_TEXT_PIN = re.compile(r"(?<!\d)([1-9]\d{2})\s?(\d{3})(?!\d)")
IDENTITY_CODES = {MULTIPLE_FACILITY_IDENTITIES, POSSIBLE_DUPLICATE, MISSING_SOURCE_PROVENANCE, "SUSPICIOUS_SOURCE_URL"}
_SEV_COST = {HIGH: 0.6, MEDIUM: 0.25, LOW: 0.05}


def facility_findings(r, pin_xy: dict, coord_names) -> list:
    """Facility-level identity/location findings for one silver row (namedtuple)."""
    out = []
    place = tuple(t for v in (r.address_city, r.address_stateOrRegion) for t in (normalize_name(v) or "").split())
    subj = description_subject_conflict(clean_str(r.description), r.name, place)
    if subj:
        out.append(finding(MULTIPLE_FACILITY_IDENTITIES, HIGH,
                           f"description is about '{subj}', not '{r.name}': \"{(clean_str(r.description) or '')[:160]}\""))
    d = r.facility_to_pin_km
    if d is not None and not pd.isna(d) and d > LOCATION_CONFLICT_KM:
        out.append(finding(COORDINATE_LOCATION_CONFLICT, HIGH if d > 100 else MEDIUM,
                           f"coordinates are {d:.0f} km from the centroid of the listed PIN {r.pincode_norm}"))
    if r.state_conflict:
        out.append(finding(MULTIPLE_INCOMPATIBLE_LOCATIONS, HIGH,
                           f"listed state '{r.address_stateOrRegion}' is not a state of PIN {r.pincode_norm} {r.pin_states_json}"))
    if r.location_source == "none":
        out.append(finding(UNRESOLVED_LOCATION, HIGH, "no valid coordinates and no reliable PIN centroid"))
    elif r.location_source == "pin_centroid":
        out.append(finding(UNRESOLVED_LOCATION, MEDIUM, "no valid coordinates; position approximated by PIN centroid"))
    if r.location_lat is not None and not pd.isna(r.location_lat):
        text = " ".join([clean_str(r.description) or ""] + parse_array(r.capability))
        far = []
        for a, b in _TEXT_PIN.findall(text):
            pin = a + b
            if pin != r.pincode_norm and pin in pin_xy:
                km = haversine_km(r.location_lat, r.location_lon, *pin_xy[pin])
                if km > 50:
                    far.append(f"{pin} ({km:.0f} km away)")
        if far:
            out.append(finding(MULTIPLE_INCOMPATIBLE_LOCATIONS, MEDIUM, "text mentions distant PIN(s): " + ", ".join(far[:3])))
    if r.facility_lat is not None and not pd.isna(r.facility_lat):
        n = coord_names.get((r.facility_lat, r.facility_lon), 1)
        if n >= 3:
            out.append(finding(SHARED_PLACEHOLDER_COORDINATE, MEDIUM,
                               f"{n} differently named facilities share exactly these coordinates"))
    if getattr(r, "pin_n_states", 0) and not pd.isna(r.pin_n_states) and r.pin_n_states > 1:
        out.append(finding(AMBIGUOUS_PIN, MEDIUM, f"PIN {r.pincode_norm} spans {int(r.pin_n_states)} states"))
    elif getattr(r, "pin_n_districts", 0) and not pd.isna(r.pin_n_districts) and r.pin_n_districts > 1:
        out.append(finding(AMBIGUOUS_PIN, LOW, f"PIN {r.pincode_norm} spans {int(r.pin_n_districts)} districts"))
    if r.possible_duplicate:
        out.append(finding(POSSIBLE_DUPLICATE, LOW, "another record with the same normalized name lies within 1 km"))
    if not json.loads(r.source_urls_list_json) and not clean_str(getattr(r, "officialWebsite", None)):
        out.append(finding(MISSING_SOURCE_PROVENANCE, MEDIUM, "no source URL and no official website"))
    if any(p in u.lower() for u in json.loads(r.source_urls_list_json) for p in SUSPICIOUS_URL_PATTERNS):
        out.append(finding("SUSPICIOUS_SOURCE_URL", LOW, "source URLs include login/payment pages"))
    return out


def identity_confidence(findings: list) -> float:
    cost = sum(_SEV_COST[f["severity"]] for f in findings if f["code"] in IDENTITY_CODES)
    return round(max(0.0, 1.0 - cost), 4)


def location_confidence(location_quality: float, findings: list) -> float:
    cost = sum(_SEV_COST[f["severity"]] * 0.5 for f in findings
               if f["code"] in (SHARED_PLACEHOLDER_COORDINATE, AMBIGUOUS_PIN, MULTIPLE_INCOMPATIBLE_LOCATIONS)
               and f["severity"] != HIGH)
    if any(f["severity"] == HIGH and f["code"] in (COORDINATE_LOCATION_CONFLICT, MULTIPLE_INCOMPATIBLE_LOCATIONS,
                                                   UNRESOLVED_LOCATION) for f in findings):
        return round(min(location_quality, 0.2), 4)
    return round(max(0.0, location_quality - cost), 4)


def _primary_url(official, urls_json) -> Optional[str]:
    o = clean_str(official)
    if o:
        return o if o.startswith("http") else "https://" + o
    for u in json.loads(urls_json):
        if not any(p in u.lower() for p in SUSPICIOUS_URL_PATTERNS):
            return u
    return None


# ---------------------------------------------------------------------------
# Gold: facility x service evidence
# ---------------------------------------------------------------------------
GOLD_CARRY = ["facility_id", "source_row_id", "name", "facilityTypeId", "address_line1", "address_city",
              "address_stateOrRegion", "address_zipOrPostcode", "location_lat", "location_lon", "location_source",
              "location_quality_score", "location_confidence_score", "identity_confidence_score", "facility_to_pin_km",
              "primary_source_url", "source_urls_list_json", "facility_flags_json", "facility_findings_json"]


def build_facility_service_gold(silver: pd.DataFrame, services=None, gates=None) -> pd.DataFrame:
    """One row per (facility, service) with any signal: evidence, category, gate results, findings."""
    from .config import GATES
    gates = gates or GATES
    services = services or list(TAXONOMY)
    recs = []
    for r in silver.to_dict("records"):
        base = {k: r.get(k) for k in GOLD_CARRY}
        fflags = json.loads(r["facility_flags_json"])
        ffind = json.loads(r["facility_findings_json"])
        parsed = parse_record(r)  # parse JSON text fields once, reuse for all services
        for s in services:
            m = match_service(parsed, s)
            kw = keyword_hit(parsed, s)
            ex = exact_term_hit(parsed, s)
            if m.evidence_level == "none" and not kw and not ex:
                continue
            category, claim_find = evidence_category(m, ffind)
            failed = eligibility(category, m.primary_excerpt, float(r["identity_confidence_score"]),
                                 float(r["location_confidence_score"]), ffind, claim_find,
                                 {"facility_id": r["facility_id"], "source_row_id": r["source_row_id"],
                                  "evidence_field": m.primary_excerpt_field}, gates)
            recs.append({
                **base, "service": s, "evidence_level": m.evidence_level, "is_candidate": m.is_candidate,
                "evidence_category": category, "eligible": not failed, "failed_gates_json": _dumps(failed),
                "claim_findings_json": _dumps(claim_find),
                "baseline_keyword_hit": kw, "baseline_exact_term_hit": ex,
                "service_relevance_score": m.service_match_score,
                "corroboration_score": m.evidence_confidence_score,
                "evidence_strength_score": evidence_strength_score(category, m.evidence_confidence_score),
                "readiness_evidence_score": m.readiness_score,
                "n_distinct_statements": m.n_distinct_statements,
                "has_description_statement": m.has_description_statement,
                "matched_fields_json": _dumps(m.matched_fields), "matched_terms_json": _dumps(m.matched_terms),
                "evidence_excerpt": m.primary_excerpt, "evidence_field": m.primary_excerpt_field,
                "excerpts_json": _dumps(m.excerpts), "specialty_tags_json": _dumps(m.specialty_tags),
                "negated_excerpts_json": _dumps(m.negated_excerpts),
                "missing_corroboration_json": _dumps(m.missing_corroboration),
                "flags_json": _dumps(sorted(set(fflags) | set(m.conflicts))),
                "pipeline_version": PIPELINE_VERSION,
            })
    gold = pd.DataFrame(recs)
    assert not gold.duplicated(["facility_id", "service"]).any(), "gold must be unique per (facility_id, service)"
    bad = gold[gold["eligible"] & gold["evidence_excerpt"].isna()]
    assert bad.empty, f"{len(bad)} eligible rows lack an evidence excerpt"
    return gold


def build_quarantine(silver: pd.DataFrame, gold: pd.DataFrame, rejected_rows: pd.DataFrame) -> pd.DataFrame:
    """careproof_facilities_quarantine: one row per (record, [service], reason code)."""
    rows = []
    for r in rejected_rows.itertuples():
        rows.append({"source_row_id": r.source_row_id, "facility_id": None, "facility_name": None, "service": None,
                     "reason_code": "CORRUPTED_SOURCE_ROW", "severity": "high",
                     "evidence": f"{r.quarantine_reason}: {str(r.unique_id)[:200]}", "human_review_required": True})
    for r in silver.itertuples():
        for f in json.loads(r.facility_findings_json):
            if f["severity"] in ("high", "medium"):
                rows.append({"source_row_id": r.source_row_id, "facility_id": r.facility_id, "facility_name": r.name,
                             "service": None, "reason_code": f["code"], "severity": f["severity"],
                             "evidence": f["evidence"], "human_review_required": f["human_review_required"]})
    claims = gold[gold["is_candidate"] | gold["evidence_level"].isin(["negated_only", "other_facility_only"])]
    for r in claims.itertuples():
        for f in json.loads(r.claim_findings_json):
            if f["severity"] in ("high", "medium"):
                rows.append({"source_row_id": r.source_row_id, "facility_id": r.facility_id, "facility_name": r.name,
                             "service": r.service, "reason_code": f["code"], "severity": f["severity"],
                             "evidence": f["evidence"], "human_review_required": f["human_review_required"]})
    q = pd.DataFrame(rows)
    q["pipeline_version"] = PIPELINE_VERSION
    return q


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
def _haversine_vec(lat, lon, lats, lons):
    p1, p2 = np.radians(lat), np.radians(lats.astype(float))
    dp, dl = p2 - p1, np.radians(lons.astype(float) - lon)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


ANSWERED = "answered"
ABSTAIN_NO_ELIGIBLE = "abstain_no_eligible_candidate"
ABSTAIN_NO_MENTIONS = "abstain_no_mentions_in_radius"
ABSTENTION_MESSAGE = ("We found records mentioning this service, but none had sufficiently consistent identity, "
                      "location, and supporting evidence for a reliable recommendation.")
NO_MENTION_MESSAGE = ("No facility record within this radius mentions this service. Absence from this dataset does "
                      "not mean the service is unavailable; try a larger radius or contact the district health office.")


def _loads(x):
    return json.loads(x) if isinstance(x, str) else (x or [])


def recommend(gold: pd.DataFrame, loc: ResolvedLocation, service_key: str, max_distance_km: float, top_k: int,
              weights: ScoringWeights = DEFAULT_WEIGHTS):
    """Retrieve -> (precomputed) assess -> hard gate -> rank eligible only -> alternatives / abstention."""
    sdef = TAXONOMY[service_key]
    g = gold[(gold["service"] == service_key) & gold["location_lat"].notna()].copy()
    g["distance_km"] = _haversine_vec(loc.lat, loc.lon, g["location_lat"].values, g["location_lon"].values)
    near = g[g["distance_km"] <= max_distance_km]
    mentions = near[near["evidence_level"] != "none"]          # retrieval stage (broad)
    elig = mentions[mentions["eligible"].astype(bool)]           # gate stage
    held = mentions[~mentions["eligible"].astype(bool)]
    summary = {
        "service": service_key, "service_label": sdef.label, "location": loc.label, "location_method": loc.method,
        "location_warnings": loc.warnings, "max_distance_km": max_distance_km,
        "retrieved_mentions_in_radius": int(len(mentions)),
        "eligible_in_radius": int(len(elig)),
        "held_back_in_radius": int(len(held)),
        "held_back_by_category": held["evidence_category"].value_counts().to_dict(),
        "pipeline_version": PIPELINE_VERSION,
    }
    results = []
    for r in elig.to_dict("records"):
        ffind, cfind = _loads(r["facility_findings_json"]), _loads(r["claim_findings_json"])
        sc = score_candidate({**r, "findings": ffind + cfind}, r["distance_km"], weights)
        results.append(_result_row(r, sc, ffind, cfind))
    results.sort(key=rank_key)
    results = collapse_duplicates(results)
    alternatives_available = max(0, len(results) - top_k)
    results = results[:top_k]
    for i, r in enumerate(results, 1):
        r["rank"] = i
    for i, r in enumerate(results):
        r["explanation"] = explain(r, sdef.label, results[i + 1] if i + 1 < len(results) else None,
                                   results[i - 1] if i > 0 else None)
    summary["status"] = ANSWERED if results else (ABSTAIN_NO_ELIGIBLE if len(mentions) else ABSTAIN_NO_MENTIONS)
    summary["message"] = None if results else (ABSTENTION_MESSAGE if len(mentions) else NO_MENTION_MESSAGE)
    summary["additional_eligible_beyond_top_k"] = alternatives_available
    summary["held_back"] = held_back_list(held)
    validate_recommendations(results)
    return results, summary


def _result_row(r, sc, ffind, cfind) -> dict:
    from .config import AVAILABILITY_NOT_VERIFIED, EVIDENCE_LABELS
    idw = [f["evidence"] for f in ffind if f["code"] in IDENTITY_CODES]
    locw = [f["evidence"] for f in ffind if f["code"] not in IDENTITY_CODES]
    missing, contra = warnings_for(_loads(r["flags_json"]), _loads(r["missing_corroboration_json"]))
    return {
        "facility_id": r["facility_id"], "source_row_id": r["source_row_id"], "facility_name": r["name"],
        "address": ", ".join(x for x in [clean_str(r["address_line1"]), clean_str(r["address_city"]),
                                         clean_str(r["address_stateOrRegion"]), clean_str(r["address_zipOrPostcode"])] if x),
        "distance_km": round(float(r["distance_km"]), 2), "location_source": r["location_source"],
        "location_lat": float(r["location_lat"]), "location_lon": float(r["location_lon"]),
        "name_norm": normalize_name(r["name"]),
        "evidence_category": r["evidence_category"], "evidence_label": EVIDENCE_LABELS[r["evidence_category"]],
        "availability_status": AVAILABILITY_NOT_VERIFIED,
        **sc,
        "matched_fields": _loads(r["matched_fields_json"]), "matched_terms": _loads(r["matched_terms_json"]),
        "specialty_tags": _loads(r["specialty_tags_json"]),
        "evidence_field": r["evidence_field"], "evidence_excerpt": r["evidence_excerpt"],
        "all_excerpts": _loads(r["excerpts_json"]),
        "source_url": r["primary_source_url"], "source_urls": _loads(r["source_urls_list_json"])[:5],
        "identity_warnings": idw, "location_warnings": locw,
        "missing_data_warnings": missing, "contradiction_warnings": contra,
        "findings": ffind + cfind, "flags": _loads(r["flags_json"]),
        "verification_warning": VERIFICATION_WARNING, "pipeline_version": r.get("pipeline_version", PIPELINE_VERSION),
    }


def held_back_list(held: pd.DataFrame, n: int = 10) -> list:
    """Nearest records that mention the service but failed a gate (planner/audit view, never a recommendation)."""
    out = []
    for r in held.sort_values(["distance_km", "facility_id"]).head(n).to_dict("records"):
        out.append({"facility_id": r["facility_id"], "facility_name": r["name"],
                    "distance_km": round(float(r["distance_km"]), 2), "evidence_category": r["evidence_category"],
                    "failed_gates": _loads(r["failed_gates_json"]),
                    "reasons": [f"{f['code']} ({f['severity']}): {f['evidence']}" for f in
                                _loads(r["facility_findings_json"]) + _loads(r["claim_findings_json"])
                                if f["severity"] in ("high", "medium")][:3],
                    "evidence_excerpt": r["evidence_excerpt"]})
    return out


def collapse_duplicates(ranked, radius_km: float = 1.0):
    """Keep the best-ranked of records with the same normalized name within ``radius_km``; note the others."""
    kept = []
    for r in ranked:
        twin = next((k for k in kept if r["name_norm"] and k["name_norm"] == r["name_norm"] and
                     haversine_km(k["location_lat"], k["location_lon"], r["location_lat"], r["location_lon"]) <= radius_km),
                    None)
        if twin is None:
            r.setdefault("duplicate_records_collapsed", [])
            kept.append(r)
        else:
            twin["duplicate_records_collapsed"].append(r["facility_id"])
            msg = (f"A near-identical record ({r['facility_id']}) was merged into this result; the listing may be "
                   "duplicated in the source data.")
            if msg not in twin["missing_data_warnings"]:
                twin["missing_data_warnings"].append(msg)
    return kept


def validate_recommendations(results):
    for r in results:
        if not (r.get("evidence_excerpt") or "").strip():
            raise AssertionError(f"Recommendation {r['facility_id']} has no evidence excerpt")
        if not r.get("evidence_field") or not r.get("source_row_id"):
            raise AssertionError(f"Recommendation {r['facility_id']} lacks provenance")
        if r["evidence_category"] in ("conflicting", "insufficient"):
            raise AssertionError(f"Recommendation {r['facility_id']} has {r['evidence_category']} evidence")
    ids = [r["facility_id"] for r in results]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate facility_id in recommendations")
    for r in results:
        for k in COMPONENTS + ("final_ranking_score", "data_quality_penalty"):
            if not 0.0 <= r[k] <= 1.0:
                raise AssertionError(f"{k}={r[k]} out of [0,1] for {r['facility_id']}")


def keyword_baseline(gold: pd.DataFrame, loc: ResolvedLocation, service_key: str, max_distance_km: float, top_k: int,
                     column: str = "baseline_keyword_hit"):
    """Nearest facility whose text matches the service (B3 synonyms: baseline_keyword_hit; B2 exact word:
    baseline_exact_term_hit). Distance only; no negation, no contradiction checks."""
    g = gold[(gold["service"] == service_key) & gold[column].astype(bool) & gold["location_lat"].notna()].copy()
    g["distance_km"] = _haversine_vec(loc.lat, loc.lon, g["location_lat"].values, g["location_lon"].values)
    g = g[g["distance_km"] <= max_distance_km].sort_values(["distance_km", "facility_id"]).head(top_k)
    return [{"rank": i, "facility_id": r["facility_id"], "facility_name": r["name"],
             "distance_km": round(float(r["distance_km"]), 2), "evidence_level": r["evidence_level"],
             "flags": json.loads(r["flags_json"])} for i, r in enumerate(g.to_dict("records"), 1)]


def nfhs_context(nfhs: pd.DataFrame, district: Optional[str], state: Optional[str], service_key: str) -> pd.DataFrame:
    """District-level community context. NOT a facility quality signal and NOT patient-level."""
    inds = TAXONOMY[service_key].nfhs_context
    if not district or not inds:
        return pd.DataFrame()
    m = nfhs[(nfhs["district_norm"] == district) & nfhs["indicator"].isin(inds)]
    if state:
        m = m[m["state_norm"] == state]
    return m[["district_name_raw", "state_ut_raw", "indicator", "raw_value", "value", "is_suppressed", "is_small_sample"]]


def nearest_any_baseline(silver: pd.DataFrame, loc: ResolvedLocation, max_distance_km: float, top_k: int):
    """B1: nearest facility of any kind, ignoring the requested service."""
    g = silver[silver["location_lat"].notna()][["facility_id", "name", "location_lat", "location_lon"]].copy()
    g["distance_km"] = _haversine_vec(loc.lat, loc.lon, g["location_lat"].values, g["location_lon"].values)
    g = g[g["distance_km"] <= max_distance_km].sort_values(["distance_km", "facility_id"]).head(top_k)
    return [{"rank": i, "facility_id": r["facility_id"], "facility_name": r["name"],
             "distance_km": round(float(r["distance_km"]), 2)} for i, r in enumerate(g.to_dict("records"), 1)]


# ---------------------------------------------------------------------------
# Type coercion for tables read back from Spark/SQL (strings, Arrow types or Parquet) -> pipeline dtypes
# ---------------------------------------------------------------------------
GOLD_NUMERIC = ("location_lat", "location_lon", "location_quality_score", "location_confidence_score",
                "identity_confidence_score", "service_relevance_score", "corroboration_score", "evidence_strength_score",
                "readiness_evidence_score", "facility_to_pin_km", "n_distinct_statements")
GOLD_BOOL = ("is_candidate", "eligible", "has_description_statement", "baseline_keyword_hit", "baseline_exact_term_hit")


def _to_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: v if isinstance(v, (bool, np.bool_)) else str(v).strip().lower() == "true").astype(bool)


def coerce_gold_types(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    for c in GOLD_NUMERIC:
        if c in g:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    for c in GOLD_BOOL:
        if c in g:
            g[c] = _to_bool(g[c])
    return g


def coerce_pin_types(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    for c in ("centroid_lat", "centroid_lon", "n_offices_with_valid_coords", "n_offices_coord_outliers",
              "post_office_count", "n_districts", "n_states", "spread_km"):
        if c in p:
            p[c] = pd.to_numeric(p[c], errors="coerce")
    for c in ("coords_defensible", "is_ambiguous", "missing_coordinates"):
        if c in p:
            p[c] = _to_bool(p[c])
    return p


def spark_column_values(s: pd.Series):
    """(kind, python_values) for building a Spark DataFrame with an explicit schema.

    Missing values become real None (SQL NULL), never float NaN: pandas `.map` on a float column silently turns
    None back into NaN, which Spark would store as a NaN DOUBLE that is NOT NULL in SQL.
    """
    def missing(v):
        return v is None or v is pd.NA or (isinstance(v, float) and v != v)
    nn = [v for v in s.tolist() if not missing(v)]
    if pd.api.types.is_bool_dtype(s) or (nn and all(isinstance(v, (bool, np.bool_)) for v in nn)):
        return "boolean", [None if missing(v) else bool(v) for v in s.tolist()]
    if pd.api.types.is_integer_dtype(s):
        return "long", [None if missing(v) else int(v) for v in s.tolist()]
    if pd.api.types.is_float_dtype(s) or (nn and all(isinstance(v, (int, float, np.integer, np.floating))
                                                    and not isinstance(v, (bool, np.bool_)) for v in nn)):
        return "double", [None if missing(v) else float(v) for v in s.tolist()]
    return "string", [None if missing(v) else str(v) for v in s.tolist()]
