"""Table-building and recommendation logic on pandas DataFrames.

Used identically by the local runner (CSV input) and the Databricks notebooks
(Marketplace tables -> toPandas). All inputs are expected as *strings or None*
per cell; the notebooks cast complex Spark types to JSON strings first.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_WEIGHTS, LOCATION_CONFLICT_KM, ScoringWeights, VERIFICATION_WARNING
from .explanations import explain, warnings_for
from .geography import EARTH_RADIUS_KM, PlaceIndex, ResolvedLocation, aggregate_pin, haversine_km
from .normalization import (
    clean_str, is_valid_uuid, normalize_district, normalize_name, normalize_pin, normalize_state,
    parse_array, parse_nfhs_value, validate_coordinates,
)
from .scoring import location_quality_score, rank_key, score_candidate
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
                          "primary_district", "is_ambiguous"]].rename(columns={
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
        if not r.state_norm or not isinstance(r.pin_states_json, str):
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
    reports["silver_rows"] = len(df)
    reports["location_source_counts"] = df["location_source"].value_counts().to_dict()
    print(f"[facilities] input={reports['input_rows']} quarantined={reports['quarantined_rows']} "
          f"exact_dups_removed={reports['exact_duplicate_rows_removed']} silver={len(df)}")
    return df.reset_index(drop=True), quarantine.reset_index(drop=True), reports


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
              "location_quality_score", "facility_to_pin_km", "primary_source_url", "source_urls_list_json",
              "facility_flags_json"]


def build_facility_service_gold(silver: pd.DataFrame, services=None) -> pd.DataFrame:
    services = services or list(TAXONOMY)
    recs = []
    for r in silver.to_dict("records"):
        base = {k: r.get(k) for k in GOLD_CARRY}
        fflags = json.loads(r["facility_flags_json"])
        parsed = parse_record(r)  # parse JSON text fields once, reuse for all services
        for s in services:
            m = match_service(parsed, s)
            kw = keyword_hit(parsed, s)
            ex = exact_term_hit(parsed, s)
            if m.evidence_level == "none" and not kw and not ex:
                continue
            all_flags = sorted(set(fflags) | set(m.conflicts))
            recs.append({
                **base, "service": s, "evidence_level": m.evidence_level, "is_candidate": m.is_candidate,
                "baseline_keyword_hit": kw, "baseline_exact_term_hit": ex,
                "service_match_score": m.service_match_score,
                "evidence_confidence_score": m.evidence_confidence_score,
                "readiness_score": m.readiness_score,
                "matched_fields_json": _dumps(m.matched_fields), "matched_terms_json": _dumps(m.matched_terms),
                "evidence_excerpt": m.primary_excerpt, "evidence_field": m.primary_excerpt_field,
                "excerpts_json": _dumps(m.excerpts), "specialty_tags_json": _dumps(m.specialty_tags),
                "negated_excerpts_json": _dumps(m.negated_excerpts),
                "missing_corroboration_json": _dumps(m.missing_corroboration),
                "flags_json": _dumps(all_flags),
            })
    gold = pd.DataFrame(recs)
    assert not gold.duplicated(["facility_id", "service"]).any(), "gold must be unique per (facility_id, service)"
    bad = gold[gold["is_candidate"] & gold["evidence_excerpt"].isna()]
    assert bad.empty, f"{len(bad)} candidate rows lack an evidence excerpt"
    return gold


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
def _haversine_vec(lat, lon, lats, lons):
    p1, p2 = np.radians(lat), np.radians(lats.astype(float))
    dp, dl = p2 - p1, np.radians(lons.astype(float) - lon)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def recommend(gold: pd.DataFrame, loc: ResolvedLocation, service_key: str, max_distance_km: float, top_k: int,
              weights: ScoringWeights = DEFAULT_WEIGHTS):
    sdef = TAXONOMY[service_key]
    g = gold[(gold["service"] == service_key) & gold["location_lat"].notna()].copy()
    g["distance_km"] = _haversine_vec(loc.lat, loc.lon, g["location_lat"].values, g["location_lon"].values)
    near = g[g["distance_km"] <= max_distance_km]
    cands = near[near["is_candidate"]]
    summary = {
        "service": service_key, "service_label": sdef.label, "location": loc.label, "location_method": loc.method,
        "location_warnings": loc.warnings, "max_distance_km": max_distance_km,
        "explicit_text_candidates_in_radius": int(len(cands)),
        "specialty_tag_only_excluded": int((near["evidence_level"] == "specialty_tag_only").sum()),
        "negated_only_excluded": int((near["evidence_level"] == "negated_only").sum()),
        "facilities_without_usable_location": int(((gold["service"] == service_key) & gold["location_lat"].isna()
                                                    & gold["is_candidate"]).sum()),
    }
    results = []
    for r in cands.to_dict("records"):
        flags = json.loads(r["flags_json"])
        sc = score_candidate({**r, "flags": flags}, r["distance_km"], weights)
        missing, contra = warnings_for(flags, json.loads(r["missing_corroboration_json"]))
        neg = json.loads(r["negated_excerpts_json"])
        if neg:
            contra.append(f'Contradicting text ({neg[0]["field"]}): "{neg[0]["text"][:160]}"')
        results.append({
            "facility_id": r["facility_id"], "facility_name": r["name"],
            "address": ", ".join(x for x in [clean_str(r["address_line1"]), clean_str(r["address_city"]),
                                             clean_str(r["address_stateOrRegion"]), clean_str(r["address_zipOrPostcode"])] if x),
            "distance_km": round(float(r["distance_km"]), 2), "location_source": r["location_source"],
            "location_lat": float(r["location_lat"]), "location_lon": float(r["location_lon"]),
            "name_norm": normalize_name(r["name"]),
            **sc,
            "matched_fields": json.loads(r["matched_fields_json"]), "matched_terms": json.loads(r["matched_terms_json"]),
            "specialty_tags": json.loads(r["specialty_tags_json"]),
            "evidence_field": r["evidence_field"], "evidence_excerpt": r["evidence_excerpt"],
            "all_excerpts": json.loads(r["excerpts_json"]),
            "source_url": r["primary_source_url"],
            "missing_data_warnings": missing, "contradiction_warnings": contra, "flags": flags,
            "verification_warning": VERIFICATION_WARNING,
        })
    results.sort(key=rank_key)
    results = collapse_duplicates(results)
    n_ranked = len(results)
    results = results[:top_k]
    summary["status"] = abstention_status(results, n_ranked)
    for i, r in enumerate(results, 1):
        r["rank"] = i
        r["explanation"] = explain(r, sdef.label)
    validate_recommendations(results)
    return results, summary


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


ABSTAIN_NO_EVIDENCE = "abstain_no_explicit_evidence_in_radius"
LOW_CONFIDENCE_ONLY = "answered_low_confidence_only"
ANSWERED = "answered"


def abstention_status(results, n_ranked) -> str:
    """Explicit abstention semantics (see docs/system_card.md)."""
    if not results:
        return ABSTAIN_NO_EVIDENCE
    if all(r["confidence"] == "Low" for r in results):
        return LOW_CONFIDENCE_ONLY
    return ANSWERED


def validate_recommendations(results):
    for r in results:
        if not (r.get("evidence_excerpt") or "").strip():
            raise AssertionError(f"Recommendation {r['facility_id']} has no evidence excerpt")
        if not r.get("evidence_field"):
            raise AssertionError(f"Recommendation {r['facility_id']} has no evidence field (provenance)")
    ids = [r["facility_id"] for r in results]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate facility_id in recommendations")
    for r in results:
        for k in ("service_match_score", "evidence_confidence_score", "location_quality_score", "distance_score",
                  "readiness_score", "final_score", "data_quality_penalty"):
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
