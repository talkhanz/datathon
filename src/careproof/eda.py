"""Question-driven exploratory analysis. Each function answers ONE question and returns a pandas DataFrame.

Used by scripts/run_local.py (writes docs/data_quality_report.md + docs/figures) and by
notebooks/01b_exploratory_analysis.py (Databricks), so local and Databricks findings come from identical code.
Inputs are the raw facility frame (all strings), the PIN raw frame, NFHS raw frame, and the silver/gold outputs.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .normalization import clean_str, is_missing, parse_array, validate_coordinates

CONTRA_FLAGS = ["location_conflict", "state_conflict", "specialty_mismatch", "negated_mention",
                "suspicious_source_url", "possible_duplicate"]


def _miss(s: pd.Series) -> pd.Series:
    return s.map(lambda v: is_missing(v) or (isinstance(v, str) and v.strip() in ("[]", "{}")))


# Q1 ---------------------------------------------------------------------------------------------------------
def q1_structure(fac_raw: pd.DataFrame, silver: pd.DataFrame, quarantine: pd.DataFrame) -> pd.DataFrame:
    """What is the grain, and how many records are broken, duplicated, or near-duplicated?"""
    s = silver
    same_coord = s[s.facility_lat.notna()].groupby(["facility_lat", "facility_lon"]).facility_id.transform("size") > 1
    same_coord_diff_name = s[s.facility_lat.notna()].groupby(["facility_lat", "facility_lon"]).facility_name_norm.transform("nunique") > 1
    rows = [
        ("raw rows (parsed)", len(fac_raw)),
        ("quarantined: structurally corrupted rows", len(quarantine)),
        ("exact duplicate rows removed", int((s.n_exact_source_copies - 1).sum())),
        ("silver facilities", len(s)),
        ("distinct unique_id in silver", s.unique_id.nunique()),
        ("records sharing exact coordinates with another record", int(same_coord.sum())),
        ("...of which the co-located records have different names", int(same_coord_diff_name.sum())),
        ("same normalized name within 1 km (possible duplicate)", int(s.possible_duplicate.sum())),
        ("content_table_id shared by >1 facility (not an identifier)",
         int(s.content_table_id.duplicated(keep=False).sum()) if "content_table_id" in s else None),
    ]
    return pd.DataFrame(rows, columns=["question: structure & duplicates", "count"])


# Q2 ---------------------------------------------------------------------------------------------------------
EVIDENCE_FIELDS = ["description", "procedure", "equipment", "capability", "specialties", "source_urls",
                   "numberDoctors", "capacity", "yearEstablished"]


def q2_missingness_overall(silver: pd.DataFrame) -> pd.DataFrame:
    """How complete are the evidence fields?"""
    out = [(c, round(100 * _miss(silver[c]).mean(), 1)) for c in EVIDENCE_FIELDS if c in silver]
    return pd.DataFrame(out, columns=["field", "pct_missing"]).sort_values("pct_missing", ascending=False)


def q2_missingness_by_group(silver: pd.DataFrame, group: str, min_n: int = 50) -> pd.DataFrame:
    """Is missingness systematic by state / facility type? (groups with >= min_n facilities)"""
    df = silver.copy()
    df["_g"] = df[group].fillna("(missing)")
    keep = df["_g"].value_counts()
    df = df[df["_g"].isin(keep[keep >= min_n].index)]
    rows = []
    for g, d in df.groupby("_g"):
        rows.append({group: g, "n": len(d), **{f"pct_missing_{c}": round(100 * _miss(d[c]).mean(), 1)
                                              for c in ("procedure", "equipment", "capacity", "numberDoctors")}})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def q2_co_missingness(silver: pd.DataFrame) -> pd.DataFrame:
    """Do evidence fields go missing together (systematically) or independently?"""
    cols = ["procedure", "equipment", "capability", "capacity", "numberDoctors"]
    m = pd.DataFrame({c: _miss(silver[c]) for c in cols})
    obs = m.T.dot(m.astype(int)) / len(m)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            p_ab, p_a, p_b = obs.loc[a, b], m[a].mean(), m[b].mean()
            rows.append({"pair": f"{a} & {b}", "p_both_missing": round(p_ab, 3),
                         "p_if_independent": round(p_a * p_b, 3),
                         "lift": round(p_ab / (p_a * p_b), 2) if p_a * p_b else None})
    return pd.DataFrame(rows).sort_values("lift", ascending=False)


# Q3 ---------------------------------------------------------------------------------------------------------
def q3_location_consistency(silver: pd.DataFrame) -> pd.DataFrame:
    """How often does a facility's position disagree with its own PIN code?"""
    d = silver.facility_to_pin_km.dropna()
    bins = [0, 1, 5, 25, 100, 500, np.inf]
    labels = ["<1 km", "1-5 km", "5-25 km", "25-100 km", "100-500 km", ">500 km"]
    c = pd.cut(d, bins=bins, labels=labels, right=False).value_counts().reindex(labels)
    out = pd.DataFrame({"facility_to_own_PIN_centroid": labels, "n": c.values,
                        "pct": (100 * c.values / len(d)).round(1)})
    return out


def q3_coordinate_issues(fac_raw: pd.DataFrame, pin_raw: pd.DataFrame) -> pd.DataFrame:
    """What kinds of coordinate errors exist in each source?"""
    rows = []
    for name, df in (("facilities", fac_raw), ("india_post_pincode_directory", pin_raw)):
        issues = pd.Series([validate_coordinates(a, b).issue for a, b in zip(df.latitude, df.longitude)])
        vc = issues.fillna("valid").value_counts()
        for k, v in vc.items():
            rows.append({"source": name, "coordinate_status": k, "rows": int(v), "pct": round(100 * v / len(df), 2)})
    return pd.DataFrame(rows)


def q3_pin_ambiguity(pin_silver: pd.DataFrame) -> pd.DataFrame:
    """Can a PIN code be resolved to one place?"""
    p = pin_silver
    rows = [("PINs", len(p)), ("PINs spanning >1 district", int((p.n_districts > 1).sum())),
            ("PINs spanning >1 state", int((p.n_states > 1).sum())),
            ("PINs with no valid office coordinates", int(p.missing_coordinates.sum())),
            ("PINs with geocoded offices that disagree (not defensible)",
             int((~p.coords_defensible & ~p.missing_coordinates).sum())),
            ("PINs with a defensible centroid", int(p.coords_defensible.sum()))]
    return pd.DataFrame(rows, columns=["question: PIN resolvability", "count"])


# Q4 ---------------------------------------------------------------------------------------------------------
def q4_claims_vs_evidence(gold: pd.DataFrame) -> pd.DataFrame:
    """How many facilities *claim* a service vs state it explicitly in text?"""
    t = gold.groupby(["service", "evidence_level"]).size().unstack(fill_value=0)
    for c in ("explicit_text", "specialty_tag_only", "negated_only", "none"):
        if c not in t:
            t[c] = 0
    t = t.rename(columns={"none": "keyword_only (baseline hit, no evidence)"})
    t["explicit_share_of_any_signal_pct"] = (100 * t["explicit_text"] / t.sum(axis=1)).round(1)
    return t.reset_index()


def q4_flag_rates(gold: pd.DataFrame) -> pd.DataFrame:
    """Among recommendable (explicit-text) rows, how often does each contradiction fire?"""
    c = gold[gold.is_candidate]
    rows = []
    for f in CONTRA_FLAGS + ["evidence_list_truncated", "coordinates_from_pin"]:
        hit = c.flags_json.str.contains(f'"{f}"')
        rows.append({"flag": f, "candidate_rows": int(hit.sum()), "pct_of_candidates": round(100 * hit.mean(), 2)})
    any_c = c.flags_json.map(lambda s: bool(set(json.loads(s)) & set(CONTRA_FLAGS[:4])))
    rows.append({"flag": "ANY of location/state/specialty/negation", "candidate_rows": int(any_c.sum()),
                 "pct_of_candidates": round(100 * any_c.mean(), 2)})
    return pd.DataFrame(rows)


def q4_source_domains(silver: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """Where does the evidence come from? (record-level source URLs)"""
    cnt = {}
    for s in silver.source_urls_list_json:
        for d in {urlparse(u).netloc.lower().removeprefix("www.") for u in json.loads(s) if u.startswith("http")}:
            cnt[d] = cnt.get(d, 0) + 1
    out = pd.Series(cnt).sort_values(ascending=False).head(top)
    return pd.DataFrame({"domain": out.index, "facilities_citing": out.values,
                         "pct_facilities": (100 * out.values / len(silver)).round(1)})


def q4_description_quality(silver: pd.DataFrame) -> pd.DataFrame:
    """Are descriptions informative, and in which script are they written?"""
    d = silver.description.map(lambda x: clean_str(x) or "")
    words = d.str.split().map(len)
    non_ascii = d.map(lambda x: bool(re.search(r"[^\x00-\x7F]", x)))
    deva = d.map(lambda x: bool(re.search(r"[ऀ-ॿ]", x)))
    q = words.quantile([.1, .25, .5, .75, .9]).round(0)
    rows = [("descriptions present", int((d != "").sum())), ("median words", int(q[.5])),
            ("p10 / p90 words", f"{int(q[.1])} / {int(q[.9])}"),
            ("descriptions with <= 10 words", int((words <= 10).sum())),
            ("contain any non-ASCII character", int(non_ascii.sum())),
            ("contain Devanagari script", int(deva.sum()))]
    return pd.DataFrame(rows, columns=["question: description quality", "value"])


# Q5 ---------------------------------------------------------------------------------------------------------
def q5_nfhs_quality(nfhs_silver: pd.DataFrame) -> pd.DataFrame:
    """Which NFHS indicators are least reliable (suppressed or small-sample)?"""
    g = nfhs_silver.groupby("indicator").agg(n=("value", "size"), suppressed=("is_suppressed", "sum"),
                                             small_sample=("is_small_sample", "sum"))
    g["pct_unreliable"] = (100 * (g.suppressed + g.small_sample) / g.n).round(1)
    return g.sort_values("pct_unreliable", ascending=False).reset_index()


def q5_nfhs_pct_out_of_range(nfhs_silver: pd.DataFrame) -> pd.DataFrame:
    """Do any *_pct indicators fall outside [0, 100]?"""
    p = nfhs_silver[nfhs_silver.indicator.str.endswith("_pct") & nfhs_silver.value.notna()]
    bad = p[(p.value < 0) | (p.value > 100)]
    return bad[["district_name_raw", "state_ut_raw", "indicator", "raw_value"]]


def q5_nfhs_join_coverage(nfhs_silver: pd.DataFrame, pin_silver: pd.DataFrame) -> pd.DataFrame:
    """How many NFHS districts can be matched to India Post districts by normalized name + state?"""
    nf = nfhs_silver[["district_norm", "state_norm"]].drop_duplicates()
    pd_ = pin_silver[pin_silver.primary_district.notna()][["primary_district", "primary_state"]].drop_duplicates()
    m = nf.merge(pd_, how="left", left_on=["district_norm", "state_norm"],
                 right_on=["primary_district", "primary_state"], indicator=True)
    n_match = int((m._merge == "both").sum())
    return pd.DataFrame([("NFHS districts", len(nf)), ("matched to India Post district+state", n_match),
                         ("match rate %", round(100 * n_match / len(nf), 1)),
                         ("NFHS rows with unrecognised state", int(nfhs_silver.state_norm.isna().sum() /
                                                                   max(1, nfhs_silver.indicator.nunique())))],
                        columns=["question: NFHS geographic join", "value"])


# Q6 ---------------------------------------------------------------------------------------------------------
def district_rurality_proxy(pin_raw: pd.DataFrame) -> pd.DataFrame:
    """Share of Branch Offices (BO) among a district's post offices.

    PROXY, not a census measure: India Post branch offices are predominantly village offices, so a higher BO
    share suggests a more rural district. Used only to look for coverage bias.
    """
    from .normalization import normalize_district, normalize_state
    df = pin_raw.assign(d=pin_raw.district.map(normalize_district), s=pin_raw.statename.map(normalize_state))
    df = df[df.d.notna() & df.s.notna()]
    g = df.groupby(["d", "s"]).agg(offices=("officetype", "size"),
                                   bo_share=("officetype", lambda x: (x == "BO").mean()))
    return g.reset_index().rename(columns={"d": "district", "s": "state"})


def q6_coverage_by_rurality(silver: pd.DataFrame, gold: pd.DataFrame, pin_raw: pd.DataFrame) -> pd.DataFrame:
    """Are rural districts (proxy) systematically under-covered or worse-evidenced?"""
    rur = district_rurality_proxy(pin_raw)
    s = silver[silver.pin_primary_district.notna()].copy()
    s["district"] = s.pin_primary_district
    st = json.loads  # noqa: F841
    s["state"] = s.pin_states_json.map(lambda x: (json.loads(x) or [None])[0] if isinstance(x, str) else None)
    cand_ids = set(gold[gold.is_candidate].facility_id)
    s["has_explicit_any"] = s.facility_id.isin(cand_ids)
    s["loc_conflict"] = s.facility_flags_json.str.contains('"location_conflict"')
    per = s.groupby(["district", "state"]).agg(facilities=("facility_id", "size"),
                                                 pct_with_explicit_evidence=("has_explicit_any", "mean"),
                                                 pct_location_conflict=("loc_conflict", "mean")).reset_index()
    m = rur.merge(per, on=["district", "state"], how="left").fillna({"facilities": 0})
    m["rurality_quartile"] = pd.qcut(m.bo_share.rank(method="first"), 4,
                                     labels=["Q1 most urban", "Q2", "Q3", "Q4 most rural"])
    out = m.groupby("rurality_quartile", observed=True).agg(
        districts=("district", "size"), median_bo_share=("bo_share", "median"),
        districts_with_zero_facilities=("facilities", lambda x: int((x == 0).sum())),
        total_facilities=("facilities", "sum"), median_facilities_per_district=("facilities", "median"),
        mean_pct_with_explicit_evidence=("pct_with_explicit_evidence", "mean"),
        mean_pct_location_conflict=("pct_location_conflict", "mean")).reset_index()
    for c in ("median_bo_share", "mean_pct_with_explicit_evidence", "mean_pct_location_conflict"):
        out[c] = (100 * out[c]).round(1) if c != "median_bo_share" else out[c].round(3)
    return out


def q6_state_coverage(silver: pd.DataFrame, gold: pd.DataFrame, top: int = 40) -> pd.DataFrame:
    """Which states have the fewest credibly evidenced facilities for critical services?"""
    s = silver.copy()
    s["state"] = s.state_norm.fillna("(unrecognised)")
    c = gold[gold.is_candidate].merge(s[["facility_id", "state"]], on="facility_id")
    t = c.groupby(["state", "service"]).facility_id.nunique().unstack(fill_value=0)
    t["facilities_total"] = s.groupby("state").size()
    return t.sort_values("facilities_total", ascending=False).head(top).reset_index()


ALL_QUESTIONS = [
    ("Q1", "Structure, grain and duplicates", "q1_structure"),
    ("Q2a", "Completeness of evidence fields", "q2_missingness_overall"),
    ("Q2b", "Missingness by facility type", "q2_by_type"),
    ("Q2c", "Missingness by state", "q2_by_state"),
    ("Q2d", "Co-missingness (lift > 1 = fields go missing together)", "q2_co_missingness"),
    ("Q3a", "Coordinate validity in each source", "q3_coordinate_issues"),
    ("Q3b", "Facility position vs its own PIN centroid", "q3_location_consistency"),
    ("Q3c", "PIN resolvability", "q3_pin_ambiguity"),
    ("Q4a", "Claims vs explicit evidence per service", "q4_claims_vs_evidence"),
    ("Q4b", "Contradiction/data-quality flags among recommendable rows", "q4_flag_rates"),
    ("Q4c", "Description quality and script", "q4_description_quality"),
    ("Q4d", "Most-cited source domains", "q4_source_domains"),
    ("Q5a", "Least reliable NFHS indicators", "q5_nfhs_quality"),
    ("Q5b", "NFHS percentage indicators outside [0,100]", "q5_nfhs_pct_out_of_range"),
    ("Q5c", "NFHS geographic join coverage", "q5_nfhs_join_coverage"),
    ("Q6a", "Coverage by district rurality (BO-share proxy)", "q6_coverage_by_rurality"),
    ("Q6b", "Explicitly evidenced facilities per service, by state", "q6_state_coverage"),
]


def run_all(fac_raw, pin_raw, silver, quarantine, pin_silver, nfhs_silver, gold) -> dict:
    return {
        "Q1": q1_structure(fac_raw, silver, quarantine),
        "Q2a": q2_missingness_overall(silver),
        "Q2b": q2_missingness_by_group(silver, "facilityTypeId", 10),
        "Q2c": q2_missingness_by_group(silver.assign(state=silver.state_norm), "state", 100),
        "Q2d": q2_co_missingness(silver),
        "Q3a": q3_coordinate_issues(fac_raw, pin_raw),
        "Q3b": q3_location_consistency(silver),
        "Q3c": q3_pin_ambiguity(pin_silver),
        "Q4a": q4_claims_vs_evidence(gold),
        "Q4b": q4_flag_rates(gold),
        "Q4c": q4_description_quality(silver),
        "Q4d": q4_source_domains(silver),
        "Q5a": q5_nfhs_quality(nfhs_silver).head(12),
        "Q5b": q5_nfhs_pct_out_of_range(nfhs_silver),
        "Q5c": q5_nfhs_join_coverage(nfhs_silver, pin_silver),
        "Q6a": q6_coverage_by_rurality(silver, gold, pin_raw),
        "Q6b": q6_state_coverage(silver, gold),
    }
