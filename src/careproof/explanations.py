"""Plain-language explanations: what is supported, what is inferred, what is unknown, what must be verified.

Wording rule: never state that a facility offers a service. Say what the record documents
("The available record contains moderate evidence that this facility may offer dialysis").
Answers the challenge's two questions for every result: WHY this facility, HOW it was ranked.
"""
from __future__ import annotations

from typing import Mapping, Optional

from .config import EVIDENCE_LABELS, VERIFICATION_WARNING

CONTRADICTION_TEXT = {
    "negated_mention": "Some text for this facility says the service is not available, not provided, planned, "
                       "historical, or provided through another facility.",
    "other_facility_attribution": "Some passages attribute the service to a different named facility.",
    "specialty_mismatch": "The facility's name/type suggests a different specialty.",
    "location_conflict": "Coordinates are far (> 25 km) from the facility's own listed PIN code.",
    "state_conflict": "The listed state does not match the state of the listed PIN code.",
    "suspicious_source_url": "Source URLs include unrelated pages (e.g. login/payment pages).",
    "possible_duplicate": "Another record with the same name lies within 1 km (possible duplicate listing).",
}
MISSING_TEXT = {
    "coordinates_from_pin": "No usable facility coordinates; position estimated from its PIN code centre.",
    "pin_missing_or_invalid": "Facility PIN code is missing or invalid, so its location cannot be cross-checked.",
    "pin_not_in_directory": "Facility PIN code is not in the India Post directory.",
    "pin_centroid_unreliable": "The facility's PIN has no reliable reference location, so its coordinates are unverified.",
    "evidence_list_truncated": "Some source lists hit the 50-item cap and may be incomplete.",
    "state_unrecognised": "The state field is missing or not a recognisable Indian state.",
}
CONTRADICTION_FLAGS = set(CONTRADICTION_TEXT)
COMPONENT_NAMES = {
    "service_relevance_score": "how directly the service is named",
    "evidence_strength_score": "evidence strength",
    "identity_confidence_score": "identity consistency",
    "location_confidence_score": "location consistency",
    "readiness_evidence_score": "readiness signals (unit, equipment counts, 24x7)",
    "distance_score": "proximity",
}


def warnings_for(flags, missing_corroboration) -> tuple[list[str], list[str]]:
    contra = [CONTRADICTION_TEXT[f] for f in sorted(set(flags)) if f in CONTRADICTION_TEXT]
    missing = [MISSING_TEXT[f] for f in sorted(set(flags)) if f in MISSING_TEXT]
    missing += [f"Missing corroboration: {m}." for m in (missing_corroboration or [])]
    return missing, contra


def service_noun(label: str) -> str:
    return label.split(" (")[0].split(" / ")[0].lower()


def explain(r: Mapping, service_label: str, below: Optional[Mapping] = None, above: Optional[Mapping] = None) -> str:
    cat = r["evidence_category"]
    svc = service_noun(service_label)
    fields = [f for f in (r.get("matched_fields") or []) if f != "specialties"]
    parts = [f"The available record contains {cat} evidence that {r['facility_name']} may offer {svc}."]
    parts.append(f'Supported: its own {r["evidence_field"]} field says "{_short(r["evidence_excerpt"])}"'
                 + (f", corroborated in {len(fields)} field(s) ({', '.join(fields)})" if len(fields) > 1 else "")
                 + (f" and by related specialty tags ({', '.join(r['specialty_tags'][:2])})" if r.get("specialty_tags") else "")
                 + ".")
    parts.append("It qualified because it has an attributable supporting excerpt, adequately consistent identity and "
                 "location, and no high-severity contradiction.")
    if below is not None:
        diffs = sorted(((k, r[k] - below[k]) for k in COMPONENT_NAMES), key=lambda x: -x[1])
        top = [COMPONENT_NAMES[k] for k, d in diffs if d > 0.01][:2]
        parts.append(f"It ranks above #{below['rank']} ({below['facility_name']}) mainly on {' and '.join(top)}."
                     if top else f"It ranks above #{below['rank']} ({below['facility_name']}) on the deterministic tie-break.")
    if above is not None and r["distance_km"] < above["distance_km"]:
        parts.append(f"Although closer than #{above['rank']}, it ranks lower because that facility's documented "
                     "evidence or consistency is stronger.")
    parts.append(f"Inferred: approximate straight-line distance {r['distance_km']:.1f} km "
                 f"(position from {r['location_source'].replace('_', ' ')}); not road distance or travel time.")
    weak = list(r.get("identity_warnings", [])) + list(r.get("location_warnings", [])) + list(r.get("missing_data_warnings", []))
    if weak:
        parts.append("Remaining weaknesses: " + "; ".join(_short(w, 120) for w in weak[:3]) + ".")
    parts.append("Unknown: whether the service, specialist, equipment and admission are available today. "
                 "Verify by phone before travel.")
    return " ".join(parts)


def _short(s: str, n: int = 160) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# HTML rendering for the notebook demo (the Streamlit app has its own UI)
# ---------------------------------------------------------------------------
_BADGE = {"strong": "#1b7f3b", "moderate": "#0969da", "weak": "#9a6700", "conflicting": "#b42318",
          "insufficient": "#57606a"}


def render_cards_html(results, summary, title: str = "") -> str:
    from html import escape as e
    out = ['<div style="font-family:system-ui,sans-serif;max-width:980px;line-height:1.4;color:#1f2328">']
    if title:
        out.append(f"<h2 style='margin:4px 0'>{e(title)}</h2>")
    out.append(f"<p style='margin:4px 0 10px'><b>{e(summary['service_label'])}</b> near <b>{e(summary['location'])}</b> "
               f"within {summary['max_distance_km']:.0f} km · {summary['retrieved_mentions_in_radius']} records mention "
               f"it · <b>{summary['eligible_in_radius']} passed all credibility gates</b> · "
               f"{summary['held_back_in_radius']} held back.</p>")
    for w in summary.get("location_warnings") or []:
        out.append(f"<div style='background:#fff8c5;padding:6px 10px;border-radius:6px;margin:4px 0'>📍 {e(w)}</div>")
    out.append(f"<div style='background:#ffebe9;padding:8px 10px;border-radius:6px;margin:8px 0'>⚠ {e(VERIFICATION_WARNING)}</div>")
    if summary.get("message"):
        out.append(f"<div style='background:#fff8c5;padding:8px 10px;border-radius:6px;margin:8px 0'><b>CareProof "
                   f"abstains.</b> {e(summary['message'])}</div>")
    for r in results:
        color = _BADGE.get(r["evidence_category"], "#57606a")
        comps = "".join(f"<td style='padding:2px 8px;text-align:center'>{r[k]:.2f}</td>" for k in
                        list(COMPONENT_NAMES) + ["data_quality_penalty", "final_ranking_score"])
        heads = "".join(f"<td style='padding:2px 8px'>{h}</td>" for h in
                        ["service", "evidence", "identity", "location", "readiness", "proximity", "penalty", "<b>rank score</b>"])
        warn = "".join(f"<li style='color:#b42318'>{e(w)}</li>" for w in r["identity_warnings"] + r["location_warnings"])
        warn += "".join(f"<li style='color:#6e7781'>{e(w)}</li>" for w in r["missing_data_warnings"])
        url = r.get("source_url")
        url_html = f"<a href='{e(url)}' target='_blank' rel='noopener'>{e(url[:90])}</a>" if url else "<i>none listed</i>"
        out.append(f"""
<div style='border:1px solid #d0d7de;border-radius:8px;padding:10px 14px;margin:10px 0'>
 <div style='display:flex;justify-content:space-between'><div><b style='font-size:1.1em'>#{r['rank']} {e(r['facility_name'])}</b>
  <span style='color:#57606a'> · {e(r['address'])}</span></div>
  <span style='background:{color};color:white;border-radius:12px;padding:2px 10px;font-size:.85em'>{e(r['evidence_label'])}</span></div>
 <div style='margin:4px 0'>📏 <b>{r['distance_km']:.1f} km</b> approximate straight-line distance · ☎ {e(r['availability_status'])}</div>
 <blockquote style='margin:6px 0;padding:6px 10px;border-left:4px solid {color};background:#f6f8fa'>“{e(r['evidence_excerpt'])}”
  <br><span style='color:#57606a;font-size:.85em'>exact excerpt · field <code>{e(r['evidence_field'])}</code> · terms:
  {e(', '.join(r['matched_terms'][:6]))}</span></blockquote>
 <table style='border-collapse:collapse;font-size:.85em'><tr style='color:#57606a'>{heads}</tr><tr>{comps}</tr></table>
 <div style='font-size:.92em'>{e(r['explanation'])}</div>
 {"<ul style='margin:4px 0 0 0;font-size:.9em'>" + warn + "</ul>" if warn else ""}
 <div style='font-size:.85em;color:#57606a;margin-top:4px'>Source (record-level): {url_html} · facility_id <code>{e(r['facility_id'])}</code></div>
</div>""")
    if summary.get("held_back"):
        out.append("<details style='margin-top:8px'><summary><b>Held back (not recommended)</b>: nearest records that "
                   "mention the service but failed a credibility gate</summary><ul style='font-size:.9em'>")
        for h in summary["held_back"]:
            out.append(f"<li><b>{e(h['facility_name'])}</b> ({h['distance_km']:.1f} km, {e(EVIDENCE_LABELS[h['evidence_category']])}): "
                       f"{e('; '.join(h['failed_gates']))}<br><span style='color:#57606a'>{e(' | '.join(h['reasons']))}</span></li>")
        out.append("</ul></details>")
    out.append("</div>")
    return "".join(out)



# ---------------------------------------------------------------------------
# Short, scannable reasons for the app cards (full text lives in the audit expander)
# ---------------------------------------------------------------------------
SHORT_FINDING = {
    "COORDINATE_LOCATION_CONFLICT": "map position disagrees with its own PIN code",
    "MULTIPLE_INCOMPATIBLE_LOCATIONS": "address mentions conflicting places",
    "MULTIPLE_FACILITY_IDENTITIES": "record seems to describe another facility",
    "SERVICE_CONTRADICTION": "text says the service is not offered here / referred elsewhere",
    "SPECIALTY_PROFILE_MISMATCH": "facility type doesn't match this service",
    "UNRESOLVED_LOCATION": "location is approximate or unknown",
    "SHARED_PLACEHOLDER_COORDINATE": "shares map point with unrelated facilities",
    "AMBIGUOUS_PIN": "PIN code spans several districts/states",
    "MISSING_SOURCE_PROVENANCE": "no source link",
    "POSSIBLE_DUPLICATE": "possible duplicate listing",
    "WEAK_SERVICE_EVIDENCE": "service only weakly mentioned",
    "MISSING_EVIDENCE_EXCERPT": "no sentence actually states the service",
    "SUSPICIOUS_SOURCE_URL": "some source links look unrelated",
}
GATE_SHORT = {
    "location_confidence_below_minimum": "unreliable location",
    "identity_confidence_below_minimum": "unclear identity",
    "high_severity_finding": "serious contradiction",
    "probable_merged_or_misattributed_record": "probably a merged record",
    "no_supporting_excerpt": "no supporting sentence",
    "missing_provenance_fields": "missing provenance",
}


def gate_reason_short(failed_gates, findings=()) -> str:
    """One plain-language reason why a record was not recommended."""
    for f in findings:
        if f.get("severity") == "high" and f.get("code") in SHORT_FINDING:
            return SHORT_FINDING[f["code"]]
    for g in failed_gates:
        if g.startswith("evidence_conflicting"):
            return "contradictory evidence"
        if g.startswith("evidence_insufficient"):
            return "no sentence actually states the service"
        if g in GATE_SHORT:
            return GATE_SHORT[g]
    return "did not pass the credibility checks"


def short_reasons(r: Mapping) -> dict:
    """{'supported': str, 'caution': [str], 'unknown': str} - at most a few words each."""
    fields = [f for f in r.get("matched_fields", []) if f != "specialties"]
    sup = (f"Stated in its {r['evidence_field']}" + (f" + {len(fields) - 1} more field(s)" if len(fields) > 1 else "")
           + (" · specialty tag agrees" if r.get("specialty_tags") else ""))
    caution = [SHORT_FINDING[f["code"]] for f in r.get("findings", []) if f["code"] in SHORT_FINDING
               and f["severity"] in ("medium", "low")][:2]
    if r["evidence_category"] == "weak":
        caution.insert(0, "mentioned only once")
    return {"supported": sup, "caution": caution, "unknown": "Availability today unknown: call first"}
