"""Plain-language explanations and warnings for each recommendation."""
from __future__ import annotations

from typing import Mapping

from .config import VERIFICATION_WARNING

CONTRADICTION_TEXT = {
    "negated_mention": "Some text for this facility says the service is NOT available, not provided, or only planned / under construction.",
    "specialty_mismatch": "The facility's name/type suggests a different specialty (e.g. eye, dental, diagnostic); "
                          "the service claim may come from a merged or mis-attributed record.",
    "location_conflict": "The facility's coordinates are far (> 25 km) from its own listed PIN code - "
                         "the record may merge two facilities or have a wrong address.",
    "state_conflict": "The listed state does not match the state of the listed PIN code.",
    "suspicious_source_url": "Source URLs include unrelated pages (e.g. login/payment pages), suggesting web-scrape noise.",
    "possible_duplicate": "Another record with the same name lies within 1 km; this may be a duplicate listing.",
}
MISSING_TEXT = {
    "coordinates_from_pin": "No usable facility coordinates; position estimated from its PIN code centre.",
    "pin_missing_or_invalid": "Facility PIN code is missing or invalid, so its location cannot be cross-checked.",
    "pin_not_in_directory": "Facility PIN code is not in the India Post directory.",
    "pin_centroid_unreliable": "The facility's PIN has no reliable reference location, so its coordinates are unverified.",
    "evidence_list_truncated": "Some evidence lists hit the 50-item source cap and may be incomplete.",
    "state_unrecognised": "The state field is missing or not a recognisable Indian state.",
}
CONTRADICTION_FLAGS = set(CONTRADICTION_TEXT)


def warnings_for(flags, missing_corroboration) -> tuple[list[str], list[str]]:
    contra = [CONTRADICTION_TEXT[f] for f in sorted(set(flags)) if f in CONTRADICTION_TEXT]
    missing = [MISSING_TEXT[f] for f in sorted(set(flags)) if f in MISSING_TEXT]
    missing += [f"Missing corroboration: {m}." for m in (missing_corroboration or [])]
    return missing, contra


def explain(r: Mapping, service_label: str) -> str:
    fields = [f for f in (r.get("matched_fields") or []) if f != "specialties"]
    field_txt = ", ".join(fields) if fields else "no text field"
    n = len(fields)
    parts = [
        f"{r['facility_name']} is ranked #{r['rank']} for {service_label} ({r['confidence']} confidence).",
        f"Why: the facility's own listing mentions this service in {n} field{'s' if n != 1 else ''} ({field_txt})",
    ]
    if r.get("specialty_tags"):
        parts[-1] += f", and it is tagged with related specialties ({', '.join(r['specialty_tags'][:3])})"
    parts[-1] += f'. Key evidence ({r["evidence_field"]}): "{_short(r["evidence_excerpt"])}".'
    parts.append(f"Where: about {r['distance_km']:.1f} km away in a straight line "
                 f"(position from {r['location_source'].replace('_', ' ')}).")
    how = (f"How: final score {r['final_score']:.2f} = weighted service match {r['service_match_score']:.2f}, "
           f"evidence {r['evidence_confidence_score']:.2f}, location quality {r['location_quality_score']:.2f}, "
           f"distance {r['distance_score']:.2f}, readiness {r['readiness_score']:.2f}")
    how += f", minus data-quality penalty {r['data_quality_penalty']:.2f}." if r["data_quality_penalty"] else "."
    parts.append(how)
    if r.get("contradiction_warnings"):
        parts.append("Caution: " + " ".join(r["contradiction_warnings"]))
    return " ".join(parts)


def _short(s: str, n: int = 160) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


__all__ = ["warnings_for", "explain", "VERIFICATION_WARNING", "CONTRADICTION_FLAGS"]


# ---------------------------------------------------------------------------
# HTML rendering for the notebook demo (no external assets)
# ---------------------------------------------------------------------------
_BADGE = {"High": "#1b7f3b", "Medium": "#9a6700", "Low": "#b42318"}


def render_cards_html(results, summary, title: str = "") -> str:
    from html import escape as e
    css = ("font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:980px;line-height:1.4;color:#1f2328")
    out = [f'<div style="{css}">']
    if title:
        out.append(f"<h2 style='margin:4px 0'>{e(title)}</h2>")
    out.append(
        f"<p style='margin:4px 0 10px'><b>{e(summary['service_label'])}</b> near <b>{e(summary['location'])}</b> "
        f"within {summary['max_distance_km']:.0f} km · {summary['explicit_text_candidates_in_radius']} facilities "
        f"with explicit text evidence · {summary['specialty_tag_only_excluded']} more only list a related specialty "
        f"tag and are <i>not</i> recommended · {summary['negated_only_excluded']} excluded for negated/planned mentions.</p>")
    status = summary.get("status")
    if status == "answered_low_confidence_only":
        out.append("<div style='background:#ffebe9;padding:6px 10px;border-radius:6px;margin:4px 0'><b>Low confidence "
                   "only.</b> Every facility found here has weak or contradictory evidence. Treat these as leads to "
                   "verify, not recommendations, or widen the search.</div>")
    elif status == "abstain_no_explicit_evidence_in_radius":
        out.append("<div style='background:#fff8c5;padding:6px 10px;border-radius:6px;margin:4px 0'><b>CareProof "
                   "abstains:</b> no facility in this radius states the service in its own text.</div>")
    for w in summary.get("location_warnings") or []:
        out.append(f"<div style='background:#fff8c5;padding:6px 10px;border-radius:6px;margin:4px 0'>📍 {e(w)}</div>")
    out.append(f"<div style='background:#ffebe9;padding:8px 10px;border-radius:6px;margin:8px 0'><b>⚠ Verify before "
               f"travel.</b> {e(VERIFICATION_WARNING)}</div>")
    if not results:
        out.append("<p><b>No facility with explicit text evidence was found in this radius.</b> Try a larger "
                   "max_distance_km. Absence here does not mean the service does not exist nearby.</p>")
    for r in results:
        color = _BADGE.get(r["confidence"], "#57606a")
        comps = "".join(
            f"<td style='padding:2px 8px;text-align:center'>{r[k]:.2f}</td>" for k in (
                "service_match_score", "evidence_confidence_score", "location_quality_score", "distance_score",
                "readiness_score", "data_quality_penalty", "final_score"))
        warn = "".join(f"<li style='color:#b42318'>{e(w)}</li>" for w in r["contradiction_warnings"])
        warn += "".join(f"<li style='color:#6e7781'>{e(w)}</li>" for w in r["missing_data_warnings"])
        url = r.get("source_url")
        url_html = (f"<a href='{e(url)}' target='_blank' rel='noopener'>{e(url[:90])}</a>" if url else "<i>none listed</i>")
        out.append(f"""
<div style='border:1px solid #d0d7de;border-radius:8px;padding:10px 14px;margin:10px 0'>
 <div style='display:flex;justify-content:space-between;align-items:baseline'>
  <div><span style='font-size:1.15em'><b>#{r['rank']} {e(r['facility_name'])}</b></span>
   <span style='color:#57606a'> · {e(r['address'])}</span></div>
  <span style='background:{color};color:white;border-radius:12px;padding:2px 10px;font-size:.9em;white-space:nowrap'>
   {e(r['confidence'])} confidence</span></div>
 <div style='margin:4px 0'>📏 <b>{r['distance_km']:.1f} km</b> approximate straight-line distance
   (not road distance or travel time) · position from {e(r['location_source'].replace('_', ' '))}</div>
 <blockquote style='margin:6px 0;padding:6px 10px;border-left:4px solid {color};background:#f6f8fa'>
   “{e(r['evidence_excerpt'])}”<br><span style='color:#57606a;font-size:.85em'>exact excerpt from field
   <code>{e(r['evidence_field'])}</code> · matched terms: {e(', '.join(r['matched_terms'][:6]))} ·
   fields: {e(', '.join(r['matched_fields']))}</span></blockquote>
 <table style='border-collapse:collapse;font-size:.85em;margin:4px 0'><tr style='color:#57606a'>
  <td style='padding:2px 8px'>service</td><td style='padding:2px 8px'>evidence</td><td style='padding:2px 8px'>location</td>
  <td style='padding:2px 8px'>distance</td><td style='padding:2px 8px'>readiness</td><td style='padding:2px 8px'>penalty</td>
  <td style='padding:2px 8px'><b>final</b></td></tr><tr>{comps}</tr></table>
 <div style='font-size:.92em'>{e(r['explanation'])}</div>
 {"<ul style='margin:4px 0 0 0;font-size:.9em'>" + warn + "</ul>" if warn else ""}
 <div style='font-size:.85em;color:#57606a;margin-top:4px'>Source (record-level; the dataset does not link each claim
   to a specific URL): {url_html} · facility_id <code>{e(r['facility_id'])}</code></div>
</div>""")
    out.append("</div>")
    return "".join(out)
