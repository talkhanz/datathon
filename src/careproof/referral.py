"""Verification workflow and printable referral summary. Pure functions; no patient-identifiable data.

User-entered verification statuses are SESSION-ONLY feedback. They never update facility truth.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence

from .config import VERIFICATION_WARNING

STATUSES = ("Not contacted", "Confirmed", "Partially confirmed", "Unreachable", "Service contradicted",
            "Not accepting referrals")
BLOCKING_STATUSES = {"Unreachable", "Service contradicted", "Not accepting referrals"}

COMMON_CHECKS = [
    "Is the requested service operating today?",
    "Is admission or a walk-in possible now?",
    "Is the required specialist on duty or on call?",
    "Should the receiving facility be notified before departure? (name of person spoken to, time)",
    "Is prior authorisation or a referral letter required (e.g. PM-JAY / state scheme)?",
    "Is assisted transport (ambulance, 108) required?",
]
SERVICE_CHECKS = {
    "dialysis": ["Is a dialysis machine/slot free today, and at what time?", "Is a nephrologist available?",
                 "Are HBsAg/HCV/HIV-positive patients dialysed (separate machine)?", "Is vascular access (catheter) placement available if needed?"],
    "c_section": ["Is an obstetrician AND an anaesthetist available now for emergency surgery?",
                  "Is the operating theatre free?", "Is blood available or arrangeable (blood bank / storage)?",
                  "Is newborn resuscitation / a NICU available if the baby needs it?"],
    "nicu": ["Is a NICU bed (and ventilator, if needed) free now?", "Is a paediatrician/neonatologist on duty?",
             "Can the facility accept an outborn (transferred) baby?", "Is transport with an incubator or warmer available?"],
    "mri": ["Is the MRI machine working today?", "What is the waiting time for an appointment / emergency scan?",
            "Is sedation or anaesthesia support available if needed?", "Is a radiologist available to report?"],
    "cardiac_surgery": ["Is a cardiac surgeon available?", "Are the cardiac OT and ICU operational with free capacity?",
                        "Is a cath lab available for diagnosis first?", "Is blood available?"],
    "burn_care": ["Is a burn unit/ward with free beds operating?", "Is a surgeon available for burn care?",
                  "Can the facility manage the burn size/severity described?", "Is ICU capacity available if needed?"],
    "trauma_emergency": ["Is the emergency department open 24x7 right now?", "Are a surgeon and an orthopaedic doctor on duty?",
                         "Are CT/X-ray and an operating theatre available now?", "Is blood available? Is ICU capacity available?"],
}


def checklist(service_key: str) -> list[str]:
    return SERVICE_CHECKS.get(service_key, []) + COMMON_CHECKS


def next_alternative(results: Sequence[Mapping], statuses: Mapping[str, str], exclude: Optional[str] = None):
    """First ranked candidate not blocked by a session status (unreachable / contradicted / not accepting)."""
    for r in results:
        if r["facility_id"] != exclude and statuses.get(r["facility_id"], "Not contacted") not in BLOCKING_STATUSES:
            return r
    return None


def availability_label(status: str) -> str:
    if status == "Confirmed":
        return "Current availability confirmed by user (session only, unverified by CareProof)"
    if status == "Partially confirmed":
        return "Partially confirmed by user (session only)"
    if status in BLOCKING_STATUSES:
        return f"{status} (reported by user this session)"
    return "Current availability not verified"


def referral_summary(service_label: str, origin: str, selected: Mapping, alternatives: Sequence[Mapping],
                     status: str, checks: Mapping[str, bool], notes: str = "",
                     now: Optional[datetime] = None) -> str:
    """Plain-text, print-friendly referral summary. Contains no patient information by design."""
    now = now or datetime.now(timezone.utc)
    lines = [
        "CAREPROOF REFERRAL SUMMARY (planning aid - not a medical order)",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M UTC')} | pipeline {selected.get('pipeline_version', '')}",
        "",
        f"Requested service : {service_label}",
        f"Origin            : {origin}",
        "",
        "SELECTED FACILITY",
        f"  {selected['facility_name']}",
        f"  {selected.get('address', '')}",
        f"  Approx. straight-line distance: {selected['distance_km']:.1f} km (not road distance or travel time)",
        f"  Documented evidence: {selected['evidence_label']}",
        f"  Verification status: {availability_label(status)}",
        f"  Supporting excerpt ({selected['evidence_field']}): \"{selected['evidence_excerpt']}\"",
        f"  Source: {selected.get('source_url') or 'none listed'} | facility_id {selected['facility_id']}",
    ]
    cav = list(selected.get("identity_warnings", [])) + list(selected.get("location_warnings", [])) + \
        list(selected.get("missing_data_warnings", []))
    lines += ["", "CAVEATS"] + ([f"  - {c}" for c in cav] or ["  - No identity/location warnings recorded. Availability is still unverified."])
    lines += ["", "CALL-AHEAD CHECKLIST"] + [f"  [{'x' if checks.get(q) else ' '}] {q}" for q in checks]
    if notes.strip():
        lines += ["", "COORDINATOR NOTES (no patient-identifiable information)", "  " + notes.strip()[:500]]
    lines += ["", "ALTERNATIVES (in rank order)"]
    alts = [a for a in alternatives if a["facility_id"] != selected["facility_id"]][:4]
    lines += [f"  {i}. {a['facility_name']} - {a['distance_km']:.1f} km - {a['evidence_label']}" for i, a in enumerate(alts, 1)] \
        or ["  None eligible within the search radius."]
    lines += ["", VERIFICATION_WARNING]
    return "\n".join(lines)
