"""CareProof Referral Navigator: Databricks App (Streamlit). Search -> scan results -> call & refer.

All ranking/evidence logic is the tested `careproof` package (identical to the notebooks), served through the
precomputed `RecommendationIndex`. No patient-identifiable information is collected or stored.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from careproof import config, pipeline as P  # noqa: E402
from careproof.config import TABLES, VERIFICATION_WARNING, fq_table, validate_identifier  # noqa: E402
from careproof.explanations import COMPONENT_NAMES, gate_reason_short, short_reasons  # noqa: E402
from careproof.geography import LocationError, resolve_free_text  # noqa: E402
from careproof.referral import (BLOCKING_STATUSES, STATUSES, checklist, next_alternative,  # noqa: E402
                                referral_summary)
from careproof.service_matching import TAXONOMY  # noqa: E402

st.set_page_config(page_title="CareProof Referral Navigator", page_icon="🩺", layout="wide")
st.markdown("<style>div.block-container{padding-top:1.4rem} blockquote{margin:.3rem 0}</style>", unsafe_allow_html=True)

BADGE = {"strong": "🟢 Strong evidence", "moderate": "🔵 Moderate evidence", "weak": "🟡 Weak evidence"}
SERVICE_SHORT = {"dialysis": "Dialysis", "c_section": "C-section / emergency obstetric surgery", "nicu": "NICU (newborn ICU)",
                 "mri": "MRI scan", "cardiac_surgery": "Cardiac surgery", "burn_care": "Burn care",
                 "trauma_emergency": "Trauma & emergency"}
_MD = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<])")


def md(text) -> str:
    return _MD.sub(r"\\\1", str(text or ""))


def clip(text, n=150) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


# ------------------------------------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Loading facility evidence…")
def load_data():
    local = os.environ.get("CAREPROOF_LOCAL_DATA")
    if local:
        gold = pd.read_parquet(os.path.join(local, "careproof_facility_service_gold.parquet"))
        pin = pd.read_parquet(os.path.join(local, "careproof_pincode_silver.parquet"))
    else:
        from databricks import sql
        from databricks.sdk.core import Config
        cfg = Config()
        cat = validate_identifier(os.environ.get("CAREPROOF_CATALOG", config.DEFAULT_TARGET_CATALOG), "catalog")
        sch = validate_identifier(os.environ.get("CAREPROOF_SCHEMA", config.DEFAULT_TARGET_SCHEMA), "schema")
        with sql.connect(server_hostname=cfg.host.replace("https://", "").rstrip("/"),
                         http_path=f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}",
                         credentials_provider=lambda: cfg.authenticate) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {fq_table(cat, sch, TABLES['facility_service_gold'])}")
            gold = cur.fetchall_arrow().to_pandas()
            cur.execute(f"SELECT * FROM {fq_table(cat, sch, TABLES['pincode_silver'])}")
            pin = cur.fetchall_arrow().to_pandas()
    gold, pin = P.coerce_gold_types(gold), P.coerce_pin_types(pin)
    return gold, P.RecommendationIndex(gold), P.place_index_from_pin_silver(pin)


def gold_row(gold, fid, svc):
    m = gold[(gold.facility_id == fid) & (gold.service == svc)]
    return None if m.empty else m.iloc[0]


ss = st.session_state
for k, v in {"res": None, "summ": None, "base": None, "q": None, "sel": None, "status": {}, "checks": {},
             "notes": "", "err": None, "k": 5}.items():
    ss.setdefault(k, v)

try:
    gold, index, places = load_data()
except Exception as e:  # pragma: no cover
    st.error(f"Could not load CareProof data ({type(e).__name__}: {e}). Run notebooks 02–03 and check app permissions.")
    st.stop()


def run_search(svc, loc_text, radius, k):
    try:
        loc = resolve_free_text(loc_text, places)
    except LocationError as e:
        ss.update(res=None, err=str(e))
        return
    res, summ = index.recommend(loc, svc, float(radius), int(k))
    ss.update(res=res, summ=summ, base=P.keyword_baseline(gold, loc, svc, float(radius), 1), err=None,
              q={"svc": svc, "label": TAXONOMY[svc].label, "origin": loc.label, "radius": radius, "text": loc_text,
                 "lat": loc.lat, "lon": loc.lon},
              sel=None, status={}, checks={}, notes="", k=k)


# ------------------------------------------------------------------------------------------------ header + search
st.markdown("## 🩺 CareProof Referral Navigator")
st.caption("Finds facilities whose own records **credibly** document a service, not just the nearest name that mentions it.")

with st.form("search", border=False):
    c1, c2, c3, c4 = st.columns([2.2, 3, 1.2, 1])
    svc = c1.selectbox("Service needed", list(TAXONOMY), format_func=SERVICE_SHORT.get, key="svc")
    loc_text = c2.text_input("Where is the patient?", "800001", key="loc_text",
                             placeholder="PIN (800001) · district (Patna, Bihar) · lat, lon")
    radius = c3.selectbox("Within", [10, 25, 50, 100, 200], index=2, format_func=lambda x: f"{x} km", key="radius")
    c4.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    go = c4.form_submit_button("Search", type="primary", use_container_width=True, key="go")
if go:
    run_search(svc, loc_text, radius, 5)

w1, w2 = st.columns([6, 1])
w1.caption("⚠️ Planning aid only. Call the facility to confirm the service and admission before travel. Emergency: **108 / 112**.")
with w2.popover("ℹ️ About"):
    st.write(VERIFICATION_WARNING)
    st.caption("Evidence levels describe how well the facility's public record documents the service, not whether it "
               "is available today. Distances are straight-line, not travel time.")

if ss.err:
    st.error(ss.err, icon="📍")
if ss.res is None:
    st.info("Pick a service, type a PIN code, district or coordinates, and press **Search**.")
    st.stop()

res, summ, q = ss.res, ss.summ, ss.q
for w in summ.get("location_warnings") or []:
    st.caption(f"📍 {w}")

# ------------------------------------------------------------------------------------------------ results
left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown(f"**{SERVICE_SHORT[q['svc']]}** near **{md(q['origin'])}** · {summ['eligible_in_radius']} of "
                f"{summ['retrieved_mentions_in_radius']} listings that mention it passed the credibility checks")
    if ss.base:  # why the nearest keyword match is / isn't first
        b = ss.base[0]
        shown = next((r for r in res if r["facility_id"] == b["facility_id"]), None)
        if shown is None:
            g = gold_row(gold, b["facility_id"], q["svc"])
            why = gate_reason_short(json.loads(g["failed_gates_json"]),
                                    json.loads(g["facility_findings_json"]) + json.loads(g["claim_findings_json"])) \
                if g is not None else "not in the evidence table"
            st.info(f"The nearest listing that mentions this service, **{md(b['facility_name'])}** "
                    f"({b['distance_km']:.1f} km), is not shown: {why}.", icon="🔎")
        elif shown["rank"] > 1:
            st.info(f"The nearest listing, **{md(b['facility_name'])}** ({b['distance_km']:.1f} km), ranks "
                    f"#{shown['rank']}: the options above have stronger or more consistent evidence.", icon="🔎")
    if not res:
        st.error(summ["message"], icon="🛑")
        if q["radius"] < 300 and st.button(f"Search within {min(q['radius'] * 2, 300)} km", key="widen"):
            run_search(q["svc"], q["text"], min(q["radius"] * 2, 300), ss.k)
            st.rerun()
    for r in res:
        stt = ss.status.get(r["facility_id"], "Not contacted")
        status_txt = ("✅ " + stt if stt in ("Confirmed", "Partially confirmed")
                      else "⛔ " + stt if stt in BLOCKING_STATUSES else "☎ Not verified")
        with st.container(border=True):
            a, b = st.columns([4, 1.4])
            a.markdown(f"**{r['rank']}. {md(r['facility_name'])}**  \n"
                       f"{r['distance_km']:.1f} km · {md(clip(r['address'], 60))}")
            b.markdown(f"{BADGE.get(r['evidence_category'], '')}  \n{status_txt}")
            st.markdown(f"> “{md(clip(r['evidence_excerpt']))}”")
            x, y, _ = st.columns([1.3, 1, 2])
            if x.button("📞 Call & refer", key=f"pick_{r['facility_id']}", use_container_width=True):
                ss.sel = r["facility_id"]
            with y.popover("Why?", use_container_width=True):
                sr = short_reasons(r)
                st.markdown(f"✅ {md(sr['supported'])}")
                for c in sr["caution"]:
                    st.markdown(f"⚠️ {md(c)}")
                st.markdown(f"❔ {sr['unknown']}")
                st.caption(f"Full quote ({r['evidence_field']}): “{md(r['evidence_excerpt'])}”")
                if r.get("source_url"):
                    st.markdown(f"[Source]({r['source_url']})")
    if res and summ["additional_eligible_beyond_top_k"] and ss.k < 10:
        if st.button("Show more options", key="more"):
            run_search(q["svc"], q["text"], q["radius"], 10)
            st.rerun()

with right:
    colors = {"strong": "#1a7f37", "moderate": "#0969da", "weak": "#bf8700"}
    pts = [{"lat": q["lat"], "lon": q["lon"], "color": "#d1242f", "size": 220}] + \
          [{"lat": r["location_lat"], "lon": r["location_lon"], "color": colors[r["evidence_category"]], "size": 140}
           for r in res]
    st.map(pd.DataFrame(pts), latitude="lat", longitude="lon", color="color", size="size", height=330)
    st.caption("🔴 patient location · 🟢 strong · 🔵 moderate · 🟡 weak evidence")

    # -------------------------------------------------------------------------------------------- call & refer
    sel = next((r for r in res if r["facility_id"] == ss.sel), None)
    if sel is None and res:
        st.markdown("##### 📞 Call & refer")
        st.caption("Press **Call & refer** on a facility to get a call checklist and a printable referral summary.")
    elif sel is not None:
        with st.container(border=True):
            st.markdown(f"##### 📞 {md(sel['facility_name'])}")
            st.markdown(f"**Phone:** {md(sel['phone'])}" if sel.get("phone") else "**Phone:** not listed: use the source link")
            items = checklist(q["svc"])
            chk = ss.checks.setdefault(sel["facility_id"], {})
            st.caption("Ask before sending the patient:")
            for i, item in enumerate(items[:4]):
                chk[item] = st.checkbox(item, value=chk.get(item, False), key=f"c_{sel['facility_id']}_{i}")
            with st.expander("More checks"):
                for i, item in enumerate(items[4:], start=4):
                    chk[item] = st.checkbox(item, value=chk.get(item, False), key=f"c_{sel['facility_id']}_{i}")
            outcome = st.radio("Call outcome", STATUSES, horizontal=True, key=f"o_{sel['facility_id']}",
                               index=STATUSES.index(ss.status.get(sel["facility_id"], "Not contacted")))
            ss.status[sel["facility_id"]] = outcome
            if outcome in BLOCKING_STATUSES:
                alt = next_alternative(res, ss.status, exclude=sel["facility_id"])
                if alt:
                    st.warning(f"Next option: **{md(alt['facility_name'])}** ({alt['distance_km']:.1f} km)", icon="➡️")
                    if st.button("Switch to next option", key="switch"):
                        ss.sel = alt["facility_id"]
                        st.rerun()
                else:
                    st.error("No other credible option in this radius. Widen the search or call 108 / the district referral desk.")
            ss.notes = st.text_input("Notes (no patient names or numbers)", ss.notes, key="notes_box", max_chars=300)
            text = referral_summary(q["label"], q["origin"], sel, res, outcome, chk, ss.notes, datetime.now(timezone.utc))
            st.download_button("⬇️ Download referral summary", text, use_container_width=True, key="dl",
                               file_name=f"careproof_referral_{datetime.now(timezone.utc):%Y%m%d_%H%M}.txt")
            with st.expander("Preview summary"):
                st.code(text, language=None)

# ------------------------------------------------------------------------------------------------ audit
with st.expander("🔍 Audit details: scores, full explanations, records held back"):
    st.caption("Scores rank only facilities that passed every credibility gate. Weights are documented policy "
               "choices (Bayesian-optimised under safety constraints), not probabilities.")
    if res:
        st.dataframe(pd.DataFrame([{"#": r["rank"], "facility": r["facility_name"], "evidence": r["evidence_label"],
                                    **{COMPONENT_NAMES[k]: r[k] for k in COMPONENT_NAMES},
                                    "penalty": r["data_quality_penalty"], "rank score": r["final_ranking_score"]}
                                   for r in res]), hide_index=True, use_container_width=True)
        for r in res:
            st.markdown(f"**{r['rank']}. {md(r['facility_name'])}**: {md(r['explanation'])}")
    if summ.get("held_back"):
        st.markdown("**Held back (mention the service but failed a check):**")
        st.dataframe(pd.DataFrame([{"facility": h["facility_name"], "km": h["distance_km"],
                                    "why not": gate_reason_short(h["failed_gates"]),
                                    "detail": " | ".join(h["reasons"])} for h in summ["held_back"]]),
                     hide_index=True, use_container_width=True)
