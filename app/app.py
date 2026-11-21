"""CareProof Referral Navigator: Databricks App (Streamlit).

Find -> Compare -> Verify -> Refer. All ranking/evidence logic is the tested `careproof` package (same code as the
notebooks). Data: project-owned tables via a SQL warehouse (Databricks), or local Parquet when
CAREPROOF_LOCAL_DATA is set (development). No patient-identifiable information is collected or stored.
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
from careproof.config import EVIDENCE_LABELS, TABLES, VERIFICATION_WARNING, fq_table, validate_identifier  # noqa: E402
from careproof.explanations import COMPONENT_NAMES  # noqa: E402
from careproof.geography import LocationError, resolve_location  # noqa: E402
from careproof.referral import (BLOCKING_STATUSES, STATUSES, availability_label, checklist, next_alternative,  # noqa: E402
                                referral_summary)
from careproof.service_matching import TAXONOMY  # noqa: E402

st.set_page_config(page_title="CareProof Referral Navigator", page_icon="🩺", layout="wide")

BADGE = {"strong": "🟢", "moderate": "🔵", "weak": "🟡", "conflicting": "🔴", "insufficient": "⚪"}
_MD = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<])")


def md(text) -> str:
    """Escape untrusted source text before rendering it as Markdown."""
    return _MD.sub(r"\\\1", str(text or ""))


# ------------------------------------------------------------------------------------------------ data loading
@st.cache_resource(show_spinner="Loading facility evidence tables…")
def load_data():
    local = os.environ.get("CAREPROOF_LOCAL_DATA")
    if local:
        gold = pd.read_parquet(os.path.join(local, "careproof_facility_service_gold.parquet"))
        pin = pd.read_parquet(os.path.join(local, "careproof_pincode_silver.parquet"))
        source = f"local Parquet ({os.path.basename(local.rstrip('/'))})"
    else:
        from databricks import sql
        from databricks.sdk.core import Config
        cfg = Config()
        cat = validate_identifier(os.environ.get("CAREPROOF_CATALOG", config.DEFAULT_TARGET_CATALOG), "catalog")
        sch = validate_identifier(os.environ.get("CAREPROOF_SCHEMA", config.DEFAULT_TARGET_SCHEMA), "schema")
        wh = os.environ["DATABRICKS_WAREHOUSE_ID"]
        with sql.connect(server_hostname=cfg.host.replace("https://", "").rstrip("/"),
                         http_path=f"/sql/1.0/warehouses/{wh}",
                         credentials_provider=lambda: cfg.authenticate) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {fq_table(cat, sch, TABLES['facility_service_gold'])}")
            gold = cur.fetchall_arrow().to_pandas()
            cur.execute(f"SELECT * FROM {fq_table(cat, sch, TABLES['pincode_silver'])}")
            pin = cur.fetchall_arrow().to_pandas()
        source = f"{cat}.{sch} (SQL warehouse)"
    gold, pin = P.coerce_gold_types(gold), P.coerce_pin_types(pin)
    return gold, P.place_index_from_pin_silver(pin), source


def gold_lookup(gold, facility_id, service):
    m = gold[(gold.facility_id == facility_id) & (gold.service == service)]
    return None if m.empty else m.iloc[0]


# ------------------------------------------------------------------------------------------------ session state
ss = st.session_state
for k, v in {"results": None, "summary": None, "baseline": None, "query": None, "selected": None,
             "statuses": {}, "checks": {}, "notes": ""}.items():
    ss.setdefault(k, v)

# ------------------------------------------------------------------------------------------------ header
st.title("🩺 CareProof Referral Navigator")
st.caption("**The nearest claimed provider is not necessarily the nearest credibly supported provider.** "
           "CareProof finds facilities whose own records document a service, shows the exact evidence and its "
           "weaknesses, and helps you verify before referral.")
st.warning(VERIFICATION_WARNING, icon="⚠️")

try:
    gold, index, data_source = load_data()
except Exception as e:  # pragma: no cover - surfaced to the user
    st.error(f"Could not load CareProof tables: {type(e).__name__}: {e}. Check that notebooks 02-03 have run and "
             "that the app has SELECT access to the target schema and CAN USE on the SQL warehouse.")
    st.stop()

# ------------------------------------------------------------------------------------------------ sidebar (inputs)
with st.sidebar:
    st.header("Search")
    svc_keys = list(TAXONOMY)
    svc = st.selectbox("Required service", svc_keys, format_func=lambda k: TAXONOMY[k].label, key="svc")
    ltype = st.radio("Location type", ["pincode", "district", "city", "coordinates"], horizontal=True, key="ltype",
                     format_func=lambda x: {"pincode": "PIN code", "district": "District", "city": "City",
                                            "coordinates": "Coordinates"}[x])
    lval, lat, lon = "", "", ""
    if ltype == "coordinates":
        c1, c2 = st.columns(2)
        lat = c1.text_input("Latitude", "25.6093", key="lat")
        lon = c2.text_input("Longitude", "85.1235", key="lon")
    else:
        hint = {"pincode": "800001", "district": "Patna, Bihar", "city": "Patna"}[ltype]
        lval = st.text_input({"pincode": "6-digit PIN code", "district": "District (optionally ', State')",
                              "city": "City (matched to a district)"}[ltype], hint, key="lval")
    radius = st.slider("Search radius (km, straight-line)", 5, 300, 50, step=5, key="radius")
    top_k = st.slider("Number of results", 1, 10, 5, key="topk")
    go = st.button("Find facilities", type="primary", use_container_width=True, key="go")
    st.caption(f"Data: {data_source} · pipeline "
               f"{gold['pipeline_version'].iloc[0] if 'pipeline_version' in gold and len(gold) else ''}")

if go:
    try:
        loc = resolve_location(ltype, lval, lat, lon, index)
        res, summ = P.recommend(gold, loc, svc, float(radius), int(top_k))
        base = P.keyword_baseline(gold, loc, svc, float(radius), 5)
        ss.update(results=res, summary=summ, baseline=base, selected=(res[0]["facility_id"] if res else None),
                  query={"service": svc, "label": TAXONOMY[svc].label, "origin": loc.label, "radius": radius},
                  statuses={}, checks={}, notes="", loc_error=None)
    except LocationError as e:
        ss.update(results=None, summary=None, baseline=None, selected=None, loc_error=str(e))

if ss.get("loc_error"):
    st.error(f"**Location could not be resolved reliably.** {ss['loc_error']}")

if ss.results is None:
    st.info("Choose a service and a location in the sidebar, then click **Find facilities**.")
    st.stop()

results, summary, q = ss.results, ss.summary, ss.query
find_t, compare_t, verify_t, refer_t = st.tabs(["1 · Find", "2 · Compare", "3 · Verify", "4 · Refer"])

# ------------------------------------------------------------------------------------------------ FIND
with find_t:
    c1, c2, c3 = st.columns(3)
    c1.metric("Records mentioning the service", summary["retrieved_mentions_in_radius"])
    c2.metric("Passed all credibility gates", summary["eligible_in_radius"])
    c3.metric("Held back (not recommended)", summary["held_back_in_radius"])
    for w in summary.get("location_warnings") or []:
        st.warning(w, icon="📍")
    if not results:
        st.error(f"**CareProof abstains.** {summary['message']}", icon="🛑")
    for r in results:
        status = ss.statuses.get(r["facility_id"], "Not contacted")
        with st.container(border=True):
            h1, h2 = st.columns([3, 1])
            h1.markdown(f"### #{r['rank']} {md(r['facility_name'])}")
            h1.caption(md(r["address"]))
            h2.markdown(f"{BADGE[r['evidence_category']]} **{r['evidence_label']}**")
            h2.markdown(f"📏 **{r['distance_km']:.1f} km** approx. straight-line")
            h2.markdown(f"☎ {availability_label(status)}")
            if status in BLOCKING_STATUSES:
                st.error(f"Reported this session: **{status}**. See the next alternative in the Verify tab.")
            st.markdown(f"> “{md(r['evidence_excerpt'])}”")
            st.caption(f"Exact excerpt · field `{r['evidence_field']}` · matched terms: {md(', '.join(r['matched_terms'][:6]))}"
                       f" · fields: {md(', '.join(r['matched_fields']))}")
            st.write(r["explanation"])
            for w in r["identity_warnings"]:
                st.markdown(f"🆔 **Identity:** {md(w)}")
            for w in r["location_warnings"]:
                st.markdown(f"🗺️ **Location:** {md(w)}")
            for w in r["missing_data_warnings"]:
                st.markdown(f"❔ **Missing information:** {md(w)}")
            if r.get("source_url"):
                st.markdown(f"Source (record-level): [{md(r['source_url'][:80])}]({r['source_url']})")
            with st.expander("Component scores (policy weights; not probabilities)"):
                st.dataframe(pd.DataFrame([{"component": COMPONENT_NAMES.get(k, k), "score": r[k]} for k in
                                           list(COMPONENT_NAMES) + ["data_quality_penalty", "final_ranking_score"]]),
                             hide_index=True, use_container_width=True)
                st.caption(f"facility_id {r['facility_id']} · source_row_id {r['source_row_id'][:12]}… · {r['pipeline_version']}")
            if st.button("Select for verification", key=f"sel_{r['facility_id']}"):
                ss.selected = r["facility_id"]
                st.success("Selected. Open the **Verify** tab.")
    if summary.get("held_back"):
        with st.expander(f"Planner / audit view: {len(summary['held_back'])} nearest records held back"):
            st.caption("These records mention the service but failed at least one credibility gate. They are shown for "
                       "transparency and data review, **not** as referral options.")
            st.dataframe(pd.DataFrame([{"facility": h["facility_name"], "km": h["distance_km"],
                                        "evidence": EVIDENCE_LABELS[h["evidence_category"]],
                                        "failed gates": "; ".join(h["failed_gates"]),
                                        "why": " | ".join(h["reasons"]), "excerpt": h["evidence_excerpt"]}
                                       for h in summary["held_back"]]), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------------------------ COMPARE
with compare_t:
    st.subheader("Nearest keyword match vs CareProof")
    base = ss.baseline or []
    left, right = st.columns(2)
    with left:
        st.markdown("**Nearest keyword match** (distance only; no evidence checks)")
        if base:
            b = base[0]
            g = gold_lookup(gold, b["facility_id"], q["service"])
            st.markdown(f"### {md(b['facility_name'])}\n📏 {b['distance_km']:.1f} km")
            if g is not None:
                st.markdown(f"Evidence: {BADGE[g['evidence_category']]} {EVIDENCE_LABELS[g['evidence_category']]}")
                failed = json.loads(g["failed_gates_json"])
                (st.error if failed else st.success)("Fails: " + "; ".join(failed) if failed else "Passes all gates")
                if isinstance(g["evidence_excerpt"], str):
                    st.markdown(f"> “{md(g['evidence_excerpt'])}”")
        else:
            st.info("No keyword match in this radius.")
    with right:
        st.markdown("**CareProof top eligible candidate**")
        if results:
            t = results[0]
            nearest = min(results, key=lambda r: r["distance_km"])
            st.markdown(f"### {md(t['facility_name'])}\n📏 {t['distance_km']:.1f} km")
            st.markdown(f"Evidence: {BADGE[t['evidence_category']]} {t['evidence_label']}")
            st.markdown(f"> “{md(t['evidence_excerpt'])}”")
            if nearest["facility_id"] != t["facility_id"]:
                st.caption(f"Nearest eligible candidate: {md(nearest['facility_name'])} ({nearest['distance_km']:.1f} km), "
                           f"ranked #{nearest['rank']} because its documented evidence or consistency is weaker.")
        else:
            st.error("No eligible candidate: CareProof abstains.")
    st.markdown("#### Why they differ")
    if base and results and base[0]["facility_id"] == results[0]["facility_id"]:
        st.success("Both methods agree on the top facility, and it passes every credibility gate.")
    elif base:
        g = gold_lookup(gold, base[0]["facility_id"], q["service"])
        reasons = json.loads(g["failed_gates_json"]) if g is not None else ["not in evidence table"]
        st.write("The nearest keyword match is ranked by distance alone. "
                 + (f"It is held back because: {', '.join(reasons)}." if reasons else
                    "It passes the gates but ranks lower on evidence strength, identity or location consistency."))
    st.dataframe(pd.DataFrame([{"keyword rank": b["rank"], "facility": b["facility_name"], "km": b["distance_km"],
                                "CareProof": next((f"#{r['rank']} · {r['evidence_label']}" for r in results
                                                   if r["facility_id"] == b["facility_id"]), "not recommended")}
                               for b in base]), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------------------------ VERIFY
with verify_t:
    if not results:
        st.error("Nothing to verify: no eligible candidate. Widen the radius or contact the district health office.")
    else:
        ids = [r["facility_id"] for r in results]
        cur_idx = ids.index(ss.selected) if ss.selected in ids else 0
        pick = st.selectbox("Facility being called", ids, index=cur_idx, key="verify_pick",
                            format_func=lambda i: next(f"#{r['rank']} {r['facility_name']}" for r in results if r["facility_id"] == i))
        ss.selected = pick
        sel = next(r for r in results if r["facility_id"] == pick)
        st.markdown(f"**{md(sel['facility_name'])}** · {sel['distance_km']:.1f} km · {sel['evidence_label']}")
        st.caption("Statuses are kept for this session only. They do not change CareProof's facility records.")
        status = st.selectbox("Call outcome", STATUSES, key=f"status_{pick}",
                              index=STATUSES.index(ss.statuses.get(pick, "Not contacted")))
        ss.statuses[pick] = status
        st.markdown("**Call-ahead checklist**")
        checks = ss.checks.setdefault(pick, {})
        for i, item in enumerate(checklist(q["service"])):
            checks[item] = st.checkbox(item, value=checks.get(item, False), key=f"chk_{pick}_{i}")
        ss.notes = st.text_area("Coordinator notes (do NOT enter patient names, phone numbers or other identifiers)",
                                ss.notes, key="notes_box", max_chars=500)
        if status in BLOCKING_STATUSES:
            alt = next_alternative(results, ss.statuses, exclude=pick)
            if alt:
                st.error(f"**{status}.** Next eligible alternative: **#{alt['rank']} {md(alt['facility_name'])}** "
                         f"({alt['distance_km']:.1f} km, {alt['evidence_label']}).", icon="➡️")
                if st.button(f"Switch to {alt['facility_name']}", key="switch_alt"):
                    ss.selected = alt["facility_id"]
                    st.rerun()
            else:
                st.error("No remaining eligible alternative in this radius. Widen the search or escalate to the "
                         "district referral desk / 108.")
        elif status == "Confirmed":
            st.success("Confirmed by you this session. Continue to the **Refer** tab for a printable summary.")

# ------------------------------------------------------------------------------------------------ REFER
with refer_t:
    if not results or ss.selected is None:
        st.info("Select and verify a facility first.")
    else:
        sel = next(r for r in results if r["facility_id"] == ss.selected)
        text = referral_summary(q["label"], q["origin"], sel, results, ss.statuses.get(sel["facility_id"], "Not contacted"),
                                ss.checks.get(sel["facility_id"], {i: False for i in checklist(q["service"])}),
                                ss.notes, datetime.now(timezone.utc))
        st.code(text, language=None)
        st.download_button("Download referral summary (.txt, print-friendly)", text,
                           file_name=f"careproof_referral_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.txt",
                           mime="text/plain", key="dl")
        st.caption("The summary contains no patient information. Attach it to your own referral form if needed.")
