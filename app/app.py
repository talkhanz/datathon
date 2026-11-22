"""CareProof Referral Navigator: Databricks App (Streamlit). Search -> scan results -> call & refer.

All ranking/evidence logic is the tested `careproof` package (identical to the notebooks), served through the
precomputed `RecommendationIndex`. The service-profile model and GA results are read from precomputed tables
(notebook 07); the app never trains anything and the model never changes who is recommended.
No patient-identifiable information is collected or stored.
"""
from __future__ import annotations

import html
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from careproof import config, pipeline as P  # noqa: E402
from careproof.config import DEFAULT_WEIGHTS, TABLES, VERIFICATION_WARNING, fq_table, validate_identifier  # noqa: E402
from careproof.explanations import COMPONENT_NAMES, gate_reason_short, short_reasons  # noqa: E402
from careproof.geography import LocationError, resolve_free_text  # noqa: E402
from careproof import coverage as CV  # noqa: E402
from careproof.modeling import LOG_FEATURES, unverified_leads  # noqa: E402
from careproof.referral import (BLOCKING_STATUSES, STATUSES, checklist, next_alternative,  # noqa: E402
                                referral_summary)
from careproof.service_matching import TAXONOMY  # noqa: E402

st.set_page_config(page_title="CareProof Referral Navigator", page_icon="🩺", layout="wide")
st.markdown("""<style>
div.block-container{padding-top:1.2rem} blockquote{margin:.3rem 0}
.cp-bar{background:#eaeef2;border-radius:6px;height:8px;width:100%;overflow:hidden}
.cp-fill{height:8px;border-radius:6px}
.cp-lab{font-size:.78rem;color:#57606a;margin:2px 0 1px 0}
.cp-pill{display:inline-block;padding:1px 9px;border-radius:12px;font-size:.78rem;font-weight:600;color:#fff}
</style>""", unsafe_allow_html=True)

EV_COLOR = {"strong": "#1a7f37", "moderate": "#0969da", "weak": "#bf8700"}
EV_LABEL = {"strong": "Strong evidence", "moderate": "Moderate evidence", "weak": "Weak evidence"}
EV_LEVEL = {"strong": 1.0, "moderate": 0.66, "weak": 0.33}
FIT_COLOR = {"high": "#1a7f37", "medium": "#bf8700", "low": "#8c959f"}
FIT_TEXT = {"high": "typical of documented providers", "medium": "partly typical", "low": "atypical for this service"}
SERVICE_SHORT = {"dialysis": "Dialysis", "c_section": "C-section / emergency obstetric surgery", "nicu": "NICU (newborn ICU)",
                 "mri": "MRI scan", "cardiac_surgery": "Cardiac surgery", "burn_care": "Burn care",
                 "trauma_emergency": "Trauma & emergency"}
SHORT_COMP = {"service_relevance_score": "Service named", "evidence_strength_score": "Evidence",
              "identity_confidence_score": "Identity", "location_confidence_score": "Location",
              "readiness_evidence_score": "Readiness", "distance_score": "Proximity"}
WKEY = {"service_relevance_score": "service_relevance", "evidence_strength_score": "evidence_strength",
        "identity_confidence_score": "identity_confidence", "location_confidence_score": "location_confidence",
        "readiness_evidence_score": "readiness_evidence", "distance_score": "distance"}
COMP_COLORS = ["#8250df", "#1a7f37", "#0969da", "#1b7c83", "#bf8700", "#cf222e"]
W = DEFAULT_WEIGHTS.normalized().as_dict()
_MD = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<])")
_TEXT_COLS = {"facility_id", "service", "feature", "feature_label", "model", "feature_set", "split", "task", "score_band",
              "kind", "group", "drop_reason", "name", "phone", "address_city", "nearest_real_facility_id",
              "nearest_real_facility_name", "source"}


def md(text) -> str:
    return _MD.sub(r"\\\1", str(text or ""))


def esc(text) -> str:
    return html.escape(str(text or ""))


def clip(text, n=150) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def bar(label, value, color, right="") -> str:
    v = max(0.0, min(1.0, float(value or 0)))
    return (f"<div class='cp-lab'>{esc(label)}<span style='float:right'>{esc(right)}</span></div>"
            f"<div class='cp-bar'><div class='cp-fill' style='width:{v*100:.0f}%;background:{color}'></div></div>")


def pill(text, color) -> str:
    return f"<span class='cp-pill' style='background:{color}'>{esc(text)}</span>"


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower() == "true"


# ------------------------------------------------------------------------------------------------ data
def _read_tables(names):
    local = os.environ.get("CAREPROOF_LOCAL_DATA")
    out = {}
    if local:
        for n in names:
            path = os.path.join(local, f"{TABLES[n]}.parquet")
            out[n] = pd.read_parquet(path) if os.path.exists(path) else None
        return out
    from databricks import sql
    from databricks.sdk.core import Config
    cfg = Config()
    cat = validate_identifier(os.environ.get("CAREPROOF_CATALOG", config.DEFAULT_TARGET_CATALOG), "catalog")
    sch = validate_identifier(os.environ.get("CAREPROOF_SCHEMA", config.DEFAULT_TARGET_SCHEMA), "schema")
    with sql.connect(server_hostname=cfg.host.replace("https://", "").rstrip("/"),
                     http_path=f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}",
                     credentials_provider=lambda: cfg.authenticate) as conn, conn.cursor() as cur:
        for n in names:
            try:
                cur.execute(f"SELECT * FROM {fq_table(cat, sch, TABLES[n])}")
                out[n] = cur.fetchall_arrow().to_pandas()
            except Exception:  # optional model tables may not exist yet
                out[n] = None
    return out


@st.cache_resource(show_spinner="Loading facility evidence…")
def load_data():
    t = _read_tables(["facility_service_gold", "pincode_silver"])
    gold, pin = P.coerce_gold_types(t["facility_service_gold"]), P.coerce_pin_types(t["pincode_silver"])
    return gold, P.RecommendationIndex(gold), P.place_index_from_pin_silver(pin), pin


@st.cache_resource(show_spinner="Loading model results…")
def load_model():
    t = _read_tables(list(config.MODEL_TABLES))
    if t.get("model_scores") is None or t.get("model_profile_summary") is None:
        return None
    for df in t.values():
        if df is None:
            continue
        for c in df.columns:
            if c in _TEXT_COLS or df[c].dtype != object:
                continue
            low = df[c].astype(str).str.lower()
            df[c] = low.eq("true") if low.isin(["true", "false"]).all() else pd.to_numeric(df[c], errors="coerce")
    t["score_lookup"] = t["model_scores"].set_index(["facility_id", "service"])[["model_score", "score_band"]]
    if t.get("model_features") is not None:
        t["model_features"] = t["model_features"].set_index("facility_id")
    if t.get("model_facilities") is not None:
        t["model_facilities"]["has_high_finding"] = truthy(t["model_facilities"]["has_high_finding"])
    return t


def gold_row(gold, fid, svc):
    m = gold[(gold.facility_id == fid) & (gold.service == svc)]
    return None if m.empty else m.iloc[0]


ss = st.session_state
for k, v in {"res": None, "summ": None, "base": None, "q": None, "sel": None, "status": {}, "checks": {},
             "notes": "", "err": None, "k": 5}.items():
    ss.setdefault(k, v)

try:
    gold, index, places, PIN = load_data()
except Exception as e:  # pragma: no cover
    st.error(f"Could not load CareProof data ({type(e).__name__}: {e}). Run notebooks 02–03 and check app permissions.")
    st.stop()
try:
    MODEL = load_model()
except Exception:  # pragma: no cover - model tables are optional
    MODEL = None


def model_fit(fid, svc):
    if MODEL is None:
        return None
    try:
        r = MODEL["score_lookup"].loc[(fid, svc)]
        return float(r["model_score"]), str(r["score_band"])
    except KeyError:
        return None


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


# ------------------------------------------------------------------------------------------------ charts
def score_chart(res):
    """Stacked contributions: each bar sums to the facility's rank score (weights x components - penalty)."""
    rows, order = [], []
    for r in res:
        name = f"{r['rank']}. {clip(r['facility_name'], 28)}"
        order.append(name)
        for c, lab in SHORT_COMP.items():
            rows.append({"facility": name, "part": lab, "points": round(W[WKEY[c]] * float(r.get(c) or 0), 4)})
        if r.get("data_quality_penalty"):
            rows.append({"facility": name, "part": "Data-quality penalty", "points": -float(r["data_quality_penalty"])})
    return alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
        x=alt.X("sum(points):Q", title="rank score = sum of weighted parts"),
        y=alt.Y("facility:N", sort=order, title=None),
        color=alt.Color("part:N", scale=alt.Scale(domain=list(SHORT_COMP.values()) + ["Data-quality penalty"],
                                                  range=COMP_COLORS + ["#6e7781"]),
                        legend=alt.Legend(orient="bottom", columns=4, title=None)),
        tooltip=["facility", "part", alt.Tooltip("points:Q", format=".3f")],
    ).properties(height=max(140, 34 * len(res)))


def funnel_chart(summ, n_shown):
    df = pd.DataFrame([{"stage": "Mention the service", "n": summ["retrieved_mentions_in_radius"]},
                       {"stage": "Pass credibility checks", "n": summ["eligible_in_radius"]},
                       {"stage": "Shown", "n": n_shown}])
    base = alt.Chart(df).encode(y=alt.Y("stage:N", sort=list(df.stage), title=None),
                                x=alt.X("n:Q", title="facilities in radius"))
    return (base.mark_bar(color="#0969da", cornerRadiusEnd=4) + base.mark_text(align="left", dx=4).encode(text="n:Q")
            ).properties(height=120)


def _model_scale(feature, v):
    if v is None or pd.isna(v):
        return None
    return math.log1p(float(v)) if feature in LOG_FEATURES else float(v)


def profile_chart(fid, svc):
    """This facility vs the GA-optimal profile vs typical documented providers (top features, 0-1 scaled)."""
    if MODEL is None or MODEL.get("model_features") is None or fid not in MODEL["model_features"].index:
        return None
    prof = MODEL["model_optimal_profile"]
    prof = prof[prof.service == svc].sort_values("permutation_importance", ascending=False).head(8)
    feats = MODEL["model_features"]
    rows = []
    for p in prof.itertuples():
        col = feats[p.feature].astype(float)
        lo, hi = col.quantile(0.01), col.quantile(0.99)
        span = (hi - lo) or 1.0
        typical = p.documented_rate if p.kind == "binary" else _model_scale(p.feature, p.documented_median)
        for who, v in (("This facility", feats.loc[fid, p.feature]), ("GA optimal profile", _model_scale(p.feature, p.optimal_value)),
                       ("Typical documented provider", typical)):
            if v is not None and not pd.isna(v):
                rows.append({"feature": p.feature_label, "who": who, "value": max(0.0, min(1.0, (float(v) - lo) / span))})
    if not rows:
        return None
    return alt.Chart(pd.DataFrame(rows)).mark_circle(size=110).encode(
        x=alt.X("value:Q", title="low ← scaled value → high", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("feature:N", sort=list(prof.feature_label), title=None),
        color=alt.Color("who:N", scale=alt.Scale(domain=["This facility", "GA optimal profile", "Typical documented provider"],
                                                 range=["#cf222e", "#8250df", "#1a7f37"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["feature", "who", alt.Tooltip("value:Q", format=".2f")]).properties(height=34 * len(prof) + 40)


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
    go = c4.form_submit_button("Search", type="primary", use_container_width=True)
if go:
    run_search(svc, loc_text, radius, 5)

w1, w2 = st.columns([6, 1])
w1.caption("⚠️ Planning aid only. Call the facility to confirm the service and admission before travel. Emergency: **108 / 112**.")
with w2.popover("ℹ️ About"):
    st.write(VERIFICATION_WARNING)
    st.caption("Evidence levels describe how well the facility's public record documents the service, not whether it "
               "is available today. Distances are straight-line, not travel time.")

tab_find, tab_plan, tab_model = st.tabs(["🔎 Find & refer", "🗺️ Capacity planner", "🧠 Model & genetic optimisation"])

# ================================================================================================ FIND
with tab_find:
    if ss.err:
        st.error(ss.err, icon="📍")
    if ss.res is None:
        st.info("Pick a service, type a PIN code, district or coordinates, and press **Search**.")
    else:
        res, summ, q = ss.res, ss.summ, ss.q
        for w in summ.get("location_warnings") or []:
            st.caption(f"📍 {w}")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Credible options", summ["eligible_in_radius"], help="Passed every credibility gate within the radius")
        k2.metric("Nearest credible", f"{min(r['distance_km'] for r in res):.1f} km" if res else "–")
        k3.metric("Held back", summ["retrieved_mentions_in_radius"] - summ["eligible_in_radius"],
                  help="Mention the service but failed a check (negated, other facility, location conflict…)")
        k4.metric("Strong evidence", sum(r["evidence_category"] == "strong" for r in res), help="Among the options shown")

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown(f"**{SERVICE_SHORT[q['svc']]}** near **{md(q['origin'])}** within {q['radius']} km")
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
                cat = r["evidence_category"]
                fit = model_fit(r["facility_id"], q["svc"])
                with st.container(border=True):
                    a, b = st.columns([4, 1.6])
                    a.markdown(f"**{r['rank']}. {md(r['facility_name'])}**  \n"
                               f"{r['distance_km']:.1f} km · {md(clip(r['address'], 60))}")
                    b.markdown(f"{pill(EV_LABEL.get(cat, cat), EV_COLOR.get(cat, '#6e7781'))}<br>"
                               f"<span style='font-size:.85rem'>{esc(status_txt)}</span>", unsafe_allow_html=True)
                    st.markdown(f"> “{md(clip(r['evidence_excerpt']))}”")
                    m1, m2, m3 = st.columns(3)
                    m1.markdown(bar("Evidence", EV_LEVEL.get(cat, 0), EV_COLOR.get(cat, "#6e7781"), cat),
                                unsafe_allow_html=True)
                    m2.markdown(bar("Proximity", r.get("distance_score"), "#cf222e", f"{r['distance_km']:.0f} km"),
                                unsafe_allow_html=True)
                    if fit:
                        m3.markdown(bar("Profile fit (model)", fit[0], FIT_COLOR[fit[1]], fit[1]), unsafe_allow_html=True)
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
                        if fit:
                            st.markdown(f"**Profile fit: {fit[1]}**: the structured profile is {FIT_TEXT[fit[1]]}. "
                                        "A cross-check only; it does not change the ranking.")
                            ch = profile_chart(r["facility_id"], q["svc"])
                            if ch is not None:
                                st.altair_chart(ch, use_container_width=True)
            if res and summ["additional_eligible_beyond_top_k"] and ss.k < 10:
                if st.button("Show more options", key="more"):
                    run_search(q["svc"], q["text"], q["radius"], 10)
                    st.rerun()

            # ---------------------------------------------------------------------------------- unverified leads
            if MODEL is not None and MODEL.get("model_facilities") is not None:
                leads = unverified_leads(MODEL["model_scores"], MODEL["model_facilities"], gold, q["svc"], q["lat"],
                                         q["lon"], float(q["radius"]))
                if len(leads):
                    with st.expander(f"🧭 {len(leads)} unverified lead(s): profile looks like a provider, "
                                     "but no documented evidence"):
                        st.warning("**Not recommendations.** These records do **not** state the service. A model trained "
                                   "on structured fields thinks their profile resembles documented providers. Use only to "
                                   "decide whom to phone if the options above fail.", icon="⚠️")
                        st.dataframe(pd.DataFrame({"facility": leads["name"].values, "km": leads.distance_km.round(1).values,
                                                   "profile fit": leads.model_score.values, "phone": leads.phone.values,
                                                   "city": leads.address_city.values}),
                                     hide_index=True, use_container_width=True,
                                     column_config={"profile fit": st.column_config.ProgressColumn(
                                         "profile fit", min_value=0.0, max_value=1.0, format="%.2f")})

        with right:
            pts = [{"lat": q["lat"], "lon": q["lon"], "color": "#d1242f", "size": 220}] + \
                  [{"lat": r["location_lat"], "lon": r["location_lon"], "color": EV_COLOR[r["evidence_category"]], "size": 140}
                   for r in res]
            st.map(pd.DataFrame(pts), latitude="lat", longitude="lon", color="color", size="size")
            st.caption("🔴 patient location · 🟢 strong · 🔵 moderate · 🟡 weak evidence")
            if res:
                st.markdown("###### Why this order")
                st.altair_chart(score_chart(res), use_container_width=True)
            st.altair_chart(funnel_chart(summ, len(res)), use_container_width=True)

            # ------------------------------------------------------------------------------------ call & refer
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
                    done = sum(bool(v) for v in chk.values())
                    st.progress(done / max(1, len(items)), text=f"{done}/{len(items)} checks done")
                    outcome = st.radio("Call outcome", STATUSES, horizontal=True, key=f"o_{sel['facility_id']}",
                                       index=STATUSES.index(ss.status.get(sel["facility_id"], "Not contacted")))
                    ss.status[sel["facility_id"]] = outcome
                    if outcome in BLOCKING_STATUSES:
                        alt_r = next_alternative(res, ss.status, exclude=sel["facility_id"])
                        if alt_r:
                            st.warning(f"Next option: **{md(alt_r['facility_name'])}** ({alt_r['distance_km']:.1f} km)", icon="➡️")
                            if st.button("Switch to next option", key="switch"):
                                ss.sel = alt_r["facility_id"]
                                st.rerun()
                        else:
                            st.error("No other credible option in this radius. Widen the search or call 108 / the district referral desk.")
                    ss.notes = st.text_input("Notes (no patient names or numbers)", ss.notes, key="notes_box", max_chars=300)
                    text = referral_summary(q["label"], q["origin"], sel, res, outcome, chk, ss.notes, datetime.now(timezone.utc))
                    st.download_button("⬇️ Download referral summary", text, use_container_width=True, key="dl",
                                       file_name=f"careproof_referral_{datetime.now(timezone.utc):%Y%m%d_%H%M}.txt")
                    with st.expander("Preview summary"):
                        st.code(text, language=None)

        # -------------------------------------------------------------------------------------------- audit
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

# ================================================================================================ PLANNER
@st.cache_data(show_spinner="Optimising sites (greedy + genetic algorithm)…", max_entries=32)
def run_plan(svc, state, thr, k, equity, weighting, min_score):
    prob = CV.build_problem(PIN, gold, MODEL["model_facilities"], svc, state, float(thr), weighting,
                            features=MODEL.get("model_features"), scores=MODEL["model_scores"], min_score=min_score)
    if prob.D.shape[1] == 0:
        return {"prob_pins": prob.pins, "empty": True, "before": prob.metrics([]), "n_providers": len(prob.providers)}
    out = CV.plan(prob, int(k), float(equity))
    d_before, d_after = prob.distances([]), prob.distances(out["selected"])
    pins = prob.pins.assign(before=d_before <= thr, after=d_after <= thr)
    return {**out, "empty": False, "prob_pins": pins, "providers": prob.providers, "n_providers": len(prob.providers),
            "n_candidates": prob.D.shape[1], "curve": CV.budget_curve(prob, 10, float(equity)),
            "frontier": CV.equity_frontier(prob, int(k))}


with tab_plan:
    st.markdown("#### Where would adding a service close the most access gaps?")
    st.caption("For **planners**, not referrals. Picks existing hospitals that do *not* yet document the service and "
               "shows how many PIN-code areas would come within reach of a credible provider if the service were added "
               "there. A genetic algorithm (seeded with the classic greedy solution) searches the combinations.")
    if MODEL is None or MODEL.get("model_facilities") is None:
        st.info("Needs the model tables (notebook 07).")
    else:
        states = sorted(PIN.primary_state.dropna().unique())
        with st.form("plan", border=True):
            p1, p2, p3 = st.columns([2, 2, 1.2])
            p_svc = p1.selectbox("Service", list(TAXONOMY), format_func=SERVICE_SHORT.get, key="p_svc",
                                 index=list(TAXONOMY).index(ss.q["svc"]) if ss.q else 0)
            p_state = p2.selectbox("State", states, index=states.index("BIHAR") if "BIHAR" in states else 0, key="p_state")
            p_thr = p3.selectbox("Reachable within", [25, 50, 100], index=1, format_func=lambda x: f"{x} km", key="p_thr")
            p4, p5, p6, p7 = st.columns([1.4, 1.6, 1.6, 1.4])
            p_k = p4.slider("Sites to add", 1, 10, 5, key="p_k")
            p_eq = p5.slider("Equity emphasis", 0.0, 1.0, 0.5, 0.25, key="p_eq",
                             help="0 = maximise demand covered; 1 = give every district equal weight (favours sparse districts)")
            p_w = p6.radio("Demand weight", ["post_offices", "equal"], key="p_w", horizontal=True,
                           format_func={"post_offices": "post offices", "equal": "each PIN equal"}.get,
                           help="No population data exists in the dataset; post offices per PIN are a rough settlement proxy")
            p_ms = p7.selectbox("Candidate hospitals", [None, 0.05, 0.2], key="p_ms",
                                format_func=lambda v: "any plausible" if v is None else f"profile fit ≥ {v}")
            plan_go = st.form_submit_button("Optimise", type="primary")
        if plan_go or ss.get("plan_args"):
            if plan_go:
                ss.plan_args = (p_svc, p_state, p_thr, p_k, p_eq, p_w, p_ms)
            args = ss.plan_args
            R = run_plan(*args)
            thr = args[2]
            if R["empty"]:
                st.warning(f"No candidate hospital in {args[1].title()} would bring new PIN areas within {thr} km "
                           f"({R['n_providers']} credible providers already; coverage {R['before']['coverage']:.0%}).")
            else:
                b, a = R["before"], R["after"]
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Demand within reach", f"{a['coverage']:.0%}", f"{(a['coverage'] - b['coverage']) * 100:+.1f} pts")
                t2.metric("Districts with no access", a["districts_without_access"],
                          f"{a['districts_without_access'] - b['districts_without_access']:+d} of {a['n_districts']}",
                          delta_color="inverse")
                t3.metric("Avg. distance to credible provider", f"{a['mean_km']:.0f} km", f"{a['mean_km'] - b['mean_km']:+.0f} km",
                          delta_color="inverse")
                o = R["objective"]
                t4.metric("Gain vs random sites", f"{(o['ga'] - o['random_mean']) * 100:+.1f} pts",
                          help="Objective of the optimised plan minus the mean of 200 random plans of the same size")

                m1, m2 = st.columns([3, 2], gap="large")
                with m1:
                    pins = R["prob_pins"]
                    status = pins.before.map({True: "covered already", False: "uncovered"}).where(~(pins.after & ~pins.before), "newly covered")
                    colors = {"covered already": "#8c959f", "newly covered": "#8250df", "uncovered": "#cf222e"}
                    pts = pd.DataFrame({"lat": pins.lat, "lon": pins.lon, "color": status.map(colors), "size": 1500})
                    prov = R["providers"]
                    pts = pd.concat([pts, pd.DataFrame({"lat": prov.lat, "lon": prov.lon, "color": "#1a7f37", "size": 3500}),
                                     pd.DataFrame({"lat": R["sites"].lat, "lon": R["sites"].lon, "color": "#ffa500", "size": 9000})])
                    st.map(pts, latitude="lat", longitude="lon", color="color", size="size")
                    st.caption("⚪ PIN already within reach · 🟣 newly within reach · 🔴 still out of reach · "
                               "🟢 existing credible provider · 🟠 proposed site")
                with m2:
                    st.markdown("###### Proposed sites")
                    sites = R["sites"]
                    st.dataframe(pd.DataFrame({"hospital": sites["name"].values, "city": sites.address_city.values,
                                               "new demand reached": sites.new_demand_share.values,
                                               "profile fit": sites.model_score.values}),
                                 hide_index=True, use_container_width=True,
                                 column_config={"new demand reached": st.column_config.ProgressColumn(
                                     "new demand reached", min_value=0.0, max_value=float(max(sites.new_demand_share.max(), 1e-9)),
                                     format="%.3f"),
                                     "profile fit": st.column_config.ProgressColumn("profile fit", min_value=0.0,
                                                                                    max_value=1.0, format="%.2f")})
                    st.caption(f"{R['n_candidates']} candidate hospitals could each reach new PIN areas; "
                               f"{R['n_providers']} credible providers already exist in or near the state.")
                    solv = pd.DataFrame({"plan": ["today", "random (mean)", "random (best 5%)", "greedy",
                                                  "GA, random start", "GA seeded with greedy"],
                                         "objective": [o["before"], o["random_mean"], o["random_p95"], o["greedy"],
                                                       o["ga_random_start"], o["ga"]]})
                    st.markdown("###### Solver comparison")
                    st.altair_chart(alt.Chart(solv).mark_bar(cornerRadiusEnd=3).encode(
                        x=alt.X("objective:Q", title="objective (share of weighted demand within reach)",
                                scale=alt.Scale(zero=False)),
                        y=alt.Y("plan:N", sort=list(solv.plan), title=None),
                        color=alt.condition(alt.datum.plan == "GA seeded with greedy", alt.value("#8250df"), alt.value("#afb8c1")),
                        tooltip=["plan", alt.Tooltip("objective:Q", format=".4f")]).properties(height=200),
                        use_container_width=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("###### Diminishing returns")
                    cur = R["curve"].melt(["sites_added"], ["coverage", "district_coverage"], "measure", "share")
                    cur["measure"] = cur.measure.map({"coverage": "demand-weighted", "district_coverage": "district-equal"})
                    st.altair_chart(alt.Chart(cur).mark_line(point=True).encode(
                        x=alt.X("sites_added:Q", title="sites added"), y=alt.Y("share:Q", title="within reach", axis=alt.Axis(format="%")),
                        color=alt.Color("measure:N", legend=alt.Legend(orient="bottom", title=None))).properties(height=220),
                        use_container_width=True)
                with c2:
                    st.markdown("###### Efficiency vs equity")
                    fr = R["frontier"]
                    st.altair_chart(alt.Chart(fr).mark_line(point=alt.OverlayMarkDef(size=80), color="#8250df").encode(
                        x=alt.X("coverage:Q", title="demand-weighted coverage", scale=alt.Scale(zero=False), axis=alt.Axis(format="%")),
                        y=alt.Y("district_coverage:Q", title="district-equal coverage", scale=alt.Scale(zero=False), axis=alt.Axis(format="%")),
                        order="equity_weight:Q",
                        tooltip=["equity_weight", alt.Tooltip("coverage:Q", format=".1%"),
                                 alt.Tooltip("district_coverage:Q", format=".1%"), "districts_without_access"]
                    ).properties(height=220), use_container_width=True)
                    st.caption("Each point is the best plan for one equity setting (0 → 1).")
                with c3:
                    st.markdown("###### Genetic algorithm convergence")
                    h = pd.concat([pd.DataFrame(R["history"]).assign(run="seeded with greedy"),
                                   pd.DataFrame(R["history_random_start"]).assign(run="random start")])
                    st.altair_chart(alt.Chart(h).mark_line().encode(
                        x="generation:Q", y=alt.Y("best:Q", title="best objective", scale=alt.Scale(zero=False)),
                        color=alt.Color("run:N", scale=alt.Scale(range=["#8250df", "#bf8700"]),
                                        legend=alt.Legend(orient="bottom", title=None))).properties(height=220),
                        use_container_width=True)

                st.markdown("###### Districts that gain the most")
                dt = R["districts"].assign(gain=lambda d: d.after - d.before).sort_values(["gain", "before"], ascending=[False, True]).head(12)
                dd = dt.melt(["district"], ["before", "after"], "when", "share")
                st.altair_chart((alt.Chart(dt).mark_rule(color="#afb8c1", strokeWidth=3).encode(
                    y=alt.Y("district:N", sort=list(dt.district), title=None), x="before:Q", x2="after:Q") +
                    alt.Chart(dd).mark_circle(size=120).encode(
                    y=alt.Y("district:N", sort=list(dt.district)), x=alt.X("share:Q", title="share of district demand within reach",
                                                                           axis=alt.Axis(format="%")),
                    color=alt.Color("when:N", scale=alt.Scale(domain=["before", "after"], range=["#cf222e", "#8250df"]),
                                    legend=alt.Legend(orient="bottom", title=None)))).properties(height=28 * len(dt) + 40),
                    use_container_width=True)
                st.caption("Straight-line distance, not travel time. Demand is a proxy (post offices or equal PIN weights), "
                           "not population. \"Adding\" a service is hypothetical: staff, equipment and cost are not modelled. "
                           "Existing providers are facilities whose records credibly document the service, not verified capacity.")


# ================================================================================================ MODEL
with tab_model:
    if MODEL is None:
        st.info("Model tables not found. Run `notebooks/07_service_profile_model` (or `python scripts/run_local.py model`).")
    else:
        st.markdown("#### What does a facility that documents this service look like?")
        st.caption("A supervised model learns, from **structured fields only** (type, operator, size, staff, specialty "
                   "tags, digital footprint, local density), whether a facility's own text documents a service. The "
                   "label comes from the evidence engine, not from human review, so the model describes **records**, "
                   "not real-world availability. It never changes who is recommended.")
        services = [s for s in TAXONOMY if s in set(MODEL["model_profile_summary"].service)]
        default = services.index(ss.q["svc"]) if ss.q and ss.q["svc"] in services else 0
        msvc = st.selectbox("Service", services, index=default, format_func=SERVICE_SHORT.get, key="msvc")
        summ_m = MODEL["model_profile_summary"].set_index("service").loc[msvc]
        met = MODEL["model_metrics"]
        mt = met[(met.service == msvc) & (met.split == "test")].assign(chosen=lambda d: truthy(d.chosen))
        rule = mt[mt.model == "specialty_tag_rule"].iloc[0]
        chosen = mt[mt.chosen].iloc[0]
        fs = MODEL["model_feature_selection"]
        fs = fs[fs.service == msvc]

        used = bool(truthy(pd.Series([summ_m.get("model_used", True)])).iloc[0])
        a, b, c, d = st.columns(4)
        a.metric("Avg. precision (held-out states)", f"{chosen.average_precision:.2f}",
                 f"{chosen.average_precision - rule.average_precision:+.2f} vs tag rule" if used else "tag rule chosen")
        b.metric("Base rate", f"{chosen.base_rate:.1%}", help="Share of test facilities whose text documents the service")
        if used:
            c.metric("Features kept", f"{int(summ_m.n_features)} / {len(fs)}",
                     help="Filter (near-constant, mutual information, correlation) then genetic-algorithm wrapper")
            d.metric("GA optimal-profile score", f"{summ_m.profile_score:.2f}",
                     f"median documented {summ_m.documented_median_score:.2f}", delta_color="off")
        else:
            st.warning("For this service no trained model beat the simple specialty-tag rule on the dev states, so **no "
                       "model is used**: no profile fit and no unverified leads are shown for it.", icon="ℹ️")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("###### Model comparison on held-out states")
            mt2 = mt.assign(system=mt.model.map({"specialty_tag_rule": "tag rule", "logreg": "logistic",
                                                 "hgb": "boosted trees"}) + " · " + mt.feature_set)
            base = alt.Chart(mt2).encode(
                y=alt.Y("system:N", sort="-x", title=None),
                x=alt.X("average_precision:Q", title="average precision (higher is better)", scale=alt.Scale(domain=[0, 1])))
            rule_line = alt.Chart(pd.DataFrame({"x": [chosen.base_rate]})).mark_rule(
                strokeDash=[4, 3], color="#cf222e").encode(x="x:Q")
            st.altair_chart((base.mark_bar(cornerRadiusEnd=3).encode(
                color=alt.Color("chosen:N", scale=alt.Scale(domain=[True, False], range=["#8250df", "#afb8c1"]), legend=None),
                tooltip=["system", alt.Tooltip("average_precision:Q", format=".3f"), alt.Tooltip("roc_auc:Q", format=".3f"),
                         alt.Tooltip("brier:Q", format=".3f"), "n_features"]) + rule_line).properties(height=220),
                use_container_width=True)
            st.caption("Purple = model chosen on the **dev** states. Red dashed line = base rate (random guessing).")
        with g2:
            if not used:
                st.stop()
            st.markdown("###### Which features matter (permutation importance)")
            imp = fs[truthy(fs.in_final_model)].sort_values("permutation_importance", ascending=False).head(12)
            st.altair_chart(alt.Chart(imp).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("permutation_importance:Q", title="drop in average precision when shuffled"),
                y=alt.Y("feature_label:N", sort="-x", title=None),
                color=alt.Color("group:N", legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["feature_label", alt.Tooltip("permutation_importance:Q", format=".4f"),
                         alt.Tooltip("mutual_info:Q", format=".4f")]).properties(height=260), use_container_width=True)

        g3, g4 = st.columns(2)
        hist = MODEL["model_ga_history"]
        hist = hist[hist.service == msvc]
        for col, task, title, ytitle, color, note in (
                (g3, "feature_selection", "Genetic algorithm 1: feature selection", "CV avg. precision − size penalty",
                 "#8250df", "Each chromosome is a feature on/off mask, scored by state-grouped cross-validation on dev states."),
                (g4, "optimal_profile", "Genetic algorithm 2: optimal provider profile", "model score − realism penalty",
                 "#1a7f37", "Each chromosome is a full facility profile, kept near real documented facilities by a "
                            "nearest-neighbour realism penalty.")):
            with col:
                st.markdown(f"###### {title}")
                h = hist[hist.task == task].melt(["generation"], ["best", "mean"], "series", "fitness")
                st.altair_chart(alt.Chart(h).mark_line(point=True).encode(
                    x=alt.X("generation:Q"), y=alt.Y("fitness:Q", title=ytitle, scale=alt.Scale(zero=False)),
                    color=alt.Color("series:N", scale=alt.Scale(domain=["best", "mean"], range=[color, "#afb8c1"]),
                                    legend=alt.Legend(orient="bottom", title=None))).properties(height=220),
                    use_container_width=True)
                st.caption(note)

        st.markdown("###### The GA-optimal profile vs real facilities")
        prof = MODEL["model_optimal_profile"]
        prof = prof[prof.service == msvc].sort_values("permutation_importance", ascending=False)
        pb = prof[prof.kind == "binary"]
        if len(pb):
            dfb = pd.concat([pb.assign(who="Documented providers", v=pb.documented_rate),
                             pb.assign(who="Others", v=pb.undocumented_rate),
                             pb.assign(who="GA optimal profile", v=pb.optimal_value)])
            st.altair_chart(alt.Chart(dfb).mark_circle(size=120).encode(
                x=alt.X("v:Q", title="share of facilities with the attribute (optimal profile: 0 = no, 1 = yes)",
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("feature_label:N", sort=list(pb.feature_label), title=None),
                color=alt.Color("who:N", scale=alt.Scale(domain=["GA optimal profile", "Documented providers", "Others"],
                                                         range=["#8250df", "#1a7f37", "#afb8c1"]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["feature_label", "who", alt.Tooltip("v:Q", format=".2f")]).properties(height=26 * len(pb) + 50),
                use_container_width=True)
        pn = prof[prof.kind == "numeric"]
        if len(pn):
            st.dataframe(pd.DataFrame({"attribute": pn.feature_label.values, "GA optimal": pn.optimal_value.round(1).values,
                                       "documented median": pn.documented_median.round(1).values,
                                       "others median": pn.undocumented_median.round(1).values}),
                         hide_index=True, use_container_width=True)
        st.caption(f"Closest real documented facility to the optimal profile: **{md(summ_m.nearest_real_facility_name)}**. "
                   "The profile describes what the model associates with documented service. It is **not** a causal "
                   "recipe: adding a tag does not create a dialysis unit.")

        with st.expander("Calibration and full metrics"):
            cal = MODEL["model_calibration"]
            cal = cal[cal.service == msvc]
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                strokeDash=[4, 3], color="#8c959f").encode(x="x:Q", y="y:Q")
            st.altair_chart((diag + alt.Chart(cal).mark_line(point=True, color="#8250df").encode(
                x=alt.X("mean_predicted:Q", title="mean predicted score", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("observed_rate:Q", title="observed documented rate", scale=alt.Scale(domain=[0, 1])),
                tooltip=["n", alt.Tooltip("mean_predicted:Q", format=".2f"), alt.Tooltip("observed_rate:Q", format=".2f")])
                ).properties(height=260), use_container_width=True)
            st.dataframe(met[met.service == msvc].drop(columns=["service"]), hide_index=True, use_container_width=True)
