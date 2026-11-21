"""Regenerate docs/data_quality_report.md and docs/figures/*.png from local data (aggregates only).

Usage: python scripts/make_eda_report.py   (after scripts/run_local.py build)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from careproof import eda  # noqa: E402

FIG = os.path.join(ROOT, "docs", "figures")
COL = {"explicit_text": "#1b7f3b", "specialty_tag_only": "#d4a72c", "negated_only": "#b42318",
       "keyword_only (baseline hit, no evidence)": "#8c959f"}


def md_table(df: pd.DataFrame, max_rows: int = 25) -> str:
    df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def figs(r):
    os.makedirs(FIG, exist_ok=True)
    # F1 claims vs evidence
    t = r["Q4a"].set_index("service")[list(COL)]
    ax = t.plot(kind="barh", stacked=True, color=[COL[c] for c in t.columns], figsize=(9, 4.6))
    ax.set_xlabel("facilities with any signal for the service"); ax.set_ylabel("")
    ax.set_title("Specialty tags overstate services: only 452 of 4,614 C-section signals are explicit text", fontsize=10)
    for i, (e, tot) in enumerate(zip(t["explicit_text"], t.sum(axis=1))):
        ax.text(tot + 40, i, f"{100 * e / tot:.0f}% explicit", va="center", fontsize=8)
    ax.set_xlim(0, t.sum(axis=1).max() * 1.18)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False); plt.tight_layout(); plt.savefig(f"{FIG}/f1_claims_vs_evidence.png", dpi=130); plt.close()
    # F2 location consistency
    t = r["Q3b"]
    ax = t.plot(kind="bar", x="facility_to_own_PIN_centroid", y="pct", legend=False, color="#0969da", figsize=(7, 3.5))
    for i, v in enumerate(t.pct):
        ax.text(i, v + 0.5, f"{v}%", ha="center", fontsize=8)
    ax.set_ylabel("% of checkable facilities"); ax.set_xlabel("distance from facility coordinates to its own PIN centroid")
    ax.set_title("1 in 6 facilities sits >25 km from its own PIN code"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(f"{FIG}/f2_location_vs_pin.png", dpi=130); plt.close()
    # F3 rurality coverage
    t = r["Q6a"]
    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax1.bar(t.rurality_quartile.astype(str), t.total_facilities, color="#8250df")
    ax1.set_ylabel("facilities in dataset")
    for i, (n, z, d) in enumerate(zip(t.total_facilities, t.districts_with_zero_facilities, t.districts)):
        ax1.text(i, n + 80, f"{int(n)} facilities\n{z}/{d} districts empty", ha="center", fontsize=8)
    ax1.set_title("Coverage collapses in rural districts (proxy: share of branch post offices)")
    ax1.set_ylim(0, t.total_facilities.max() * 1.25)
    plt.tight_layout(); plt.savefig(f"{FIG}/f3_rural_coverage.png", dpi=130); plt.close()
    # F4 flags
    t = r["Q4b"][~r["Q4b"].flag.isin(["evidence_list_truncated"])]
    ax = t.plot(kind="barh", x="flag", y="pct_of_candidates", legend=False, color="#cf222e", figsize=(7, 3.5))
    ax.set_xlabel("% of recommendable facility-service rows"); ax.set_ylabel("")
    ax.set_title("A quarter of explicit claims carry a contradiction signal")
    plt.tight_layout(); plt.savefig(f"{FIG}/f4_flag_rates.png", dpi=130); plt.close()


CAPTIONS = {
    "Q1": "Implication: `unique_id` is a valid key after removing 11 exact duplicates; `content_table_id` is NOT an identifier (shared by unrelated facilities). Co-located records with different names suggest coordinates snapped to a shared point (e.g. a locality centroid), so precise positions cannot be assumed.",
    "Q2a": "Implication: `capacity` and `numberDoctors` are mostly missing and are not evidence of a service anyway; they are excluded from scoring. Description is always present but often short (Q4c).",
    "Q2b": "Implication: clinics and dentists lack equipment lists more often than hospitals; absence of an equipment mention is therefore weak negative evidence and is never penalised.",
    "Q2c": "Implication: missingness varies modestly by state (procedure missing 5-12%); no state is systematically empty, but Bihar/Jharkhand/Odisha are slightly worse.",
    "Q2d": "Implication: procedure and capability go missing together far more than chance (lift ~10): missingness is systematic (whole-record extraction failures), not random. Records missing both are a distinct low-evidence group.",
    "Q3a": "Implication: the PIN directory has ~2,600 unusable coordinates incl. 791 lat/lon swaps; swaps are flagged, never silently corrected, and PIN centroids use outlier-robust medians.",
    "Q3b": "Implication: 16.8% of facilities are >25 km and 11.2% >100 km from their own PIN centroid. Either the coordinates or the address is wrong (or the record merges facilities). This is the `location_conflict` flag and lowers `location_quality_score`.",
    "Q3c": "Implication: 2,192 PINs have geocoded offices that disagree; CareProof refuses to place a query at those PINs and asks for coordinates instead of guessing.",
    "Q4a": "Implication: specialty tags overstate services: only 9.8% of facilities with any C-section signal state it explicitly. Tag-only facilities are never recommended.",
    "Q4b": "Implication: 25.6% of recommendable rows have at least one substantive contradiction (location, state, specialty profile, negation) -> shown as warnings and penalties, not hidden. 61% hit the 50-item array cap, so evidence lists may be truncated (missing-data warning, no penalty).",
    "Q4c": "Implication: descriptions are short (median 16 words) and essentially English-only (9 contain Devanagari); facilities described only in Indian languages would be missed -> language/source bias.",
    "Q4d": "Implication: evidence comes predominantly from social media and aggregators (Facebook 88%, Justdial 54%), not official registries. Claims must be treated as self-reported.",
    "Q5a": "Implication: several NFHS indicators are mostly suppressed or small-sample; CareProof shows only service-relevant indicators, with raw values and flags, as district context.",
    "Q5b": "Implication: no percentage indicator is outside [0,100] after parsing.",
    "Q5c": "Implication: 84.7% of NFHS districts match an India Post district by normalized name+state; unmatched districts (renamed/split districts) get no context rather than a fuzzy guess.",
    "Q6a": "Implication: the most rural quartile of districts (proxy) contains 2.0% of facilities and 114 of 189 such districts have none. Recommendations there will be far away or absent: CareProof must abstain honestly and flag coverage, and benchmark results on urban queries do not generalise to rural ones.",
    "Q6b": "Implication: absolute counts of explicitly evidenced providers are small outside the large states (e.g. Odisha: 2 C-section, 1 burn care). Absence in this dataset is not absence on the ground.",
}


def main():
    rd = lambda n: pd.read_csv(f"{ROOT}/data/raw/{n}.csv", dtype=str, keep_default_na=False)
    dv = lambda n: pd.read_parquet(f"{ROOT}/data/derived/careproof_{n}.parquet")
    r = eda.run_all(rd("facilities"), rd("india_post_pincode_directory"), dv("facilities_silver"),
                    dv("facilities_quarantine"), dv("pincode_silver"), dv("nfhs_silver"), dv("facility_service_gold"))
    figs(r)
    out = ["# Data-quality and exploratory analysis report",
           "", "Generated by `scripts/make_eda_report.py` from the **local CSV exports** (2026-11-21) using "
           "`src/careproof/eda.py`. The same functions run on the Marketplace tables in "
           "`notebooks/01b_exploratory_analysis.py`; if numbers differ, Databricks is authoritative. Each table answers "
           "one question; the caption states the implication for the system. Observations are measured; items marked "
           "*proxy* or *hypothesis* are not.", "",
           "![claims vs evidence](figures/f1_claims_vs_evidence.png)", "",
           "![location vs PIN](figures/f2_location_vs_pin.png)", "",
           "![rural coverage](figures/f3_rural_coverage.png)", "",
           "![flags](figures/f4_flag_rates.png)", ""]
    for code, title, _ in eda.ALL_QUESTIONS:
        out += [f"## {code}. {title}", "", md_table(r[code]) if len(r[code]) else "_No rows (check passed)._", "",
                f"*{CAPTIONS.get(code, '')}*", ""]
    with open(os.path.join(ROOT, "docs", "data_quality_report.md"), "w") as f:
        f.write("\n".join(out))
    print("wrote docs/data_quality_report.md and", len(os.listdir(FIG)), "figures")


if __name__ == "__main__":
    main()
