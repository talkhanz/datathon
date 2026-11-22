"""Download and parse the official external reference lists named in evaluation/external_sources_v1.csv.

Source: State Health Agency Haryana, Ayushman Bharat PM-JAY (official government publications).
Raw PDFs go to data/external/raw/ (git-ignored, never committed). The parsed table keeps ONLY hospital ID, name,
type, district, empanelled specialities and list status. Nodal officers' phone numbers and e-mails are discarded.

Usage:  python scripts/fetch_external.py        -> data/external/careproof_ext_pmjay_hospitals.parquet
Needs:  pdfplumber (dev only), curl
"""
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from careproof import external as E  # noqa: E402

RAW = os.path.join(ROOT, "data", "external", "raw")
OUT = os.path.join(ROOT, "data", "external", "careproof_ext_pmjay_hospitals.parquet")


def main():
    os.makedirs(RAW, exist_ok=True)
    man = pd.read_csv(os.path.join(ROOT, "evaluation", "external_sources_v1.csv"), dtype=str)
    parsed = {}
    for r in man.itertuples():
        path = os.path.join(RAW, f"{r.source_id}.pdf")
        if not os.path.exists(path):
            subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0", "--max-time", "120", "-o", path, r.url], check=True)
        df = E.parse_pmjay_pdf(path, None if r.district == "ALL" else r.district)
        parsed[r.source_id] = df.assign(list_type=r.list_type, list_date=r.list_date, source_id=r.source_id)
        print(f"{r.source_id:40s} {len(df):4d} rows")
    ref = E.build_reference(parsed)
    ref.to_parquet(OUT, index=False)
    print("wrote", OUT, len(ref), "hospitals;", ref.has_specialities.sum(), "with specialities")


if __name__ == "__main__":
    main()
