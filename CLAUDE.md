# CareProof India – notes for AI assistants and contributors

## Hard rules
- Marketplace tables under `databricks_virtue_foundation_dataset_dais_2026` are **read-only**. Never write to them.
- Write only `careproof_`-prefixed tables, only into the configured `target_catalog.target_schema`
  (default `workspace.careproof_india`). Use `fq_table()` / `safe_write()`; they enforce this.
- Never drop a catalog or schema. The only DROP in the project is the uniquely named
  `careproof_tmp_writecheck_*` table in notebook 00.
- Never commit data (`data/raw`, `data/derived`) or credentials.
- Never fabricate metrics, quotes, URLs. Precision@K stays `PENDING_HUMAN_REVIEW` until humans label.
- A recommendation must have a non-empty exact excerpt from the facility's own fields.
- Weights/penalties in `src/careproof/config.py` are policy choices, not learned. Say so.

## Layout
- `src/careproof/` pure Python business logic (no Spark). Tested by `tests/` with `pytest`.
- `src/careproof/pipeline.py` builds tables on pandas; identical code runs locally and in notebooks.
- `notebooks/` Databricks source notebooks; `_common.py` is `%run` by all others.
- `scripts/run_local.py build|query|eval` local end-to-end on CSVs.

## Commands
```
python -m pytest -q tests
python scripts/run_local.py build
python scripts/run_local.py query dialysis pincode 800001 25 5
python scripts/run_local.py eval
```
