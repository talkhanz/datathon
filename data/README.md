# Data

The authoritative data are the read-only Databricks Marketplace tables:

| Key | Table |
|---|---|
| facilities | `databricks_virtue_foundation_dataset_dais_2026`.`virtue_foundation_dataset`.`facilities` |
| pincode | `databricks_virtue_foundation_dataset_dais_2026`.`virtue_foundation_dataset`.`india_post_pincode_directory` |
| nfhs | `databricks_virtue_foundation_dataset_dais_2026`.`virtue_foundation_dataset`.`nfhs_5_district_health_indicators` |

For local development, place CSV exports in `data/raw/` (git-ignored; **do not commit** — redistribution
rights are not confirmed):

```
data/raw/facilities.csv
data/raw/india_post_pincode_directory.csv
data/raw/nfhs_5_district_health_indicators.csv
```

`python scripts/run_local.py build` writes derived pickles to `data/derived/` (also git-ignored).

## Observed local-file facts (2026-11-21; Databricks is authoritative if it differs)

* `facilities.csv`: 10,088 parsed rows × 51 columns. Missing values are the literal string `null`.
  88 rows are fragments of multi-line text that broke CSV quoting (non-UUID `unique_id`) → quarantined.
  11 exact duplicate rows. 30 rows without coordinates; 6 outside India. Array fields are JSON strings,
  capped at 50 elements, and frequently repeat elements.
* `india_post_pincode_directory.csv`: 165,627 post-office rows, 19,586 PINs (post-office grain, not PIN grain).
  12,007 rows with `NA` coordinates, ~2,600 outside India (incl. ~800 that look lat/lon-swapped), a few
  degree-symbol strings. 1,478 PINs span multiple districts, 290 span multiple states.
* `nfhs_5_district_health_indicators.csv`: 706 districts × 109 columns. `*` = suppressed (4,125 cells),
  `(x)` = small-sample estimate (5,068 cells), trailing spaces common, state typos (`Maharastra`).
