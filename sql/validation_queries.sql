-- CareProof India: validation queries (read-only).
-- Run in a Databricks SQL editor or a notebook %sql cell AFTER notebooks 02 and 03.
-- If you changed target_catalog / target_schema, replace `workspace`.`careproof_india` below.
-- Every query is written so that a PASSING result is either zero rows or a row with status = 'PASS'.

USE CATALOG workspace;
USE SCHEMA careproof_india;

-- 1. Row-count reconciliation: bronze = silver + quarantine + exact duplicates removed
SELECT
  (SELECT COUNT(*) FROM databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities) AS bronze_rows,
  (SELECT COUNT(*) FROM careproof_facilities_silver)                                                         AS silver_rows,
  (SELECT COALESCE(SUM(n_exact_source_copies - 1), 0) FROM careproof_facilities_silver)                      AS exact_dups_removed,
  (SELECT COUNT(*) FROM careproof_facilities_rejected_rows)                                                  AS quarantined_rows,
  CASE WHEN bronze_rows = silver_rows + exact_dups_removed + quarantined_rows THEN 'PASS' ELSE 'FAIL' END   AS status;
-- NOTE: if careproof_facilities_rejected_rows does not exist (no corrupted rows on Databricks), replace that subquery with 0.

-- 2. PIN silver is one row per PIN and reconciles to the post-office directory
SELECT
  (SELECT COUNT(DISTINCT pincode) FROM databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.india_post_pincode_directory
     WHERE CAST(pincode AS STRING) RLIKE '^[1-9][0-9]{5}$')       AS bronze_distinct_valid_pins,
  (SELECT COUNT(*) FROM careproof_pincode_silver)                   AS silver_pins,
  (SELECT SUM(post_office_count) FROM careproof_pincode_silver)     AS silver_post_offices,
  CASE WHEN bronze_distinct_valid_pins = silver_pins THEN 'PASS' ELSE 'FAIL' END AS status;

-- 3. Duplicate identifiers (expect zero rows each)
SELECT facility_id, COUNT(*) n FROM careproof_facilities_silver GROUP BY facility_id HAVING n > 1;
SELECT pincode, COUNT(*) n FROM careproof_pincode_silver GROUP BY pincode HAVING n > 1;
SELECT facility_id, service, COUNT(*) n FROM careproof_facility_service_gold GROUP BY ALL HAVING n > 1;
SELECT district_norm, state_norm, indicator, COUNT(*) n FROM careproof_nfhs_silver GROUP BY ALL HAVING n > 1;

-- 4. Invalid coordinates (expect zero rows): every usable location lies inside India's bounding box
SELECT facility_id, name, location_lat, location_lon, location_source
FROM careproof_facilities_silver
WHERE location_lat IS NOT NULL AND NOT (location_lat BETWEEN 6 AND 37.6 AND location_lon BETWEEN 68 AND 97.5);

SELECT pincode, centroid_lat, centroid_lon FROM careproof_pincode_silver
WHERE coords_defensible AND NOT (centroid_lat BETWEEN 6 AND 37.6 AND centroid_lon BETWEEN 68 AND 97.5);

-- 5. PIN codes preserved as six-digit strings (expect zero rows)
SELECT pincode FROM careproof_pincode_silver WHERE NOT pincode RLIKE '^[1-9][0-9]{5}$';
SELECT facility_id, pincode_norm FROM careproof_facilities_silver
WHERE pincode_norm IS NOT NULL AND NOT pincode_norm RLIKE '^[1-9][0-9]{5}$';

-- 6. Null evidence: no ELIGIBLE gold row may lack an excerpt or provenance (expect zero rows)
SELECT facility_id, service FROM careproof_facility_service_gold
WHERE eligible AND (evidence_excerpt IS NULL OR trim(evidence_excerpt) = '' OR evidence_field IS NULL OR source_row_id IS NULL);

-- 6b. Gate integrity: conflicting / insufficient evidence or high-severity findings must never be eligible (expect zero rows)
SELECT facility_id, service, evidence_category FROM careproof_facility_service_gold
WHERE eligible AND (evidence_category IN ('conflicting', 'insufficient')
                    OR facility_findings_json LIKE '%"severity": "high"%' OR claim_findings_json LIKE '%"severity": "high"%');

-- 6c. Quarantine table is well-formed (expect zero rows)
SELECT * FROM careproof_facilities_quarantine
WHERE reason_code IS NULL OR severity NOT IN ('high', 'medium') OR pipeline_version IS NULL;

-- 7. Excerpt provenance: the excerpt must be an exact substring of the cited original field (expect zero rows)
SELECT g.facility_id, g.service, g.evidence_field
FROM careproof_facility_service_gold g JOIN careproof_facilities_silver s USING (facility_id)
WHERE g.eligible AND NOT (
  CASE g.evidence_field
    WHEN 'description' THEN instr(s.description, g.evidence_excerpt) > 0
    WHEN 'name'        THEN instr(s.name, g.evidence_excerpt) > 0
    WHEN 'procedure'   THEN instr(s.procedure, g.evidence_excerpt) > 0
    WHEN 'equipment'   THEN instr(s.equipment, g.evidence_excerpt) > 0
    WHEN 'capability'  THEN instr(s.capability, g.evidence_excerpt) > 0
    ELSE false END);
-- NOTE: array fields are stored as JSON text; excerpts containing quotes or backslashes are JSON-escaped there,
-- so a handful of rows may appear here as false positives. The Python check in notebook 05 is authoritative.

-- 8. Join fanout (expect fanout_rows = 0 for every join)
SELECT name, left_rows, output_rows, fanout_rows, match_rate, duplicate_key_rate,
       CASE WHEN fanout_rows = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM careproof_join_report;

-- 9. Gold references only known facilities (expect zero rows)
SELECT g.facility_id FROM careproof_facility_service_gold g
LEFT ANTI JOIN careproof_facilities_silver s USING (facility_id);

-- 10. Score ranges (expect zero rows)
SELECT facility_id, service FROM careproof_facility_service_gold
WHERE NOT (service_relevance_score BETWEEN 0 AND 1 AND evidence_strength_score BETWEEN 0 AND 1
           AND readiness_evidence_score BETWEEN 0 AND 1 AND location_confidence_score BETWEEN 0 AND 1
           AND identity_confidence_score BETWEEN 0 AND 1);

-- 11. Recommendations missing citations or provenance (from the last evaluation run; expect zero rows)
SELECT query_id, rank, facility_id FROM careproof_eval_results_detail
WHERE evidence_excerpt IS NULL OR trim(evidence_excerpt) = '' OR evidence_field IS NULL
   OR final_ranking_score NOT BETWEEN 0 AND 1 OR data_quality_penalty NOT BETWEEN 0 AND 1
   OR evidence_category IN ('conflicting', 'insufficient');

-- 12. Evidence-level overview (informational)
SELECT service, evidence_category, eligible, COUNT(*) n FROM careproof_facility_service_gold
GROUP BY ALL ORDER BY service, evidence_category;
SELECT reason_code, severity, COUNT(*) n FROM careproof_facilities_quarantine GROUP BY ALL ORDER BY n DESC;
