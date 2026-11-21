# Safety and limitations

**Always displayed:** "CareProof is a referral planning and facility-discovery tool. Public facility information may
be incomplete or outdated. Contact the receiving facility to confirm service and admission availability before
travel. CareProof does not provide medical advice or emergency dispatch."

## CareProof does not claim
Guaranteed availability · verified clinical quality · current bed availability · current specialist availability ·
patient-specific suitability · travel time · improved patient outcomes (that would need a prospective evaluation).

## Design safeguards
* Hard eligibility gates before ranking; conflicting or insufficient evidence is never recommended.
* Explicit abstention instead of weakened thresholds.
* Exact quotations only. No LLM paraphrase is ever presented as evidence.
* Passages about other facilities, referrals, tie-ups, planned or historical services are not counted as evidence.
* Verification statuses are session-only feedback and never overwrite facility data.
* No patient-identifiable information is collected. The notes field warns against entering identifiers, and the
  printed summary has no patient fields.
* Marketplace tables are read-only. Derived tables live in a configured `careproof_` schema.

## Known limitations
* Source data is LLM-extracted from public web pages, mostly social media and aggregators. It may be stale, wrong
  or describe the wrong facility.
* Rural under-coverage: in the most rural quartile of districts (branch-office proxy), 114 of 189 have no facility
  in the dataset. Abstention is common there.
* Descriptions are essentially English-only, so facilities described in Indian languages may be missed.
* Straight-line distance understates travel in hilly or poorly connected areas.
* Merged-record detection is rule-based. Look-alike branches of the same chain can slip through, and fuzzy
  matching can occasionally flag spelling variants.
* No metric of real-world correctness exists yet. Precision and quarantine accuracy await human review.
* NFHS indicators are district population statistics, shown only as context. They never describe a patient and
  never affect ranking.
