# Referral workflow (CareProof Referral Navigator app)

1. **Find.** Choose a service and a location (PIN, district, city or coordinates), a radius and the number of
   results. CareProof shows how many records *mention* the service, how many *passed every credibility gate* and how
   many were *held back*. Each eligible card shows: rank, name, location, approximate straight-line distance,
   evidence category, current-verification status, explanation (what is supported, inferred, unknown), the exact
   excerpt with matched field and terms, source URL, identity / location / missing-information warnings, component
   scores (expander) and a *Select for verification* button. Held-back records appear only in a planner/audit
   expander with their failed gates.
2. **Compare.** The nearest keyword match (distance only) next to CareProof's top eligible candidate, with the
   nearest keyword match's evidence category and failed gates, and a short "why they differ".
3. **Verify.** A service-specific call-ahead checklist (e.g. C-section: obstetrician *and* anaesthetist, OT, blood,
   newborn resuscitation) and a call outcome: Not contacted · Confirmed · Partially confirmed · Unreachable ·
   Service contradicted · Not accepting referrals. If the outcome is Unreachable, Contradicted or Not accepting, the
   **next eligible alternative** is shown prominently with a one-click switch. Statuses are session-only and never
   change facility data.
4. **Refer.** A print-friendly plain-text summary to download: service, origin, selected facility, supporting
   excerpt, caveats, checklist state, verification status, alternatives, timestamp, safety warning. It has no
   patient fields.

**Abstention:** "We found records mentioning this service, but none had sufficiently consistent identity, location,
and supporting evidence for a reliable recommendation." Or, if nothing mentions it, a message explaining that
absence from the dataset is not absence of the service.
