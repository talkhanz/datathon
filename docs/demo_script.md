# Three-minute demo script (CareProof Referral Navigator app)

**Before recording:** notebooks 00–03 have been run and the Databricks App is deployed. Open the app, search once
to warm the cache, and keep notebook 05's summary open in another tab.

| Time | Screen | Say |
|---|---|---|
| 0:00–0:20 | App header + safety banner | "An ASHA worker in Bihar needs dialysis for a patient. The data is about 10,000 facility listings that an AI extracted from the web. Some are merged, some contradict themselves. **The nearest claimed provider is not necessarily the nearest credibly supported provider.**" |
| 0:20–0:55 | **Find**: dialysis, PIN 800001, 25 km | "Of the records that mention dialysis here, only those passing every credibility gate are recommended. Each card quotes the facility's **own words**, gives an evidence category ('Strong documented evidence', not 'offers dialysis'), shows identity and location checks, and says **current availability is not verified**." Open the component-score expander. |
| 0:55–1:30 | Coordinates 11.938235, 79.485443, 5 km | "Here a keyword search would send you to an *eye hospital* claiming a dialysis unit. Its record mixes Hyderabad, a Delhi PIN and Puducherry coordinates. The other nearby record's dialysis sentence is about a different hospital. CareProof **abstains**: 'We found records mentioning this service, but none had sufficiently consistent identity, location, and supporting evidence.'" Open the planner/audit expander to show the failed gates. |
| 1:30–2:00 | **Compare** tab (cardiac surgery, PIN 380001) | "The nearest keyword match is 1.3 km away, but its coordinates disagree with its own PIN by more than 25 km. CareProof's top candidate is a cardiology institute with a corroborated description. On 56 random queries, **a third of the keyword baseline's recommendations would fail our evidence gates.**" |
| 2:00–2:35 | **Verify** tab | Tick two checklist items and set the outcome to *Unreachable*. "The next eligible alternative is promoted immediately. Statuses stay in this session and never overwrite the data." |
| 2:35–2:50 | **Refer** tab | "A print-friendly summary with the quote, caveats, checklist, alternatives and a timestamp, and no patient data." |
| 2:50–3:00 | Notebook 05 / evaluation protocol | "Every citation is an exact quote, abstention is a measured safety behaviour, and precision is honestly marked pending human review, with a 142-pair blind review set ready to fill." |
