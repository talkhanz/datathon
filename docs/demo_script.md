# Three-minute demo script

**Setup before recording:** notebooks 00–03 have been run. Open `04_recommendation_demo` with the widgets visible,
having already run it once so the caches are warm.

| Time | Screen | Say |
|---|---|---|
| 0:00–0:20 | Title cell | "A health worker in rural Bihar needs dialysis for a patient. Where should they go, and why should they trust the answer? This data is about 10,000 facility listings scraped from the web. Some records merge two hospitals, some contradict themselves. Our core idea: **the nearest *claimed* provider is not the nearest *credibly supported* provider.**" |
| 0:20–0:55 | Scenario **A** cards (dialysis, PIN 800001) | "A non-technical user picks a service and a PIN in these widgets. Each card quotes the facility's **own words**. For example: 'first standalone dialysis centre in Bihar… offering 24x7 dialysis'. It shows which fields say so, the straight-line distance, and a score you can audit: service, evidence, location, distance, readiness, minus penalties. High confidence here means several independent fields agree." |
| 0:55–1:35 | Scenario **B** cards and baseline table | "Now a trap. This record is called *Aravind Eye Hospital*. It claims a 20-bed dialysis unit. Its city says Hyderabad, its PIN is in Delhi, and its coordinates are in Puducherry. A keyword search ranks it **#1 at 0 km**. CareProof still shows the quote, but labels it **Low confidence**, subtracts a penalty, and spells out each contradiction." |
| 1:35–2:10 | Scenario **C** (cardiac surgery, Ahmedabad) | "Against the 'nearest keyword match' baseline: the closest hit is 1.3 km away, but its coordinates are more than 25 km from its own PIN code, and only one line mentions a valve procedure. CareProof's #1 is a dedicated cardiology institute with corroborated evidence. In our benchmark, the baseline's top pick was weak or contradictory in 9 of 18 answerable queries." |
| 2:10–2:30 | Scenario **D** (burn care, Mumbai) | "And when the evidence isn't there, we say so. The nearby 'burn' keyword hits are skin and cosmetic clinics. CareProof returns *no credibly supported provider within 25 km* instead of inventing one." |
| 2:30–2:50 | Notebook 05 summary | "Evaluation: every cited excerpt is an exact substring of the source (77 of 77 locally), every result is geographically valid, and the ranking is stable under alternative weights. Precision@K is honestly marked pending human review, and we ship the labeling template." |
| 2:50–3:00 | Verification banner | "Every result carries this warning: availability is not verified, call before you travel, and in an emergency call 108. Thank you." |

**Fallback:** if a live query errors, scroll to the pre-run scenario cells. They are deterministic.
**Numbers:** before recording, replace the local numbers above with the Databricks run of notebook 05.
