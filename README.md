# What Makes Automated-Vehicle Crashes Severe?
### Predicting injury outcomes in ADS and Level 2 ADAS crashes from NHTSA Standing General Order data

**Team:** Emily Xu (emilyxu@bu.edu)  
**Course:** CS 506, Fall 2026

---

## 1. Project Description

Automated driving and its research are now expanding more than ever. Robotaxi fleets, such as Waymo and Zoox, now operate commercially in several U.S. cities, and driver-assistance systems like Tesla Autopilot, GM Super Cruise, and Ford BlueCruise are standard on millions of cars. Since 2021, NHTSA's **Standing General Order (SGO) 2021-01** has required manufacturers and operators to report crashes involving:

- **Automated Driving Systems (ADS):** SAE Levels 3–5, where the vehicle drives itself.
- **Level 2 Advanced Driver Assistance Systems (ADAS):** the system steers and controls speed, but a human must supervise.

NHTSA publishes these reports as public CSV files. They are one of the only real-world, multi-manufacturer datasets on how automated vehicles crash.

This project uses that data to study **what conditions are associated with more severe outcomes when an automated-vehicle crash occurs**. It also examines whether crash patterns differ between fully driverless systems and human-supervised driver assistance.

**An important framing note.** The SGO data contains *only crashes*. It does not include miles driven or trips taken, so it cannot tell us whether one company or system is "safer" than another, or than human drivers. All questions in this project are therefore **conditional on a crash having occurred**. This limitation will be stated explicitly throughout the analysis.

## 2. Goals

### Primary goal
**Predict whether a reported ADS or Level 2 ADAS crash resulted in an injury, using only pre-crash and environmental conditions.**

- **Target:** binary. The label is 1 if the report's highest alleged injury severity is minor, moderate, serious, or fatal, and 0 if there was no injury or only property damage.
- **Inputs:** conditions known before or at the moment of the crash:
  - roadway type and the reported weather conditions;
  - the vehicle's pre-crash movement and speed (posted speed limit is only in the 2021–2025 files, so it's used in a secondary analysis);
  - what the vehicle collided with (vehicle, pedestrian, cyclist, fixed object);
  - automation type (ADS vs. Level 2);
  - monthly weather context for the crash location and month (see Section 3).
- **Success criteria:**
  - Beat a **majority-class baseline** and a **logistic regression baseline** on a held-out test set.
  - Report **macro-F1** and **precision-recall AUC** (PR-AUC). These suit an imbalanced label; accuracy alone would be misleading.
  - Target: at least 0.10 PR-AUC improvement over the majority baseline.

### Secondary goal
**Determine whether crash conditions differ between ADS (driverless) and Level 2 ADAS (human-supervised) vehicles.**

- **Measurable version:** train a classifier to predict ADS vs. Level 2 from crash conditions alone.
  - An ROC-AUC well above 0.5 means the two systems crash under meaningfully different circumstances.
  - Feature importances show *which* circumstances differ (e.g., urban low-speed intersections vs. highway speeds).
- This is supported by descriptive statistics and chi-square tests on key categorical features.

### Fallback goal (if the primary goal proves too ambitious)
Produce a descriptive comparison of ADS vs. Level 2 crash patterns, with a simple interpretable classifier (logistic regression or a shallow decision tree) for injury vs. no injury.

## 3. Data Collection Plan

| Source | What it provides | How it's collected |
|---|---|---|
| **NHTSA SGO 2021-01 incident reports** ([nhtsa.gov](https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting)) | One row per incident report: date, city/state, manufacturer/operator, automation type, roadway, weather, pre-crash movement and speed, crash partner, injury severity, narrative | A Python script downloads the ADS and Level 2 ADAS CSV files, covering both the current release and the archived pre-June-16-2025 release, directly from NHTSA's website |
| **Open-Meteo Historical Weather API** ([open-meteo.com](https://open-meteo.com/)) | Historical precipitation, temperature and wind by location | NHTSA releases crash dates as month and year only (the day is withheld as PII), so a same-day weather join isn't possible. Instead, use each crash's reported location and compute **monthly weather context** for the crash month (e.g., rainy days, mean temperature). This complements the weather conditions recorded in the report itself. Free, no API key required |
| **NHTSA SGO data element definitions (PDF)** | Field definitions, schema change log, known data limitations | Used as documentation to guide cleaning |

**Collection will be fully scripted.** `make data` will download and cache all raw files, so the dataset can be rebuilt from scratch with one command. NHTSA updates the SGO data monthly. The project will **freeze a snapshot** at a fixed release date so results are reproducible.

**Approximate scale:** several thousand incident reports across ADS and Level 2 ADAS from July 2021 to the present, before deduplication.

## 4. Data Cleaning Plan (preliminary)

The SGO data is messy, and cleaning it well is a core part of the project:

1. **Schema reconciliation.** The SGO was amended in 2025, and NHTSA publishes data before and after June 16, 2025 in separate files with different columns. Columns will be mapped to a common schema:
   - The old injury categories vs. the new, more detailed ones that split by hospitalization. Both are mapped to the same injury / no-injury label.
   - The old and new weather and road-surface columns.
   - Fields that exist in only one schema (e.g., `Posted Speed Limit (MPH)`) are limited to a secondary analysis. Lighting is out of scope because the post-June-2025 files don't include it.
2. **Deduplication.** A single crash can generate several reports: the same entity may submit updated versions, and different entities (e.g., a manufacturer and an operator) may each report it. Keep only the latest `Report Version` of each `Report ID`, then collapse reports sharing NHTSA's `Same Incident ID` into one row per crash.
3. **Redactions and missing values.** Some fields are redacted as confidential business information, and others are marked "Unknown." These become explicit missing categories rather than being silently dropped, and the analysis will check whether missingness itself correlates with the target.
4. **Automation-type validation.** NHTSA notes that reporters have sometimes mislabeled ADS vs. Level 2 ADAS. Labels will be cross-checked against make/model and reporting entity.
5. **Leakage prevention.** Post-crash fields (e.g., airbag deployment, towing) and the free-text narrative are excluded from the primary model's inputs because they encode the outcome.

## 5. Modeling Plan (preliminary)

- **Baselines:** majority-class classifier and logistic regression on one-hot encoded features.
- **Main models:** random forest and gradient-boosted trees (XGBoost). Both handle mixed categorical and numeric data and non-linear interactions (e.g., speed × crash partner).
- **Class imbalance:** class weighting, with a comparison against resampling (e.g., SMOTE) if time permits.
- **Interpretability:** permutation importance and SHAP values, to identify which conditions drive injury predictions.
- **Stretch goal:** TF-IDF features from the crash narrative, where not redacted, as a secondary model. This will be kept clearly separate from the leakage-free primary model.

## 6. Visualization Plan (preliminary)

- **Interactive U.S. map of crashes** (Plotly), colored by automation type and sized by injury severity, with filters by year and manufacturer.
- **Crash-condition comparisons:** grouped bar charts of pre-crash movement, roadway type, and crash partner for ADS vs. Level 2.
- **Injury rate by condition:** e.g., by road type, weather, speed bin, and crash partner, with confidence intervals.
- **Model performance:** precision-recall curves for all models against baselines, a confusion matrix, and SHAP summary plots.
- **Crash volume over time:** monthly report counts by automation type. Annotations will note that growth in reports reflects fleet growth and rule changes, not necessarily declining safety.

## 7. Test Plan

- **Temporal split (primary):** train on crashes reported before a cutoff date and test on crashes after it (e.g., train through 2025, test on 2026). This mimics the real use case of predicting outcomes of *future* crashes. It also tests whether the model survives shifts in fleets and reporting rules.
- **Stratified 5-fold cross-validation** on the training set for model selection and hyperparameter tuning.
- **Grouping by crash:** reports of the same crash never appear in both train and test.
- **Metrics:** macro-F1, PR-AUC and ROC-AUC, all compared against the baselines in Section 2. Plain accuracy is not used as a headline metric: because most crashes have no injury, a model that always predicts "no injury" would score high accuracy while learning nothing.
  - **Macro-F1:** F1 combines *precision* (of crashes flagged as injury, how many truly were) and *recall* (of true injury crashes, how many were caught) into one score from 0 to 1. "Macro" computes F1 separately for the injury and no-injury classes and averages them equally, so the rarer injury class counts as much as the common one. This measures performance at the chosen decision threshold.
  - **PR-AUC (precision-recall AUC):** the area under the precision-recall curve, traced across all decision thresholds. It focuses on how well the model finds the rare (injury) class. A random model scores roughly the share of injury crashes in the data, which is the baseline this project aims to beat by at least 0.10. This is the headline metric for the injury model.
  - **ROC-AUC:** the area under the ROC curve (true-positive rate vs. false-positive rate across all thresholds). It equals the probability that the model ranks a randomly chosen positive case above a randomly chosen negative one: 0.5 is random guessing, and 1.0 is perfect. This is the main metric for the secondary goal (ADS vs. Level 2 classification), where the classes are more balanced.

## 8. Timeline (~10 weeks)

| Week(s) | Dates (approx.) | Tasks |
|---|---|---|
| 1 | Sep 28 – Oct 4 | Repo setup, Makefile skeleton, SGO download script, data dictionary review |
| 2 | Oct 5 – Oct 11 | Monthly weather-context script; freeze data snapshot |
| 3–4 | Oct 12 – Oct 25 | Cleaning: schema merge, deduplication, missing/redacted handling. Exploratory visualizations. **October check-in** |
| 5–6 | Oct 26 – Nov 8 | Feature engineering; baseline and logistic regression; first tree models |
| 7 | Nov 9 – Nov 15 | Tuning, temporal-split evaluation, SHAP analysis. **November check-in** |
| 8 | Nov 16 – Nov 22 | Secondary goal (ADS vs. Level 2 classifier); interactive map and final plots |
| 9 | Nov 23 – Nov 29 | Tests, GitHub Actions workflow, README final report *(Thanksgiving week — lighter load)* |
| 10 | Nov 30 – Dec 9 | Record and upload 10-minute presentation; final polish. **Due Dec 9** |

## 9. Known Limitations

- **No exposure data.** Crash counts cannot be converted into crash *rates*, so no conclusions will be drawn about which system or company is safest overall.
- **Reporting bias.** Reporting thresholds differ between ADS and Level 2 ADAS.
  - ADS crashes are reportable even for property damage or a tow-away.
  - Level 2 crashes are reportable only if they involve a vulnerable road user, a fatality, an airbag deployment, or a hospital transport.
  - As a result, Level 2 crashes skew severe *by construction*. The injury model will be evaluated separately for each system type, and the ADS vs. Level 2 comparison will be interpreted with this in mind.
  - Manufacturers also learn about crashes in different ways (live telematics vs. customer complaints), so the two populations are not directly comparable.
- **Self-reported and unverified data.** Injury severity is *alleged* by the reporting entity and may be incomplete or later revised.
- **Redactions.** Some fields are withheld as confidential business information, which limits the available features for some manufacturers.

---

## AI Use Disclosure

This project used Claude (Anthropic) as an assistant during the proposal stage for:
- editing the proposal README, including making sure all sections align with the project requirements;
- reviewing the NHTSA data dictionaries to identify relevant fields and schema differences.

All AI-assisted content was checked against primary sources (NHTSA documentation and data files).

---

*Build, run, and test instructions will be added here as the codebase is developed. The final version of this README will serve as the project report.*