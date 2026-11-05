# Model Input Prep

Prepared the cleaned ADS dataset for training and testing the injury prediction models.

- Added the incident year from the incident date.
- Joined model year back from the raw ADS datasets using Report ID, keeping the latest version of each report.
- Added vehicle age (incident year minus model year), setting negative ages to 0.
- Split the data by year: train on 2021–2024, validate on 2025, and test on 2026.
- Checked that no crash appears in more than one split.
- Built a shared encoding step for the models, with an option to leave out Crash With.

## Results

The training set contains 1,017 crashes (101 injury, 9.9%), the validation set contains 1,011 crashes (101 injury, 10.0%), and the test set contains 850 crashes (79 injury, 9.3%). Model year was found for all but 6 crashes.

A quick logistic regression scored a test PR-AUC of 0.18 with Crash With and 0.12 without it, compared to 0.09 for random guessing.

The splits are saved in `data/processed/model_input/`.

## Takeaways

One of the main decisions was splitting by year instead of randomly, so the models are tested on crashes that happened after the ones they learned from. I left incident year out of the model features, since every 2026 test crash would have a year the model never saw during training.

The training set is small because most ADS crashes happened in 2025 and 2026. With only 101 injury crashes to learn from, model scores will be noisy.

Removing Crash With lowered the score noticeably, so what the vehicle hit carries a lot of the signal for predicting injuries.

## Limitations / TODO

- 2026 data only goes through August, so the test set is a partial year.
- 21 vehicles had a model year newer than the crash date, so their vehicle age was set to 0.
- Vehicle age is low across the dataset (median 1 year), so it may not be a strong predictor.
- Decide whether to retrain on 2021–2025 before the final 2026 test.

## File locations

Code: `scripts/prepare_ads_model.py`, `scripts/model_features.py`  
Output: `data/processed/model_input/`