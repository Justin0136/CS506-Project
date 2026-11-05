# ADS Cleaning

Cleaned and combined the ADS archive and current datasets to prepare the data for injury prediction.

- Filtered for reports where the automation system engaged was ADS.
- Kept the latest version of each Report ID.
- Removed duplicate crashes using Same Incident ID.
- Prioritized reports with known injury outcomes and more complete feature information.
- Removed one incident with contradictory injury labels.
- Standardized injury severity into binary labels (0 = no injury, 1 = injury).
- Combined weather and roadway-surface features across both datasets.

## Results

The final dataset contains 2,878 ADS crashes, with 2,597 no-injury crashes and 281 injury crashes (9.8%).

The cleaned data is saved in `data/processed/ads_model.csv`.

## Takeaways

One of the main challenges was handling duplicate reports, since multiple reports could refer to the same crash. I also had to account for differences between the archive and current datasets, especially in weather and roadway-surface fields.

The injury labels are imbalanced, with significantly fewer injury crashes. This will be important when evaluating our prediction models.

## Limitations / TODO

- Some features still contain missing or unknown values.
- Injury outcomes are self-reported and may not be completely accurate.
- Decide how to handle missing values during modeling.
- Test baseline models and account for class imbalance.

## File locations

Code: `scripts/clean_ads.py`  
Output: `data/processed/ads_model.csv`