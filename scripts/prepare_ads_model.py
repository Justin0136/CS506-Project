import pandas as pd
from pathlib import Path

# locate data folders
data_folder = Path(__file__).resolve().parent.parent / "data"
output_folder = data_folder / "processed" / "model_input"
output_folder.mkdir(parents=True, exist_ok=True)

# load cleaned ADS dataset
ads = pd.read_csv(data_folder / "processed" / "ads_model.csv")

print("ADS crashes:", len(ads))

# add year each crash happened
ads["Incident Year"] = pd.to_datetime(ads["Incident Date"].str.title(), format="%b-%Y").dt.year

# load model year from raw ADS
raw = pd.concat([
    pd.read_csv(data_folder / "raw" / "ADS_archive.csv", low_memory=False),
    pd.read_csv(data_folder / "raw" / "ADS_current.csv", low_memory=False)
])

# keep only latest version of each report
model_years = (
    raw.sort_values("Report Version").drop_duplicates(subset="Report ID", keep="last") [["Report ID", "Model Year"]]
)

# join model year using report ID
ads = ads.merge(model_years, on="Report ID", how="left")

print("\nMissing model years:", ads["Model Year"].isna().sum())

# vehicle age at time of crash (if model year is newer set to 0)
ads["Vehicle Age"] = (ads["Incident Year"] - ads["Model Year"]).clip(lower=0)

print("\nVehicle Age:")
print(ads["Vehicle Age"].describe())

# split by year: train 2021-2024, validate 2025, test 2026
train = ads[ads["Incident Year"] <= 2024]
validate = ads[ads["Incident Year"] == 2025]
test = ads[ads["Incident Year"] == 2026]

# check size and injury rate for split
print("\nCrashes per year:")
print(pd.crosstab(ads["Incident Year"], ads["Injury"]))
 
for name, df in [("Train", train), ("Val", validate), ("Test", test)]:
    print(f"\n{name}: {len(df)} crashes, {df['Injury'].sum()} injuries, "
          f"{df['Injury'].mean():.1%} injury rate")

# make sure no crashes appear in multiple splits
overlap = (
    set(train["Report ID"]) & set(validate["Report ID"])
    | set(train["Report ID"]) & set(test["Report ID"])
    | set(validate["Report ID"]) & set(test["Report ID"])
)
 
print("\nCrashes in more than one split:", len(overlap))

# save splits
train.to_csv(output_folder / "train.csv", index=False)
validate.to_csv(output_folder / "validate.csv", index=False)
test.to_csv(output_folder / "test.csv", index=False)

print("\nSaved train, val, and test splits to:", output_folder)

