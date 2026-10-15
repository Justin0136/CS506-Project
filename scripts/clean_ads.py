import pandas as pd
from pathlib import Path

# Locate the raw data folder
data_folder = Path(__file__).resolve().parent.parent / "data" / "raw"

# Load the ADS datasets
ads_archive = pd.read_csv(data_folder / "ADS_archive.csv", low_memory=False)
ads_current = pd.read_csv(data_folder / "ADS_current.csv", low_memory=False)

# Keep the latest version of each report
archive_latest = (
    ads_archive
    .sort_values("Report Version")
    .drop_duplicates(subset="Report ID", keep="last")
)

current_latest = (
    ads_current
    .sort_values("Report Version")
    .drop_duplicates(subset="Report ID", keep="last")
)

# Check the results
print("ADS Archive:", len(ads_archive), "->", len(archive_latest))
print("ADS Current:", len(ads_current), "->", len(current_latest))

# Standardize injury severity into a binary label
injury_map = {
    "No Injuries Reported": 0,
    "No Injured Reported": 0,
    "Property Damage. No Injured Reported": 0,
    "Minor": 1,
    "Minor W/O Hospitalization": 1,
    "Minor W/ Hospitalization": 1,
    "Moderate": 1,
    "Moderate W/O Hospitalization": 1,
    "Moderate W/ Hospitalization": 1,
    "Serious": 1,
    "Serious W/ Hospitalization": 1,
    "Fatality": 1
}

# Apply the same mapping to both datasets
for df in [archive_latest, current_latest]:
    df["Injury"] = df["Highest Injury Severity Alleged"].map(injury_map)

# Check the injury distribution after report deduplication
print("\nADS Archive Injury:")
print(archive_latest["Injury"].value_counts(dropna=False))

print("\nADS Current Injury:")
print(current_latest["Injury"].value_counts(dropna=False))

# Find incidents with both injury (1) and no injury (0) reported
for name, df in [
    ("ADS Archive", archive_latest),
    ("ADS Current", current_latest)
]:
    conflicts = (
        df.dropna(subset=["Same Incident ID", "Injury"])
        .groupby("Same Incident ID")["Injury"]
        .nunique()
    )

    conflicting_ids = conflicts[conflicts > 1].index

    print(f"\n{name} Contradictory Injury Incidents:")
    print("Count:", len(conflicting_ids))
    print("Incident IDs:", conflicting_ids.tolist())

# Remove incidents with contradictory known injury labels
conflicting_ids = (
    archive_latest
    .dropna(subset=["Same Incident ID", "Injury"])
    .groupby("Same Incident ID")["Injury"]
    .nunique()
)

conflicting_ids = conflicting_ids[conflicting_ids > 1].index

archive_latest = archive_latest[
    ~archive_latest["Same Incident ID"].isin(conflicting_ids)
].copy()

print("\nADS Archive after removing contradictory incidents:")
print("Remaining reports:", len(archive_latest))

# Features used to determine report completeness
features = [
    "Crash With",
    "Roadway Type",
    "SV Precrash Speed (MPH)",
    "SV Pre-Crash Movement"
]

for df in [archive_latest, current_latest]:
    # Give priority to reports with known injury outcomes
    df["known_injury"] = df["Injury"].notna().astype(int)

    # Count available feature values
    df["completeness"] = (
        df[features]
        .replace(["Unknown", ""], pd.NA)
        .notna()
        .sum(axis=1)
    )

# Sort reports by our selection priorities
archive_latest = archive_latest.sort_values(
    ["known_injury", "completeness", "Report Version"]
)

current_latest = current_latest.sort_values(
    ["known_injury", "completeness", "Report Version"]
)

# Keep one report per valid incident ID
# Preserve reports with missing incident IDs
for_name = []

for df in [archive_latest, current_latest]:
    valid_id = (
        df["Same Incident ID"].notna()
        & df["Same Incident ID"].astype(str).str.strip().ne("")
    )

    duplicate = df["Same Incident ID"].duplicated(keep="last")

    for_name.append(df[~(valid_id & duplicate)].copy())

archive_clean, current_clean = for_name

print("\nADS Archive after incident deduplication:", len(archive_clean))
print("ADS Current after incident deduplication:", len(current_clean))

# Combine the cleaned ADS datasets
ads_clean = pd.concat(
    [archive_clean, current_clean],
    ignore_index=True
)

# Check the combined dataset
print("\nCombined ADS Dataset:")
print("Total rows:", len(ads_clean))
print("\nInjury distribution:")
print(ads_clean["Injury"].value_counts(dropna=False))

# Keep only reports with known injury outcomes
ads_model = ads_clean.dropna(subset=["Injury"]).copy()

# Convert injury labels from floats to integers
ads_model["Injury"] = ads_model["Injury"].astype(int)

# Verify the results
print("\nADS Modeling Dataset:")
print("Total rows:", len(ads_model))
print(ads_model["Injury"].value_counts())

# Features selected for initial modeling
features = [
    "Crash With",
    "Roadway Type",
    "SV Precrash Speed (MPH)",
    "SV Pre-Crash Movement"
]

# Check missing values for each feature
print("\nMissing Feature Values:")
print(ads_model[features].isna().sum())

# Check the data types
print("\nFeature Data Types:")
print(ads_model[features].dtypes)

# Check common values for categorical features
for feature in features:
    print(f"\n{feature}:")
    print(ads_model[feature].value_counts(dropna=False).head(10))

# Find weather columns shared by both datasets
archive_weather = {
    col for col in ads_archive.columns
    if col.startswith("Weather")
}

current_weather = {
    col for col in ads_current.columns
    if col.startswith("Weather")
}

shared_weather = sorted(archive_weather & current_weather)

print("\nShared Weather Features:")
for col in shared_weather:
    print(col)

# Inspect values and missing data in shared weather columns
for col in shared_weather:
    print(f"\n{col}:")
    print(ads_model[col].value_counts(dropna=False))

# Count selected weather conditions per report
weather_count = ads_model[shared_weather].eq("Y").sum(axis=1)

print("\nNumber of weather conditions selected:")
print(weather_count.value_counts().sort_index())

# Find reports without any of the five shared weather conditions
no_shared_weather = ads_model[weather_count == 0]

# Find all weather-related columns in the combined dataset
all_weather = [
    col for col in ads_model.columns
    if col.startswith("Weather")
]

# Check values in the other weather columns
other_weather = [
    col for col in all_weather
    if col not in shared_weather
]

print("\nOther weather columns:")
print(other_weather)

for col in other_weather:
    print(f"\n{col}:")
    print(no_shared_weather[col].value_counts(dropna=False))

# Convert shared weather indicators to binary values
for col in shared_weather:
    ads_model[col] = ads_model[col].eq("Y").astype(int)

# Verify the conversion
print("\nWeather features after encoding:")
print(ads_model[shared_weather].head())

print("\nWeather indicator totals:")
print(ads_model[shared_weather].sum())

# Check unknown values in categorical features
categorical_features = [
    "Crash With",
    "Roadway Type",
    "SV Pre-Crash Movement"
]

for col in categorical_features:
    print(f"\n{col}:")
    print(
        ads_model[col]
        .value_counts(dropna=False)
        .loc[lambda counts: counts.index.astype(str).str.contains(
            "Unknown|Unk", case=False, regex=True
        )]
    )

# Select features for the initial model
model_features = features + shared_weather

# Create the final modeling DataFrame
ads_final = ads_model[model_features + ["Injury"]].copy()

# Verify its dimensions and columns
print("\nFinal ADS Modeling Dataset:")
print("Shape:", ads_final.shape)
print("\nColumns:")
print(ads_final.columns.tolist())

print("\nMissing values:")
print(ads_final.isna().sum())

# Create the processed data folder
output_folder = data_folder.parent / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

# Save the modeling dataset
output_file = output_folder / "ads_model.csv"
ads_final.to_csv(output_file, index=False)

print("\nSaved cleaned ADS dataset to:", output_file)
