import pandas as pd
from pathlib import Path

# Locate the raw data folder
data_folder = Path(__file__).resolve().parent.parent.parent / "CS506-Project" / "data" / "raw"

# Load the ADAS datasets
# ADAS_archive contains an invalid UTF-8 byte sequence, so we use encoding_errors="replace" to handle it
# but also the invalid byte sequence is in the column "city" which doesn't matter at all, mostly just a loading issue.
adas_archive = pd.read_csv(
    data_folder / "ADAS_archive.csv",
    low_memory=False,
    encoding="utf-8",
    encoding_errors="replace"
)
adas_current = pd.read_csv(data_folder / "ADAS_current.csv", low_memory=False)

adas_archive = adas_archive[adas_archive["Automation System Engaged?"] == "ADAS"].copy()
adas_current = adas_current[adas_current["Automation System Engaged?"] == "ADAS"].copy()

print("\nArchive engagement columns:")
print([col for col in adas_archive.columns if "Engaged" in col])

print("\nCurrent engagement columns:")
print([col for col in adas_current.columns if "Engaged" in col])

# Find road surface-related columns
print("\nADAS Archive surface columns:")
print([col for col in adas_archive.columns if "Surface" in col])

print("\nADAS Current surface columns:")
print([col for col in adas_current.columns if "Surface" in col])

# Inspect road surface values in both datasets
print("\nADAS Archive Roadway Surface:")
print(adas_archive["Roadway Surface"].value_counts(dropna=False))

print("\nADAS Current Wet Surface Condition:")
print(adas_current["Roadway-Wet Surface Condition"].value_counts(dropna=False))

# Standardize wet road surface across both datasets
adas_archive["Wet Surface"] = adas_archive["Roadway Surface"].map({
    "Wet": 1,
    "Dry": 0,
    "Snow / Slush / Ice": 0
})

adas_current["Wet Surface"] = (
    adas_current["Roadway-Wet Surface Condition"] == "Y"
).astype(int)

# Keep the latest version of each report
archive_latest = (
    adas_archive
    .sort_values("Report Version")
    .drop_duplicates(subset="Report ID", keep="last")
)

current_latest = (
    adas_current
    .sort_values("Report Version")
    .drop_duplicates(subset="Report ID", keep="last")
)

# Check the results
print("\nDropping duplicates report versions: ")
print("ADAS Archive:", len(adas_archive), "->", len(archive_latest))
print("ADAS Current:", len(adas_current), "->", len(current_latest))

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
    "Serious W/O Hospitalization": 1,
    "Fatality": 1
}

# Apply the same mapping to both datasets
for df in [archive_latest, current_latest]:
    df["Injury"] = df["Highest Injury Severity Alleged"].map(injury_map)

# Check the injury distribution after report deduplication
print("\nADAS Archive Injury:")
print(archive_latest["Injury"].value_counts(dropna=False))

print("\nADAS Current Injury:")
print(current_latest["Injury"].value_counts(dropna=False))

# We were double counting " " as injury IDs, so need to preserve them here
# They are usually missing some data, but can perhaps be useful for comparisons since they have valid injury labels
for df in [archive_latest, current_latest]:
    df["Same Incident ID"] = (
        df["Same Incident ID"]
        .replace(r"^\s*$", pd.NA, regex=True)
    )

# Find incidents with both injury (1) and no injury (0) reported
for name, df in [
    ("ADAS Archive", archive_latest),
    ("ADAS Current", current_latest)
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

print("\nADAS Archive after removing contradictory incidents:")
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

print("\nADAS Archive after incident deduplication:", len(archive_clean))
print("ADAS Current after incident deduplication:", len(current_clean))

# Combine the cleaned ADAS datasets
adas_clean = pd.concat(
    [archive_clean, current_clean],
    ignore_index=True
)

# Count how many of the selected features are populated
features = [
    "Crash With",
    "Roadway Type",
    "SV Precrash Speed (MPH)",
    "SV Pre-Crash Movement"
]

adas_clean["completeness"] = (
    adas_clean[features]
    .replace(["Unknown", ""], pd.NA)
    .notna()
    .sum(axis=1)
)

# Keep the row with the most complete data for each Report ID
adas_clean = (
    adas_clean
    .sort_values("completeness")
    .drop_duplicates(subset="Report ID", keep="last")
    .drop(columns="completeness")
    .reset_index(drop=True)
)

# Check the combined dataset
print("\nCombined ADAS Dataset:")
print("Total rows:", len(adas_clean))
print("Duplicate Report IDs:",
      adas_clean["Report ID"].duplicated().sum())


print("\nInjury distribution:")
print(adas_clean["Injury"].value_counts(dropna=False))


# Keep only reports with known injury outcomes
adas_model = adas_clean.dropna(subset=["Injury"]).copy()

# Convert injury labels from floats to integers
adas_model["Injury"] = adas_model["Injury"].astype(int)

# Verify the results
print("\nADAS Modeling Dataset:")
print("Total rows:", len(adas_model))
print(adas_model["Injury"].value_counts())

# Features selected for initial modeling
features = [
    "Crash With",
    "Roadway Type",
    "SV Precrash Speed (MPH)",
    "SV Pre-Crash Movement"
]

# Check missing values for each feature
print("\nMissing Feature Values:")
print(adas_model[features].isna().sum())

# Check the data types
print("\nFeature Data Types:")
print(adas_model[features].dtypes)

# Check common values for categorical features
for feature in features:
    print(f"\n{feature}:")
    print(adas_model[feature].value_counts(dropna=False).head(10))

# Find weather columns shared by both datasets
archive_weather = {
    col for col in adas_archive.columns
    if col.startswith("Weather")
}

current_weather = {
    col for col in adas_current.columns
    if col.startswith("Weather")
}

shared_weather = sorted(archive_weather & current_weather)

print("\nShared Weather Features:")
for col in shared_weather:
    print(col)

# Inspect values and missing data in shared weather columns
for col in shared_weather:
    print(f"\n{col}:")
    print(adas_model[col].value_counts(dropna=False))

# Count selected weather conditions per report
weather_count = adas_model[shared_weather].eq("Y").sum(axis=1)

print("\nNumber of weather conditions selected:")
print(weather_count.value_counts().sort_index())

# Find reports without any of the five shared weather conditions
no_shared_weather = adas_model[weather_count == 0]

# Find all weather-related columns in the combined dataset
all_weather = [
    col for col in adas_model.columns
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
    adas_model[col] = adas_model[col].eq("Y").astype(int)

# Combine cloudy and partly cloudy conditions
adas_model["Weather - Cloudy"] = (
    (adas_model["Weather - Cloudy"] == 1)
    | (adas_model["Weather - Partly Cloudy"] == "Y")
).astype(int)

# Combine fog/smoke categories
adas_model["Weather - Fog/Smoke/Haze Combined"] = (
    (adas_model["Weather - Fog/Smoke"] == "Y")
    | (adas_model["Weather - Fog/Smoke/Haze"] == "Y")
).astype(int)

print("\nCombined weather totals:")
print(adas_model[[
    "Weather - Cloudy",
    "Weather - Fog/Smoke/Haze Combined"
]].sum())

# Verify the conversion
print("\nWeather features after encoding:")
print(adas_model[shared_weather].head())

print("\nWeather indicator totals:")
print(adas_model[shared_weather].sum())

# Check unknown values in categorical features
categorical_features = [
    "Crash With",
    "Roadway Type",
    "SV Pre-Crash Movement"
]

for col in categorical_features:
    print(f"\n{col}:")
    print(
        adas_model[col]
        .value_counts(dropna=False)
        .loc[lambda counts: counts.index.astype(str).str.contains(
            "Unknown|Unk", case=False, regex=True
        )]
    )

# Select features for the initial model
model_features = features + shared_weather + [
    "Weather - Fog/Smoke/Haze Combined",
    "Wet Surface"
]

# Keep identifiers and date for tracking and future train/test splitting
metadata = ["Incident Date", "Report ID", "Same Incident ID"]

print("\nWet Surface distribution:")
print(adas_model["Wet Surface"].value_counts(dropna=False))

# Create the final modeling DataFrame
adas_final = adas_model[metadata + model_features + ["Injury"]].copy()

# Verify its dimensions and columns
print("\nFinal ADAS Modeling Dataset:")
print("Shape:", adas_final.shape)
print("\nColumns:")
print(adas_final.columns.tolist())

print("\nMissing values:")
print(adas_final.isna().sum())

# Create the processed data folder
output_folder = data_folder.parent.parent.parent /"CS506-Project" / "data" / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

# Save the modeling dataset
output_file = output_folder / "level2_model.csv"
adas_final.to_csv(output_file, index=False)

print("\nSaved cleaned ADAS dataset to:", output_file)
