import pandas as pd
from pathlib import Path

# locate data folder
proj_root = Path(__file__).resolve().parent.parent
data_folder = proj_root / "data" / "raw"

# load ADS datasets
ads_archive = pd.read_csv(data_folder / "ads_archive.csv", low_memory=False)
ads_current = pd.read_csv(data_folder / "ads_current.csv", low_memory=False)

# check dimensions
###print("ADS Archive Dimensions:", ads_archive.shape)
###print("ADS Current Dimensions:", ads_current.shape)

# convert column names to sets
archive_cols = set(ads_archive.columns)
current_cols = set(ads_current.columns)

# find similar columns
common_cols = archive_cols & current_cols

# find columns that are in archive but not in current
archive_only_cols = archive_cols - current_cols

# find columns that are in current but not in archive
current_only_cols = current_cols - archive_cols

# print results
###print("\nCommon Columns:", common_cols)
###print("\nArchive Only Columns:",)
###for col in sorted(archive_only_cols):
###    print(col)
###print("\nCurrent Only Columns:")
###for col in sorted(current_only_cols):
###    print(col)
    
# inspect injury severity categories
column = "Highest Injury Severity Alleged"

###print("\nADS Archive Injury Severity")
###print(ads_archive[column].value_counts(dropna=False))

###print("\nADS Current Injury Severity")
###print(ads_current[column].value_counts(dropna=False))

# map different injury severity categories to a common set
# 0 = No Injury
# 1 = Injury
# NaN for Unknown or Missing Values

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
    "Fatality": 1,
}

# column containing orginal injury severity labels
original_column = "Highest Injury Severity Alleged"

# create a new injury column in both datasets
ads_archive["Injury"] = ads_archive[original_column].map(injury_map)
ads_current["Injury"] = ads_current[original_column].map(injury_map)

# inspect the new injury column
###print("\nADS Archive:")
###print(ads_archive["Injury"].value_counts(dropna=False))

###print("\nADS Current:")
###print(ads_current["Injury"].value_counts(dropna=False))

# check how many rows share incident ID
# exclude missing IDs so unrelated crashes aren't grouped together
###print("\nADS Archive Duplicate Incident IDs:")
###print(
###    ads_archive["Same Incident ID"]
###    .replace(r"^\s*$", pd.NA, regex=True)
###    .dropna()
###    .duplicated()
###    .sum()
###)

###print("\nADS Current Duplicate Incident IDs:")
###print(
###    ads_current["Same Incident ID"]
###    .replace(r"^\s*$", pd.NA, regex=True)
###    .dropna()
###    .duplicated()
###    .sum()
###)

# checks repeated Report IDs
###print("\nADS Archive Duplicate Report IDs:")
###print(ads_archive["Report ID"].duplicated().sum())

###print("\nADS Current Duplicate Report IDs:")
###print(ads_current["Report ID"].duplicated().sum())

# checks for different report versions and frequency of each version
###print("\nADS Archive Report Versions:")
###print(ads_archive["Report Version"].value_counts(dropna=False))

###print("\nADS Current Report Versions:")
###print(ads_current["Report Version"].value_counts(dropna=False))

# find reports with repeated Report IDs
###duplicate_reports = ads_archive[
###    ads_archive["Report ID"].notna()
###    & ads_archive["Report ID"].duplicated(keep=False)
###]

# display first 10 rows with relevant information
###print("\n Example Duplicate Reports:")
###print(
###    duplicate_reports[
###        ["Report ID", "Report Version",
###         "Same Incident ID", "Highest Injury Severity Alleged"]
###    ]
###    .sort_values(["Report ID", "Report Version"])
###    .head(10)
###    .to_string(index=False)
###)

# check whether injury labels change across reports
for name, df in [
    ("ADS Archive", ads_archive),
    ("ADS Current", ads_current)
]:  
###    print(f"\n{name}:")
###   print("Total rows:", len(df))
###    print("Unique Report IDs:", df["Report ID"].nunique())
###    print("Missing Report IDs:", df["Report ID"].isna().sum())
###    print("Example Report IDs:")
###    print(df["Report ID"].head(10).to_list())
    
    # exclude missing Report IDs
    valid_reports = df.dropna(subset=["Report ID"])
    
    # count distinct injury labels for each Report ID
    injury_changes = (
        valid_reports.groupby("Report ID")["Injury"]
        .nunique(dropna=False)
    )
    
    # count reports with more than one injury label
    changed_reports = (injury_changes > 1).sum()
    
###    print(f"\n{name}:")
###    print(f"Reports with different injury labels: {changed_reports}")
    
    # keep the latest for each Report ID
    archive_latest = (
        ads_archive
        .sort_values("Report Version")
        .drop_duplicates(subset=["Report ID"], keep="last")
        .copy()
    )
    
    current_latest = (
        ads_current
        .sort_values("Report Version")
        .drop_duplicates(subset=["Report ID"], keep="last")
        .copy()
    )
    
    # compare number of rows before and after
###    print("\nADS Archive:")
###    print("Before:", len(ads_archive))
###    print("After:", len(archive_latest))
    
###    print("\nADS Current:")
###    print("Before:", len(ads_current))
###    print("After:", len(current_latest))
    
    # check for duplicate incidents after removing old versions
    for name, df in [
        ("ADS Archive", archive_latest),
        ("ADS Current", current_latest)
    ]:
        # remove missing and blank incidents IDs from the check
        incident_ids = (
            df["Same Incident ID"]
            .replace(r"^\s*$", pd.NA, regex=True)
            .dropna()
        )

###        print(f"\n{name}:")
###        print("Total rows:", len(df))
###        print("Unique incident IDs:", incident_ids.nunique())
###        print("Duplicate incident IDs:", incident_ids.duplicated().sum())
###        print("Missing incident IDs:", len(df) - len(incident_ids))

# find rows with repeating incident IDs
# exclude missing and blank incident IDs from the check
valid_incidents = archive_latest[
    archive_latest["Same Incident ID"].notna()
    & archive_latest["Same Incident ID"].astype(str).str.strip().ne("")
]

# keep all rows with repeating incident IDs
duplicate_incidents = valid_incidents[
    valid_incidents["Same Incident ID"].duplicated(keep=False)
]

# display examples of repeated incident IDs
###print("\nExample Duplicate Incident IDs:")
###print(
###    duplicate_incidents[
###        [
###            "Report ID",
###            "Report Version",
###            "Same Incident ID",
###            "Highest Injury Severity Alleged"
###        ]
###    ]
###    .sort_values("Same Incident ID")
###    .head(15)
###    .to_string(index=False)
###)

# check whether reports for the same incident have different injury labels
for name, df in [
    ("ADS Archive", archive_latest),
    ("ADS Current", current_latest)
]:
    # exclude missing and blank incident IDs from the check
    valid_incidents = df[
        df["Same Incident ID"].notna()
        & df["Same Incident ID"].astype(str).str.strip().ne("")
    ]
    
    # count distinct injury labels for each incident
    injury_counts = (
        valid_incidents
        .groupby("Same Incident ID")["Injury"]
        .nunique(dropna=False)
    )
    
    # count incidents with conflicting labels
    conflicting_incidents = (injury_counts > 1).sum()
    
###    print(f"\n{name}:")
###    print("Incidents with conflicting injury labels:", conflicting_incidents)
    
# exclude reports with missing or blank incident IDs
valid_incidents = archive_latest[
    archive_latest["Same Incident ID"].notna()
    & archive_latest["Same Incident ID"].astype(str).str.strip().ne("")
]

# find incident IDs with different injury labels
injury_counts = (
    valid_incidents
    .groupby("Same Incident ID")["Injury"]
    .nunique(dropna=False)
)

conflicting_ids = injury_counts[injury_counts > 1].index

# display every report belonging to those incidents
conflicting_reports = valid_incidents[valid_incidents["Same Incident ID"].isin(conflicting_ids)]

###print("\nConflicting Reports:")
###print(
###    conflicting_reports[
###        [
###            "Report ID",
###            "Report Version",
###            "Same Incident ID",
###            "Highest Injury Severity Alleged",
###            "Injury"
###        ]
###    ]
###    .sort_values(["Same Incident ID", "Report ID"])
###    .to_string(index=False)
###)

# Get valid incident IDs from ADS Archive
archive_ids = set(
    archive_latest["Same Incident ID"]
    .dropna()
    .astype(str)
    .str.strip()
) - {""}

# Get valid incident IDs from ADS Current
current_ids = set(
    current_latest["Same Incident ID"]
    .dropna()
    .astype(str)
    .str.strip()
) - {""}

# Find incident IDs that appear in BOTH datasets
overlapping_ids = archive_ids.intersection(current_ids)

###print("\nADS Archive and Current Overlap:")
###print("Archive unique incident IDs:", len(archive_ids))
###print("Current unique incident IDs:", len(current_ids))
###print("Overlapping incident IDs:", len(overlapping_ids))

# Show a few overlapping IDs, if any exist
###print("Example overlapping IDs:", list(overlapping_ids)[:10])

# Features we plan to use in our model
features_to_check = [
    "Crash With",
    "Roadway Type",
    "SV Precrash Speed (MPH)",
    "SV Pre-Crash Movement"
]

# Exclude rows with missing or blank incident IDs
valid_incidents = archive_latest[
    archive_latest["Same Incident ID"].notna()
    & archive_latest["Same Incident ID"].astype(str).str.strip().ne("")
]

# Check whether each feature differs between reports of the same incident
print("\nADS Archive Feature Conflicts:")

for feature in features_to_check:
    # Count distinct values for each incident
    distinct_values = (
        valid_incidents
        .groupby("Same Incident ID")[feature]
        .nunique(dropna=False)
    )

    # Count incidents with different values
    conflicts = (distinct_values > 1).sum()

    print(f"{feature}: {conflicts} conflicting incidents")

# Check conflicts where reports have different KNOWN values
print("\nADS Archive Actual Feature Disagreements:")

for feature in features_to_check:

    # Count unique non-missing values for each incident
    distinct_known_values = (
        valid_incidents
        .groupby("Same Incident ID")[feature]
        .nunique(dropna=True)
    )

    # More than one known value means an actual disagreement
    disagreements = (distinct_known_values > 1).sum()

    print(f"{feature}: {disagreements} actual disagreements")

# Find incidents with different known values for at least one feature
conflicting_ids = set()

for feature in features_to_check:
    distinct_values = (
        valid_incidents
        .groupby("Same Incident ID")[feature]
        .nunique(dropna=True)
    )

    # Add IDs with actual disagreements
    conflicting_ids.update(
        distinct_values[distinct_values > 1].index
    )

# Select all reports belonging to those incidents
conflicting_reports = valid_incidents[
    valid_incidents["Same Incident ID"].isin(conflicting_ids)
]

# Display their relevant information
print("\nIncidents with Actual Feature Disagreements:")
print(
    conflicting_reports[
        [
            "Same Incident ID",
            "Report ID",
            "Report Version",
            "Injury",
            "Crash With",
            "Roadway Type",
            "SV Precrash Speed (MPH)",
            "SV Pre-Crash Movement"
        ]
    ]
    .sort_values(["Same Incident ID", "Report ID"])
    .to_string(index=False)
)

# Count how many of our four features are available in each report
features_to_check = [
    "Crash With",
    "Roadway Type",
    "SV Precrash Speed (MPH)",
    "SV Pre-Crash Movement"
]

for name, df in [
    ("ADS Archive", archive_latest),
    ("ADS Current", current_latest)
]:
    # Treat blank strings as missing values
    feature_data = df[features_to_check].replace(
        r"^\s*$", pd.NA, regex=True
    )

    # Count non-missing features per report
    completeness = feature_data.notna().sum(axis=1)

    print(f"\n{name} Feature Completeness:")
    print(completeness.value_counts().sort_index())
