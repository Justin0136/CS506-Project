import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Locate the processed data folder
data_folder = Path(__file__).resolve().parent.parent / "data" / "processed"

# Choose where to save the plots
output_folder = Path(__file__).resolve().parent.parent / "data" / "plots"

# Load the processed datasets
ads_model = pd.read_csv(data_folder / "ads_model.csv", low_memory=False)
level2_model = pd.read_csv(data_folder / "level2_model.csv", low_memory=False)

# Get the value counts of the "CrashWith" column for both datasets
ads_VC = ads_model["Crash With"].value_counts()
level2_VC = level2_model["Crash With"].value_counts()

# Sort alphabetically by index for easier comparison
ads_sortedVC = ads_VC.sort_index()
level2_sortedVC = level2_VC.sort_index()

# Rename the Series before converting to DataFrames
ads_sortedDF = ads_sortedVC.rename("ADS").to_frame()
level2_sortedDF = level2_sortedVC.rename("Level 2").to_frame()

# Combine, aligning categories by index
combined_counts = pd.concat(
    [ads_sortedDF, level2_sortedDF],
    axis=1
).fillna(0).astype(int)

combined_counts.index.name = "Crash With"

# Sort by category alphabetically if needed again
combined_counts = combined_counts.sort_index()


# Print for reference
print("\nADS Crash With Dataframe:")
print(ads_sortedDF)
print("\nLevel 2 Crash With Dataframe:")
print(level2_sortedDF)

# Sum counts for normalization
total_ads = combined_counts["ADS"].sum()
total_level2 = combined_counts["Level 2"].sum()
print(f"Total ADS: {total_ads}, Total Level 2: {total_level2}")

# Plot grouped bar chart (not normalized)
ax = combined_counts.plot(
    kind="bar",
    stacked=False,
    figsize=(12, 6),
    title=f"Non-Normalized Crashed With Counts: ADS (n={total_ads}) vs Level 2 (n={total_level2})",
    ylabel="Count",
    xlabel="Crashed With",
    width=0.8
)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# save the plot to the output folder
plt.savefig(
    output_folder / "crashed_with_counts.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# Normalize counts to percentages
combined_counts["ADS (%)"] = (combined_counts["ADS"] / total_ads) * 100
combined_counts["Level 2 (%)"] = (combined_counts["Level 2"] / total_level2) * 100

# Plot grouped bar chart (normalized)
ax_norm = combined_counts[["ADS (%)", "Level 2 (%)"]].plot(
    kind="bar",
    stacked=False,
    figsize=(12, 6),
    title=f"Normalized Crashed With Counts: ADS (n={total_ads}) vs Level 2 (n={total_level2})",
    ylabel="Percentage of Each Dataset (%)",
    xlabel="Crashed With",
    width=0.8
)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# save the normalized plot to the output folder
plt.savefig(
    output_folder / "crashed_with_normalized.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# For reference
print(combined_counts[["ADS", "Level 2", "ADS (%)", "Level 2 (%)"]])