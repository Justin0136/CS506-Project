import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Locate the processed data folder
data_folder = Path(__file__).resolve().parent.parent.parent / "CS506-Project" / "data" / "processed"

# Choose where to save the plots
output_folder = Path(__file__).resolve().parent.parent.parent / "CS506-Project" / "data" / "plots"

# Load the processed datasets
ads_model = pd.read_csv(data_folder / "ads_model.csv", low_memory=False)
level2_model = pd.read_csv(data_folder / "level2_model.csv", low_memory=False)

# Injury mask
injury_mask_ads = ads_model["Injury"] == 1
injury_mask_level2 = level2_model["Injury"] == 1
no_injury_mask_ads = ads_model["Injury"] == 0
no_injury_mask_level2 = level2_model["Injury"] == 0

# Apply the injury mask to filter the datasets
ads_model_injury = ads_model[injury_mask_ads]
level2_model_injury = level2_model[injury_mask_level2]
ads_model_no_injury = ads_model[no_injury_mask_ads]
level2_model_no_injury = level2_model[no_injury_mask_level2]

# Get the value counts of the "Roadway Type" column for both datasets
ads_VC = ads_model_injury["Roadway Type"].value_counts()
level2_VC = level2_model_injury["Roadway Type"].value_counts()
ads_VC_no_injury = ads_model_no_injury["Roadway Type"].value_counts()
level2_VC_no_injury = level2_model_no_injury["Roadway Type"].value_counts()

# Sort alphabetically by index for easier comparison
ads_sortedVC = ads_VC.sort_index()
level2_sortedVC = level2_VC.sort_index()
ads_sortedVC_no_injury = ads_VC_no_injury.sort_index()
level2_sortedVC_no_injury = level2_VC_no_injury.sort_index()

# Rename the Series before converting to DataFrames
ads_sortedDF = ads_sortedVC.rename("ADS").to_frame()
level2_sortedDF = level2_sortedVC.rename("Level 2").to_frame()
ads_sortedDF_no_injury = ads_sortedVC_no_injury.rename("ADS (No Injury)").to_frame()
level2_sortedDF_no_injury = level2_sortedVC_no_injury.rename("Level 2 (No Injury)").to_frame() 

# Print for reference
print("\nADS Roadway Type Dataframe:")
print(ads_sortedDF)
print("\nLevel 2 Roadway Type Dataframe:")
print(level2_sortedDF)
print("\nADS (No Injury) Roadway Type Dataframe:")
print(ads_sortedDF_no_injury)
print("\nLevel 2 (No Injury) Roadway Type Dataframe:")
print(level2_sortedDF_no_injury)

# Count each category for normalization
total_ads_injury = ads_sortedDF.sum()
total_level2_injury = level2_sortedDF.sum()
total_ads_no_injury = ads_sortedDF_no_injury.sum()
total_level2_no_injury = level2_sortedDF_no_injury.sum()

#print totals for reference
print(f"\nTotal ADS (Injury): {total_ads_injury.values[0]}")
print(f"Total Level 2 (Injury): {total_level2_injury.values[0]}")
print(f"Total ADS (No Injury): {total_ads_no_injury.values[0]}")
print(f"Total Level 2 (No Injury): {total_level2_no_injury.values[0]}")

# Plot two separate figures: Injury and No Injury

output_folder.mkdir(parents=True, exist_ok=True)

# Use a shared category order within each figure for easier comparison
injury_categories = ads_sortedVC.index.union(level2_sortedVC.index).sort_values()
no_injury_categories = (
    ads_sortedVC_no_injury.index
    .union(level2_sortedVC_no_injury.index)
    .sort_values()
)

ads_injury_plot = ads_sortedVC.reindex(injury_categories, fill_value=0)
level2_injury_plot = level2_sortedVC.reindex(injury_categories, fill_value=0)

ads_no_injury_plot = ads_sortedVC_no_injury.reindex(no_injury_categories, fill_value=0)
level2_no_injury_plot = level2_sortedVC_no_injury.reindex(no_injury_categories, fill_value=0)

# Figure 1: Injury = 1
fig_injury, axes_injury = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

ads_injury_plot.plot(kind="bar", ax=axes_injury[0], color="tomato")
axes_injury[0].set_title(f"ADS — Injury (n = {int(ads_injury_plot.sum())})")
axes_injury[0].set_xlabel("Roadway Type")
axes_injury[0].set_ylabel("Count")

level2_injury_plot.plot(kind="bar", ax=axes_injury[1], color="steelblue")
axes_injury[1].set_title(f"Level 2 — Injury (n = {int(level2_injury_plot.sum())})")
axes_injury[1].set_xlabel("Roadway Type")
axes_injury[1].set_ylabel("Count")

fig_injury.suptitle("Roadway Type Counts — Injury", fontsize=16)

for ax in axes_injury:
    ax.tick_params(axis="x", labelrotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

fig_injury.tight_layout()
fig_injury.savefig(
    output_folder / "injury_roadway_type_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# Figure 2: Injury = 0 (No Injury)
fig_no_injury, axes_no_injury = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

ads_no_injury_plot.plot(kind="bar", ax=axes_no_injury[0], color="tomato")
axes_no_injury[0].set_title(
    f"ADS — No Injury (n = {int(ads_no_injury_plot.sum())})"
)
axes_no_injury[0].set_xlabel("Roadway Type")
axes_no_injury[0].set_ylabel("Count")

level2_no_injury_plot.plot(kind="bar", ax=axes_no_injury[1], color="steelblue")
axes_no_injury[1].set_title(
    f"Level 2 — No Injury (n = {int(level2_no_injury_plot.sum())})"
)
axes_no_injury[1].set_xlabel("Roadway Type")
axes_no_injury[1].set_ylabel("Count")

fig_no_injury.suptitle("Roadway Type Counts — No Injury", fontsize=16)

for ax in axes_no_injury:
    ax.tick_params(axis="x", labelrotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

fig_no_injury.tight_layout()
fig_no_injury.savefig(
    output_folder / "no_injury_roadway_type_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Make the 2 plots for normalized counts (percentages) for easier comparison
# Normalize counts to percentages
ads_sortedDF["ADS (%)"] = (ads_sortedDF["ADS"] / total_ads_injury.values[0]) * 100
level2_sortedDF["Level 2 (%)"] = (level2_sortedDF["Level 2"] / total_level2_injury.values[0]) * 100
ads_sortedDF_no_injury["ADS (No Injury %)"] = (ads_sortedDF_no_injury["ADS (No Injury)"] / total_ads_no_injury.values[0]) * 100
level2_sortedDF_no_injury["Level 2 (No Injury %)"] = (level2_sortedDF_no_injury["Level 2 (No Injury)"] / total_level2_no_injury.values[0]) * 100

# Figure 1: Injury = 1
fig_injury, axes_injury = plt.subplots(
    1, 2, figsize=(16, 7), sharey=True
)

# Plot percentages, not counts
ads_sortedDF["ADS (%)"].plot(
    kind="bar", ax=axes_injury[0], color="tomato"
)
axes_injury[0].set_title(
    f"ADS — Injury (n = {int(total_ads_injury.values[0])})"
)
axes_injury[0].set_xlabel("Roadway Type")
axes_injury[0].set_ylabel("Percentage (%)")
axes_injury[0].set_ylim(0, 100)

level2_sortedDF["Level 2 (%)"].plot(
    kind="bar", ax=axes_injury[1], color="steelblue"
)
axes_injury[1].set_title(
    f"Level 2 — Injury (n = {int(total_level2_injury.values[0])})"
)
axes_injury[1].set_xlabel("Roadway Type")
axes_injury[1].set_ylabel("Percentage (%)")
axes_injury[1].set_ylim(0, 100)

fig_injury.suptitle("Roadway Type % — Injury", fontsize=16)

for ax in axes_injury:
    ax.tick_params(axis="x", labelrotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

fig_injury.tight_layout()
fig_injury.savefig(
    output_folder / "normalized_injury_roadway_type_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# Figure 2: Injury = 0 (No Injury)
fig_no_injury, axes_no_injury = plt.subplots(
    1, 2, figsize=(16, 7), sharey=True
)

# Plot percentages, not counts
ads_sortedDF_no_injury["ADS (No Injury %)"].plot(
    kind="bar", ax=axes_no_injury[0], color="tomato"
)
axes_no_injury[0].set_title(
    f"ADS — No Injury (n = {int(total_ads_no_injury.values[0])})"
)
axes_no_injury[0].set_xlabel("Roadway Type")
axes_no_injury[0].set_ylabel("Percentage (%)")
axes_no_injury[0].set_ylim(0, 100)

level2_sortedDF_no_injury["Level 2 (No Injury %)"].plot(
    kind="bar", ax=axes_no_injury[1], color="steelblue"
)
axes_no_injury[1].set_title(
    f"Level 2 — No Injury (n = {int(total_level2_no_injury.values[0])})"
)
axes_no_injury[1].set_xlabel("Roadway Type")
axes_no_injury[1].set_ylabel("Percentage (%)")
axes_no_injury[1].set_ylim(0, 100)

fig_no_injury.suptitle("Roadway Type % — No Injury", fontsize=16)

for ax in axes_no_injury:
    ax.tick_params(axis="x", labelrotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

fig_no_injury.tight_layout()
fig_no_injury.savefig(
    output_folder / "normalized_no_injury_roadway_type_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()