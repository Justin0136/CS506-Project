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

# Binary weather and surface columns
weather_cols = [
    "Weather - Clear",
    "Weather - Cloudy",
    "Weather - Rain",
    "Weather - Severe Wind",
    "Weather - Snow",
    "Weather - Fog/Smoke/Haze Combined",
    "Wet Surface"
]

# Store the four subsets in a dictionary
datasets = {
    "ads_injury": ads_model_injury,
    "level2_injury": level2_model_injury,
    "ads_no_injury": ads_model_no_injury,
    "level2_no_injury": level2_model_no_injury
}

# Create one value-count DataFrame for each subset
weather_counts = {}

for name, df in datasets.items():
    counts = pd.concat(
        {
            col: df[col].value_counts()
            for col in weather_cols
        },
        axis=1
    )

    # Ensure both binary values are present as rows
    counts = counts.reindex([0, 1]).fillna(0).astype(int)

    # Store the result
    weather_counts[name] = counts

# Assign individual DataFrames
ads_weather_injuryDF = weather_counts["ads_injury"]
level2_weather_injuryDF = weather_counts["level2_injury"]

ads_weather_no_injuryDF = weather_counts["ads_no_injury"]
level2_weather_no_injuryDF = weather_counts["level2_no_injury"]

# Print the concatenated DataFrames for reference
print("\nADS — Injury:")
print(ads_weather_injuryDF)

print("\nLevel 2 — Injury:")
print(level2_weather_injuryDF)

print("\nADS — No Injury:")
print(ads_weather_no_injuryDF)

print("\nLevel 2 — No Injury:")
print(level2_weather_no_injuryDF)

# Sum counts for normalization
total_ads_injury = ads_weather_injuryDF.sum()
total_level2_injury = level2_weather_injuryDF.sum()
total_ads_no_injury = ads_weather_no_injuryDF.sum()
total_level2_no_injury = level2_weather_no_injuryDF.sum()
print(f"Total ADS (Injury): {total_ads_injury}, Total Level 2 (Injury): {total_level2_injury}")
print(f"Total ADS (No Injury): {total_ads_no_injury}, Total Level 2 (No Injury): {total_level2_no_injury}")

# Figure 1: Injury = 1
fig_injury, axes_injury = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

ads_weather_injuryDF.plot(kind="bar", ax=axes_injury[0])
axes_injury[0].set_title("ADS — Injury")
axes_injury[0].set_xlabel("Weather Conditon Presence (0 = No, 1 = Yes)")
axes_injury[0].set_ylabel("Count")

level2_weather_injuryDF.plot(kind="bar", ax=axes_injury[1])
axes_injury[1].set_title("Level 2 — Injury")
axes_injury[1].set_xlabel("Weather Conditon Presence (0 = No, 1 = Yes)")
axes_injury[1].set_ylabel("Count")

fig_injury.suptitle("Weather Condition Counts — Injury", fontsize=16)

for ax in axes_injury:
    ax.tick_params(axis="x", labelrotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

fig_injury.tight_layout()
fig_injury.savefig(
    output_folder / "injury_weather_counts_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# Figure 2: Injury = 0 (No Injury)
fig_no_injury, axes_no_injury = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

ads_weather_no_injuryDF.plot(kind="bar", ax=axes_no_injury[0])
axes_no_injury[0].set_title(
    "ADS — No Injury"
)
axes_no_injury[0].set_xlabel("Weather Conditon Presence (0 = No, 1 = Yes)")
axes_no_injury[0].set_ylabel("Count")

level2_weather_no_injuryDF.plot(kind="bar", ax=axes_no_injury[1])
axes_no_injury[1].set_title(
    "Level 2 — No Injury"
)
axes_no_injury[1].set_xlabel("Weather Conditon Presence (0 = No, 1 = Yes)")
axes_no_injury[1].set_ylabel("Count")

fig_no_injury.suptitle("Weather Condition Counts — No Injury", fontsize=16)

for ax in axes_no_injury:
    ax.tick_params(axis="x", labelrotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

fig_no_injury.tight_layout()
fig_no_injury.savefig(
    output_folder / "no_injury_weather_counts_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Normalize counts to percentages for injury and no injury datasets
ads_weather_injuryDF_normalized = ads_weather_injuryDF.div(total_ads_injury) * 100
level2_weather_injuryDF_normalized = level2_weather_injuryDF.div(total_level2_injury) * 100
ads_weather_no_injuryDF_normalized = ads_weather_no_injuryDF.div(total_ads_no_injury) * 100
level2_weather_no_injuryDF_normalized = level2_weather_no_injuryDF.div(total_level2_no_injury) * 100

# Create normalized plots
# Figure 1: Injury = 1
fig_injury, axes_injury = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

ads_weather_injuryDF_normalized.plot(kind="bar", ax=axes_injury[0])
axes_injury[0].set_title("ADS — Injury")
axes_injury[0].set_xlabel("Weather Conditon Presence (0 = No, 1 = Yes)")
axes_injury[0].set_ylabel("Percentage")

level2_weather_injuryDF_normalized.plot(kind="bar", ax=axes_injury[1])
axes_injury[1].set_title("Level 2 — Injury")
axes_injury[1].set_xlabel("Weather Conditon Presence (0 = No, 1 = Yes)")
axes_injury[1].set_ylabel("Percentage")

fig_injury.suptitle("Weather Condition Percentage of Feature — Injury", fontsize=16)

for ax in axes_injury:
    ax.tick_params(axis="x", labelrotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

fig_injury.tight_layout()
fig_injury.savefig(
    output_folder / "injury_weather_percentages_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# Figure 2: Injury = 0 (No Injury)
fig_no_injury, axes_no_injury = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

ads_weather_no_injuryDF_normalized.plot(kind="bar", ax=axes_no_injury[0])
axes_no_injury[0].set_title(
    "ADS — No Injury"
)
axes_no_injury[0].set_xlabel("Weather Conditon Presence (0 = No, 1 = Yes)")
axes_no_injury[0].set_ylabel("Percentage")

level2_weather_no_injuryDF_normalized.plot(kind="bar", ax=axes_no_injury[1])
axes_no_injury[1].set_title(
    "Level 2 — No Injury"
)
axes_no_injury[1].set_xlabel("Weather Conditon Presence (0 = No, 1 = Yes)")
axes_no_injury[1].set_ylabel("Percentage")

fig_no_injury.suptitle("Weather Condition Percentages — No Injury", fontsize=16)

for ax in axes_no_injury:
    ax.tick_params(axis="x", labelrotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

fig_no_injury.tight_layout()
fig_no_injury.savefig(
    output_folder / "no_injury_weather_percentages_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()  