# ADS injury rate by crash condition
# run from repo root: python scripts/eda_injury_rates.py
# injury rate = share of crashes in a group with Injury = 1 (mean of the Injury column)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parent.parent
ads = pd.read_csv(root / "data" / "processed" / "ads_model.csv")
(root / "results" / "injury_rates").mkdir(parents=True, exist_ok=True)
(root / "docs" / "figures").mkdir(parents=True, exist_ok=True)

MIN_CRASHES = 30  # smaller groups are too noisy to show
overall = ads["Injury"].mean()
print(f"{len(ads)} ADS crashes, overall injury rate {overall:.1%}")

# speed bins in mph (0 = vehicle was stopped)
ads["Speed (mph)"] = pd.cut(ads["SV Precrash Speed (MPH)"], [-0.1, 0, 10, 25, 45, 200],
                            labels=["0", "1-10", "11-25", "26-45", "45+"])

# weather / wet surface are 0/1 columns -> one label per crash for the "yes" rows
weather_cols = [c for c in ads.columns if c.startswith("Weather") or c == "Wet Surface"]
weather = ads.melt(id_vars="Injury", value_vars=weather_cols, var_name="Weather / surface")
weather = weather[weather["value"] == 1]

rows = []
for df, col in [(ads, "Crash With"), (ads, "Roadway Type"), (ads, "Speed (mph)"), (weather, "Weather / surface")]:
    t = df.groupby(col, observed=True)["Injury"].agg(injury_rate="mean", crashes="size")
    rows.append(t.reset_index().rename(columns={col: "value"}).assign(condition=col))
rates = pd.concat(rows)
rates = rates[rates["crashes"] >= MIN_CRASHES]
rates.round(3).to_csv(root / "results" / "injury_rates" / "injury_rates.csv", index=False)

assert rates["injury_rate"].between(0, 1).all()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (cond, t) in zip(axes.flat, rates.groupby("condition", sort=False)):
    ax.barh(t["value"].astype(str) + "  (n=" + t["crashes"].astype(str) + ")", t["injury_rate"] * 100)
    ax.axvline(overall * 100, color="gray", linestyle="--")
    ax.set_title(cond)
    ax.set_xlabel("Injury rate (% of crashes)")
fig.suptitle(f"ADS injury rate by condition (dashed line = overall {overall:.1%})")
fig.tight_layout()
fig.savefig(root / "docs" / "figures" / "injury_rates.png", dpi=150)