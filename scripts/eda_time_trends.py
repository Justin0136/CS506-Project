# Crash trends over time: monthly report counts and unique crash counts for ADS and Level 2 ADAS
# run from repo root: python scripts/eda_time_trends.py

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

proj_root = Path(__file__).resolve().parent.parent
data_folder = proj_root / "data" / "raw"
out_folder = proj_root / "results" / "time_trends"
fig_folder = proj_root / "docs" / "figures"
out_folder.mkdir(parents=True, exist_ok=True)
fig_folder.mkdir(parents=True, exist_ok=True)

# same features Justin uses in clean_ads.py to pick the most complete report of a crash
features = ["Crash With", "Roadway Type", "SV Precrash Speed (MPH)", "SV Pre-Crash Movement"]

cols = ["Report ID", "Report Version", "Same Incident ID", "Incident Date",
        "Report Submission Date", "Automation System Engaged?", "Highest Injury Severity Alleged"] + features

# (file, type, release, value of "Automation System Engaged?" to keep)
files = [
    ("ADS_archive.csv", "ADS", "archive", "ADS"),
    ("ADS_current.csv", "ADS", "current", "ADS"),
    ("ADAS_archive.csv", "Level 2 ADAS", "archive", "ADAS"),
    ("ADAS_current.csv", "Level 2 ADAS", "current", "ADAS"),
]

# injury mapping copied from clean_ads.py (only used to pick the representative report of a crash)
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

# load the 4 files
# ADAS_archive.csv has some bad utf-8 bytes in text columns, so use encoding_errors="replace"
dfs = []
for file_name, auto_type, release, engaged in files:
    df = pd.read_csv(data_folder / file_name, usecols=cols, dtype=str, encoding_errors="replace")
    df["type"] = auto_type
    df["release"] = release
    df["engaged_ok"] = df["Automation System Engaged?"] == engaged
    dfs.append(df)
raw = pd.concat(dfs, ignore_index=True)
raw["Report Version"] = raw["Report Version"].astype(int)

# dates look like "JUN-2025"
raw["incident_month"] = pd.to_datetime(raw["Incident Date"].str.strip().str.title(), format="%b-%Y", errors="coerce")
raw["submit_month"] = pd.to_datetime(raw["Report Submission Date"].str.strip().str.title(), format="%b-%Y", errors="coerce")

# same rule as clean_ads.py: keep rows where "Automation System Engaged?" matches the file (ADS / ADAS),
# and do it before keeping the latest version
kept = raw[raw["engaged_ok"]].copy()

# first submission month of each report (version 1, or the lowest version we have)
kept = kept.sort_values("Report Version", kind="stable")
first_submit = kept.groupby(["type", "Report ID"])["submit_month"].first()

# keep only the latest version of each Report ID
# do this after combining archive + current, because 37 ADAS report ids are in both files
latest = kept.drop_duplicates(["type", "Report ID"], keep="last").copy()
latest["first_submit"] = first_submit.reindex(pd.MultiIndex.from_frame(latest[["type", "Report ID"]])).values

# ---- crash level ----
# like clean_ads.py: reports with the same Same Incident ID (inside one release file) are one crash.
# keep the report with known injury first, then most complete features, then highest version.
# reports without Same Incident ID count as their own crash.
# archive and current are not matched, Same Incident ID is different between them.
latest["incident_key"] = latest["Same Incident ID"].str.strip()
no_id = latest["incident_key"].isna() | (latest["incident_key"] == "")
latest.loc[no_id, "incident_key"] = "no id: " + latest.loc[no_id, "Report ID"]
latest["crash_id"] = latest["type"] + " | " + latest["release"] + " | " + latest["incident_key"]

latest["known_injury"] = latest["Highest Injury Severity Alleged"].map(injury_map).notna().astype(int)
latest["completeness"] = latest[features].replace(["Unknown", ""], pd.NA).notna().sum(axis=1)

reports_per_crash = latest.groupby("crash_id")["Report ID"].transform("size")
months_per_crash = latest.groupby("crash_id")["incident_month"].transform("nunique")

crashes = (latest.sort_values(["known_injury", "completeness", "Report Version"], kind="stable")
           .drop_duplicates("crash_id", keep="last"))

# missing date -> drop, before 2021-07 -> not in monthly counts (only report how many)
def split_by_date(df):
    missing = df[df["incident_month"].isna()]
    dated = df[df["incident_month"].notna()]
    before = dated[dated["incident_month"] < "2021-07-01"]
    used = dated[dated["incident_month"] >= "2021-07-01"]
    return missing, before, used

missing_date, before_sgo, used = split_by_date(latest)
crash_missing, crash_before, crash_used = split_by_date(crashes)

# monthly counts
last_month = used["incident_month"].max()
last_submit = raw["submit_month"].max()
all_months = pd.date_range("2021-07-01", last_month, freq="MS")

def monthly_counts(df, file_name):
    monthly = df.groupby(["incident_month", "type"]).size().unstack(fill_value=0)
    monthly = monthly.reindex(index=all_months, columns=["ADS", "Level 2 ADAS"], fill_value=0)
    monthly.index.name = "month"
    monthly = monthly.reset_index()
    # last month in data is not complete yet (reports come in late)
    monthly["possibly_incomplete"] = monthly["month"] >= last_submit
    monthly["month"] = monthly["month"].dt.strftime("%Y-%m")
    monthly.to_csv(out_folder / file_name, index=False)
    return monthly

monthly = monthly_counts(used, "monthly_report_counts.csv")
monthly_crash = monthly_counts(crash_used, "monthly_crash_counts.csv")

# data quality numbers
lag = ((used["first_submit"].dt.year - used["incident_month"].dt.year) * 12
       + (used["first_submit"].dt.month - used["incident_month"].dt.month))

summary = {}
for t in ["ADS", "Level 2 ADAS"]:
    r = raw[raw["type"] == t]
    k = kept[kept["type"] == t]
    l = latest[latest["type"] == t]
    is_t = latest["type"] == t

    summary[t] = {
        "raw rows (all versions)": len(r),
        "raw rows archive": (r["release"] == "archive").sum(),
        "raw rows current": (r["release"] == "current").sum(),
        "rows dropped by automation engaged filter": len(r) - len(k),
        "rows after automation engaged filter": len(k),
        "report ids after keeping latest version": len(l),
        "old versions removed": len(k) - len(l),
        "report ids in both archive and current": (k.groupby("Report ID")["release"].nunique() > 1).sum(),
        "missing incident date (dropped)": (missing_date["type"] == t).sum(),
        "incident before 2021-07 (not in monthly counts)": (before_sgo["type"] == t).sum(),
        "reports in monthly counts": (used["type"] == t).sum(),
        "sum of monthly report counts": monthly[t].sum(),
        "reports without Same Incident ID (own crash)": (no_id & is_t).sum(),
        "unique crashes": (crashes["type"] == t).sum(),
        "crashes with more than 1 report": (latest[is_t & (reports_per_crash > 1)]["crash_id"].nunique()),
        "crashes whose reports have different incident months": (latest[is_t & (months_per_crash > 1)]["crash_id"].nunique()),
        "crashes missing incident date (dropped)": (crash_missing["type"] == t).sum(),
        "crashes before 2021-07 (not in monthly counts)": (crash_before["type"] == t).sum(),
        "crashes in monthly counts": (crash_used["type"] == t).sum(),
        "sum of monthly crash counts": monthly_crash[t].sum(),
        "earliest incident month (all)": l["incident_month"].min().strftime("%Y-%m"),
        "first month used": used[used["type"] == t]["incident_month"].min().strftime("%Y-%m"),
        "last month used": used[used["type"] == t]["incident_month"].max().strftime("%Y-%m"),
        "% submitted within 1 month of incident": round(100 * (lag[used["type"] == t] <= 1).mean(), 1),
    }
summary = pd.DataFrame(summary)
summary.index.name = "metric"
summary.to_csv(out_folder / "data_quality_summary.csv")
print(summary)

# simple checks
for t in ["ADS", "Level 2 ADAS"]:
    s = summary[t]
    assert s["sum of monthly report counts"] == s["reports in monthly counts"]
    assert s["report ids after keeping latest version"] == (s["reports in monthly counts"]
           + s["missing incident date (dropped)"] + s["incident before 2021-07 (not in monthly counts)"])
    assert s["sum of monthly crash counts"] == s["crashes in monthly counts"]
    assert s["unique crashes"] == (s["crashes in monthly counts"]
           + s["crashes missing incident date (dropped)"] + s["crashes before 2021-07 (not in monthly counts)"])
    assert s["unique crashes"] <= s["report ids after keeping latest version"]
assert not latest.duplicated(["type", "Report ID"]).any()
max_version = kept.groupby(["type", "Report ID"])["Report Version"].max()
assert (latest.set_index(["type", "Report ID"])["Report Version"].sort_index() == max_version.sort_index()).all()
# every crash has exactly one row, and every report belongs to one kept crash
assert not crashes["crash_id"].duplicated().any()
assert set(latest["crash_id"]) == set(crashes["crash_id"])
# no crash count is above the report count in any month
assert (monthly_crash[["ADS", "Level 2 ADAS"]] <= monthly[["ADS", "Level 2 ADAS"]]).all().all()
print("checks ok")

# compare with Justin's ads_model.csv (if it is there)
# ads_model also drops crashes with unknown injury and archive incidents with conflicting injury labels,
# so do the same here only for this comparison
model_file = proj_root / "data" / "processed" / "ads_model.csv"
if model_file.exists():
    ads_model = pd.read_csv(model_file, dtype=str)
    a = latest[(latest["type"] == "ADS") & (latest["release"] == "archive")].copy()
    a["injury"] = a["Highest Injury Severity Alleged"].map(injury_map)
    conflict = a.dropna(subset=["injury"]).groupby("crash_id")["injury"].nunique()
    conflict = set(conflict[conflict > 1].index)
    mine = crashes[(crashes["type"] == "ADS") & (crashes["known_injury"] == 1) & ~crashes["crash_id"].isin(conflict)]
    mine_ids, justin_ids = set(mine["Report ID"]), set(ads_model["Report ID"])
    ads_crashes = (crashes["type"] == "ADS").sum()
    print(f"\nADS crashes in this EDA: {ads_crashes}, with known injury and no conflict: {len(mine)}, "
          f"ads_model rows: {len(ads_model)}")
    print(f"same Report IDs: {len(mine_ids & justin_ids)}, only here: {len(mine_ids - justin_ids)}, "
          f"only in ads_model: {len(justin_ids - mine_ids)}")
    pd.DataFrame({
        "metric": ["ADS crashes in this EDA", "dropped for ads_model: unknown injury",
                   "dropped for ads_model: conflicting injury labels", "expected ads_model rows",
                   "ads_model rows", "same representative Report IDs", "only in this EDA", "only in ads_model"],
        "value": [ads_crashes, (crashes[crashes["type"] == "ADS"]["known_injury"] == 0).sum(),
                  crashes[(crashes["type"] == "ADS") & (crashes["known_injury"] == 1)]["crash_id"].isin(conflict).sum(),
                  len(mine), len(ads_model), len(mine_ids & justin_ids),
                  len(mine_ids - justin_ids), len(justin_ids - mine_ids)],
    }).to_csv(out_folder / "ads_model_comparison.csv", index=False)

# plots (read back the csv so plot = csv)
def plot(csv_name, png_name, title, ylabel):
    m = pd.read_csv(out_folder / csv_name)
    m["month"] = pd.to_datetime(m["month"])

    plt.figure(figsize=(12, 5.5))
    plt.plot(m["month"], m["ADS"], marker="o", markersize=3, label="ADS")
    plt.plot(m["month"], m["Level 2 ADAS"], marker="o", markersize=3, label="Level 2 ADAS")

    # from the NHTSA data dictionaries: archive = reports up to June 13 2025, current = third amended SGO reports from June 16 2025
    top = max(m["ADS"].max(), m["Level 2 ADAS"].max())
    plt.axvline(pd.Timestamp("2025-06-16"), color="gray", linestyle="--")
    plt.text(pd.Timestamp("2025-06-16"), top * 0.95, " Third Amended SGO\n reports from Jun 16, 2025", fontsize=9, color="gray")

    inc = m[m["possibly_incomplete"]]
    if len(inc) > 0:
        plt.axvspan(inc["month"].min() - pd.Timedelta(days=15), inc["month"].max() + pd.Timedelta(days=15),
                    color="orange", alpha=0.15, label="not complete yet")

    plt.title(title)
    plt.xlabel("Incident month")
    plt.ylabel(ylabel)
    plt.ylim(0, top * 1.08)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left")
    plt.figtext(0.01, 0.01, "Counts only, not crash rates (no mileage data, different reporting rules for ADS and Level 2)",
                fontsize=8, color="gray")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(fig_folder / png_name, dpi=150)
    plt.close()

plot("monthly_report_counts.csv", "monthly_report_counts.png",
     "Monthly NHTSA SGO Crash Reports: ADS vs Level 2 ADAS", "Number of reports (latest version per Report ID)")
plot("monthly_crash_counts.csv", "monthly_crash_counts.png",
     "Monthly Unique Crashes (Same Incident ID): ADS vs Level 2 ADAS", "Number of unique crashes")
print("saved csv files and plots")
