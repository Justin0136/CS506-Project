# Crash trends over time: monthly report counts for ADS and Level 2 ADAS
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

cols = ["Report ID", "Report Version", "Same Incident ID", "Incident Date",
        "Report Submission Date", "Automation System Engaged?"]

files = [
    ("ADS_archive.csv", "ADS", "archive"),
    ("ADS_current.csv", "ADS", "current"),
    ("ADAS_archive.csv", "Level 2 ADAS", "archive"),
    ("ADAS_current.csv", "Level 2 ADAS", "current"),
]

# load the 4 files
# ADAS_archive.csv has some bad utf-8 bytes in text columns, so use encoding_errors="replace"
dfs = []
for file_name, auto_type, release in files:
    df = pd.read_csv(data_folder / file_name, usecols=cols, dtype=str,
                     keep_default_na=False, encoding_errors="replace")
    df["type"] = auto_type
    df["release"] = release
    dfs.append(df)
raw = pd.concat(dfs, ignore_index=True)
raw["Report Version"] = raw["Report Version"].astype(int)

# dates look like "JUN-2025"
raw["incident_month"] = pd.to_datetime(raw["Incident Date"].str.strip().str.title(), format="%b-%Y", errors="coerce")
raw["submit_month"] = pd.to_datetime(raw["Report Submission Date"].str.strip().str.title(), format="%b-%Y", errors="coerce")

# first submission month of each report (version 1, or the lowest version we have)
raw = raw.sort_values("Report Version")
first_submit = raw.groupby(["type", "Report ID"])["submit_month"].first()

# keep only the latest version of each Report ID
# do this after combining archive + current, because 37 ADAS report ids are in both files
latest = raw.drop_duplicates(["type", "Report ID"], keep="last").copy()
latest["first_submit"] = first_submit.reindex(pd.MultiIndex.from_frame(latest[["type", "Report ID"]])).values

# missing date -> drop, before 2021-07 -> not in monthly counts (only report how many)
missing_date = latest[latest["incident_month"].isna()]
dated = latest[latest["incident_month"].notna()]
before_sgo = dated[dated["incident_month"] < "2021-07-01"]
used = dated[dated["incident_month"] >= "2021-07-01"]

# monthly counts
monthly = used.groupby(["incident_month", "type"]).size().unstack(fill_value=0)
all_months = pd.date_range("2021-07-01", used["incident_month"].max(), freq="MS")
monthly = monthly.reindex(all_months, fill_value=0)[["ADS", "Level 2 ADAS"]]
monthly.index.name = "month"
monthly = monthly.reset_index()

# last month in data is not complete yet (reports come in late)
last_submit = raw["submit_month"].max()
monthly["possibly_incomplete"] = monthly["month"] >= last_submit
monthly["month"] = monthly["month"].dt.strftime("%Y-%m")
monthly.to_csv(out_folder / "monthly_report_counts.csv", index=False)

# data quality numbers
lag = ((used["first_submit"].dt.year - used["incident_month"].dt.year) * 12
       + (used["first_submit"].dt.month - used["incident_month"].dt.month))

summary = {}
for t in ["ADS", "Level 2 ADAS"]:
    r = raw[raw["type"] == t]
    l = latest[latest["type"] == t]
    u = used[used["type"] == t]
    right_label = "ADS" if t == "ADS" else "ADAS"

    # just for checking: count distinct incidents inside each file
    # (Same Incident ID is different between archive and current, so can't match across files)
    has_id = l[l["Same Incident ID"].str.strip() != ""]
    n_incidents = len(has_id[["release", "Same Incident ID"]].drop_duplicates()) + (len(l) - len(has_id))

    summary[t] = {
        "raw rows (all versions)": len(r),
        "raw rows archive": (r["release"] == "archive").sum(),
        "raw rows current": (r["release"] == "current").sum(),
        "report ids after keeping latest version": len(l),
        "old versions removed": len(r) - len(l),
        "report ids in both archive and current": (r.groupby("Report ID")["release"].nunique() > 1).sum(),
        "missing incident date (dropped)": (missing_date["type"] == t).sum(),
        "incident before 2021-07 (not in monthly counts)": (before_sgo["type"] == t).sum(),
        "reports in monthly counts": len(u),
        "sum of monthly counts": monthly[t].sum(),
        "earliest incident month (all)": l["incident_month"].min().strftime("%Y-%m"),
        "first month used": u["incident_month"].min().strftime("%Y-%m"),
        "last month used": u["incident_month"].max().strftime("%Y-%m"),
        "distinct incidents in each file (check only, not used)": n_incidents,
        "automation engaged field not matching file (incl unknown)": (l["Automation System Engaged?"].str.strip() != right_label).sum(),
        "% submitted within 1 month of incident": round(100 * (lag[used["type"] == t] <= 1).mean(), 1),
    }
summary = pd.DataFrame(summary)
summary.index.name = "metric"
summary.to_csv(out_folder / "data_quality_summary.csv")
print(summary)

# simple checks
for t in ["ADS", "Level 2 ADAS"]:
    s = summary[t]
    assert s["sum of monthly counts"] == s["reports in monthly counts"]
    assert s["report ids after keeping latest version"] == (s["reports in monthly counts"]
           + s["missing incident date (dropped)"] + s["incident before 2021-07 (not in monthly counts)"])
assert not latest.duplicated(["type", "Report ID"]).any()
max_version = raw.groupby(["type", "Report ID"])["Report Version"].max()
assert (latest.set_index(["type", "Report ID"])["Report Version"].sort_index() == max_version.sort_index()).all()
print("checks ok")

# plot (read back the csv so plot = csv)
m = pd.read_csv(out_folder / "monthly_report_counts.csv")
m["month"] = pd.to_datetime(m["month"])

plt.figure(figsize=(12, 5.5))
plt.plot(m["month"], m["ADS"], marker="o", markersize=3, label="ADS")
plt.plot(m["month"], m["Level 2 ADAS"], marker="o", markersize=3, label="Level 2 ADAS")

# from the NHTSA data dictionaries: archive = reports up to June 13 2025, current = third amended SGO reports from June 16 2025
plt.axvline(pd.Timestamp("2025-06-16"), color="gray", linestyle="--")
plt.text(pd.Timestamp("2025-06-16"), 255, " Third Amended SGO\n reports from Jun 16, 2025", fontsize=9, color="gray")

inc = m[m["possibly_incomplete"]]
if len(inc) > 0:
    plt.axvspan(inc["month"].min() - pd.Timedelta(days=15), inc["month"].max() + pd.Timedelta(days=15),
                color="orange", alpha=0.15, label="not complete yet")

plt.title("Monthly NHTSA SGO Crash Reports: ADS vs Level 2 ADAS")
plt.xlabel("Incident month")
plt.ylabel("Number of reports")
plt.ylim(bottom=0)
plt.grid(alpha=0.3)
plt.legend(loc="upper left")
plt.figtext(0.01, 0.01, "Report counts only, not crash rates (no mileage data, different reporting rules)",
            fontsize=8, color="gray")
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(fig_folder / "monthly_report_counts.png", dpi=150)
print("saved csv files and plot")
