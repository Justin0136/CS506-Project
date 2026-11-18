# Crash trends over time: monthly report counts and unique crash counts for ADS and Level 2 ADAS,
# and ADS injury rates by incident year, vehicle model year, and operator (Waymo vs non-Waymo)
# run from repo root: python scripts/eda_time_trends.py

import numpy as np
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
        "Report Submission Date", "Automation System Engaged?", "Highest Injury Severity Alleged",
        "Model Year", "Operating Entity", "Reporting Entity"] + features

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

# ---- ADS injury rate by incident year, model year and operator ----
# one row per ADS crash (the crashes table above). injury rate = injured / known injury.
# unknown = no report of the crash has a known injury; conflict = its reports disagree (injury vs no injury).
# both are counted but left out of the rate, ads_model.csv drops both.
ads_reports = latest[latest["type"] == "ADS"].copy()
ads_reports["injury"] = ads_reports["Highest Injury Severity Alleged"].map(injury_map)
labels_per_crash = ads_reports.groupby("crash_id")["injury"].nunique()
conflict_ids = set(labels_per_crash[labels_per_crash > 1].index)

ads = crashes[crashes["type"] == "ADS"].copy()
ads["injury"] = ads["Highest Injury Severity Alleged"].map(injury_map)
ads["status"] = ads["injury"].map({1: "injured", 0: "non_injured"}).fillna("unknown")
ads.loc[ads["crash_id"].isin(conflict_ids), "status"] = "conflict"
ads["incident_year"] = ads["incident_month"].dt.year

# model year of the representative report, like prepare_ads_model.py (archive has "2021.0", current "2021")
ads["model_year"] = pd.to_numeric(ads["Model Year"], errors="coerce").astype("Int64")
ads["model_year_group"] = ads["model_year"].map(lambda v: "Unknown" if pd.isna(v) else str(int(v)))

# operator from the Operating Entity of every report of the crash (Transdev files reports for Waymo cars).
# Waymo is written many ways ("Waymo LLC", "Waymo, LLC.", "WAYMO LLC"), so match "waymo" ignoring case.
op = ads_reports["Operating Entity"].str.strip()
ads_reports["op_waymo"] = op.str.contains("waymo", case=False, na=False)
ads_reports["op_other"] = op.notna() & (op != "") & ~ads_reports["op_waymo"]
op_flags = ads_reports.groupby("crash_id")[["op_waymo", "op_other"]].any()
operator = pd.Series("Operator unknown", index=op_flags.index)
operator[op_flags["op_waymo"] & ~op_flags["op_other"]] = "Waymo"
operator[~op_flags["op_waymo"] & op_flags["op_other"]] = "Non-Waymo"
operator[op_flags["op_waymo"] & op_flags["op_other"]] = "Operator mixed"
ads["operator"] = ads["crash_id"].map(operator)

STATUS = ["injured", "non_injured", "unknown", "conflict"]
MIN_N = 30  # same cutoff as eda_injury_rates.py, applied to the rate denominator


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(mid - half, 0.0), min(mid + half, 1.0)


def injury_table(df, by):
    t = df.groupby(by)["status"].value_counts().unstack(fill_value=0)
    t = t.reindex(columns=STATUS, fill_value=0)
    t.loc["All"] = t.sum()
    t.columns.name = None
    t.index.name = by
    t.insert(0, "total_crashes", t[STATUS].sum(axis=1))
    t.insert(1, "known_injury", t["injured"] + t["non_injured"])
    ci = [wilson(k, n) for k, n in zip(t["injured"], t["known_injury"])]
    t["injury_rate_pct"] = (100 * t["injured"] / t["known_injury"].replace(0, np.nan)).round(2)
    t["ci95_low_pct"] = [round(100 * lo, 2) for lo, _ in ci]
    t["ci95_high_pct"] = [round(100 * hi, 2) for _, hi in ci]
    t["small_sample"] = t["known_injury"] < MIN_N
    return t.reset_index()


months_per_year = pd.Series(all_months.year).value_counts().sort_index()
year_table = injury_table(ads, "incident_year")
year_table["months_covered"] = year_table["incident_year"].map(months_per_year).astype("Int64")
year_table["partial_year"] = (year_table["months_covered"] < 12).fillna(False).astype(bool)
first_m = pd.Series(all_months).groupby(all_months.year).min().dt.strftime("%b")
last_m = pd.Series(all_months).groupby(all_months.year).max().dt.strftime("%b")
year_table["note"] = [f"{first_m[y]}-{last_m[y]} only" if p else "" for y, p in
                      zip(year_table["incident_year"], year_table["partial_year"])]
year_table.loc[year_table["incident_year"] == 2025, "note"] = "Third Amended SGO reports from Jun 16, 2025"
year_table.to_csv(out_folder / "injury_rate_by_incident_year.csv", index=False)

my_table = injury_table(ads, "model_year_group")
my_table.to_csv(out_folder / "injury_rate_by_model_year.csv", index=False)

groups = [("All ADS", ads)] + [(g, ads[ads["operator"] == g])
                               for g in ["Waymo", "Non-Waymo", "Operator mixed", "Operator unknown"]]
op_table = pd.concat([injury_table(df, "incident_year").assign(group=g) for g, df in groups if len(df) > 0])
op_table = op_table[["group"] + [c for c in op_table.columns if c != "group"]]
op_table.to_csv(out_folder / "injury_rate_waymo_vs_all_by_year.csv", index=False)

print("\nADS injury rate by incident year:")
print(year_table.to_string(index=False))
print("\nADS injury rate by model year:")
print(my_table.to_string(index=False))
print("\nADS injury rate by operator and year:")
print(op_table.to_string(index=False))

# checks: every row is written to injury_rate_checks.csv, rows with an expected value must match
checks = []


def check(name, value, expected=None):
    ok = "" if expected is None else bool(value == expected)
    checks.append({"check": name, "value": value, "expected": "" if expected is None else expected, "pass": ok})


total = year_table.set_index("incident_year").loc["All"]
check("ADS crashes (same as data_quality_summary unique crashes)", len(ads), summary.loc["unique crashes", "ADS"])
check("ADS crashes", int(total["total_crashes"]), 2928)
check("known injury crashes", int(total["known_injury"]), 2878)
check("unknown injury crashes", int(total["unknown"]), 49)
check("conflict crashes", int(total["conflict"]), 1)
check("conflict crashes in the current file (clean_ads.py only checks archive)",
      sum(cid.startswith("ADS | current") for cid in conflict_ids), 0)
check("crash ids unique", int(ads["crash_id"].duplicated().sum()), 0)

for name, t in [("incident year", year_table), ("model year", my_table), ("operator x year", op_table)]:
    check(f"{name}: known + unknown + conflict = total in every row",
          bool((t["known_injury"] + t["unknown"] + t["conflict"] == t["total_crashes"]).all()), True)
    check(f"{name}: injured + non_injured = known in every row",
          bool((t["injured"] + t["non_injured"] == t["known_injury"]).all()), True)
    rate = (100 * t["injured"] / t["known_injury"].replace(0, np.nan)).round(2)
    check(f"{name}: injury rate = injured / known", bool(np.allclose(rate, t["injury_rate_pct"], equal_nan=True)), True)
    check(f"{name}: rate inside 95% CI",
          bool(((t["ci95_low_pct"] <= t["injury_rate_pct"]) & (t["injury_rate_pct"] <= t["ci95_high_pct"])).all()), True)

for name, t in [("incident year", year_table), ("model year", my_table)]:
    rows, all_row = t[t.iloc[:, 0] != "All"], t[t.iloc[:, 0] == "All"].iloc[0]
    for c in ["total_crashes", "known_injury", "injured", "unknown", "conflict"]:
        check(f"{name}: sum of groups = All ({c})", int(rows[c].sum()), int(all_row[c]))

check("model year: crashes with missing model year", int(ads["model_year"].isna().sum()))
check("model year: Unknown group = crashes with missing model year",
      int(my_table.loc[my_table["model_year_group"] == "Unknown", "total_crashes"].sum()), int(ads["model_year"].isna().sum()))
check("model year: All row = ADS crashes", int(my_table.loc[my_table["model_year_group"] == "All", "total_crashes"].iloc[0]),
      len(ads))
check("model year: crashes with model year after incident year", int((ads["model_year"] > ads["incident_year"]).sum()))
for d, n in (ads["model_year"] - ads["incident_year"]).value_counts().sort_index().items():
    if d > 0:
        check(f"model year: crashes with model year {d} year(s) after incident year", int(n))
my_per_crash = pd.to_numeric(ads_reports["Model Year"], errors="coerce").groupby(ads_reports["crash_id"]).nunique()
check("model year: crashes whose reports have different model years", int((my_per_crash > 1).sum()))

per_group = ads["operator"].value_counts()
for g in ["Waymo", "Non-Waymo", "Operator mixed", "Operator unknown"]:
    check(f"operator: {g} crashes", int(per_group.get(g, 0)))
check("operator: groups add up to All ADS", int(per_group.sum()), len(ads))
check("operator: every crash in exactly one group", int(ads["operator"].isna().sum()), 0)
op_all = op_table[op_table["incident_year"] == "All"].set_index("group")
check("operator: Waymo + Non-Waymo + mixed + unknown known injury = All ADS known injury",
      int(op_all.drop("All ADS")["known_injury"].sum()), int(op_all.loc["All ADS", "known_injury"]))
check("operator: Waymo crashes where any report's Reporting Entity is not Waymo",
      int(ads_reports[ads_reports["crash_id"].isin(ads.loc[ads["operator"] == "Waymo", "crash_id"])]
          .groupby("crash_id")["Reporting Entity"].apply(lambda s: (~s.str.contains("waymo", case=False, na=False)).any())
          .sum()))

# compare with Justin's files under the same rules (known injury, no conflict)
if model_file.exists():
    jm = pd.read_csv(model_file, dtype=str)
    jm["incident_year"] = pd.to_datetime(jm["Incident Date"].str.title(), format="%b-%Y").dt.year
    jm["Injury"] = jm["Injury"].astype(int)
    j = jm.groupby("incident_year")["Injury"].agg(known_injury="size", injured="sum")
    mine = year_table[year_table["incident_year"] != "All"].set_index("incident_year")[["known_injury", "injured"]]
    mine.index = mine.index.astype(int)
    both_years = mine.join(j, rsuffix="_ads_model", how="outer")
    for c in ["known_injury", "injured"]:
        check(f"ads_model.csv: {c} per year matches",
              bool((both_years[c] == both_years[f"{c}_ads_model"]).all()), True)

    # crashes where ads_model picked a different representative report (ties in clean_ads.py's sort)
    report_to_crash = ads_reports.set_index("Report ID")["crash_id"]
    justin_rep = pd.Series(jm["Report ID"].values, index=jm["Report ID"].map(report_to_crash).values)
    known = ads[ads["status"].isin(["injured", "non_injured"])].set_index("crash_id")
    differ = known.index[known["Report ID"] != justin_rep.reindex(known.index)]
    check("crashes where ads_model.csv picked a different report", len(differ), 10)
    report_my = pd.to_numeric(ads_reports.set_index("Report ID")["Model Year"], errors="coerce").astype("Float64")
    here_my = known.loc[differ, "model_year"].astype("Float64")
    other_my = pd.Series(report_my.reindex(justin_rep[differ].values).values, index=differ)
    same_my = (here_my == other_my).fillna(False) | (here_my.isna() & other_my.isna())
    my_differ = list(differ[~same_my.values])
    check("of those, crashes where the other report has a different model year", len(my_differ))
    for cid in my_differ:
        check(f"  model year here vs ads_model report ({cid})",
              f"{known.loc[cid, 'model_year']} vs {report_my[justin_rep[cid]]}")

    split_files = [proj_root / "data" / "processed" / "model_input" / f for f in ["train.csv", "validate.csv", "test.csv"]]
    if all(f.exists() for f in split_files):
        split = pd.concat([pd.read_csv(f, usecols=["Report ID", "Model Year"]) for f in split_files])
        split = split.rename(columns={"Model Year": "model_year_split"})
        both = known.reset_index()[["Report ID", "model_year"]].merge(split, on="Report ID")
        a, b = both["model_year"].astype("Float64"), both["model_year_split"].astype("Float64")
        same = (a == b).fillna(False) | (a.isna() & b.isna())
        check("model_input splits: crashes with the same report as ads_model", len(both))
        check("model_input splits: same report -> same model year", int((~same).sum()), 0)

checks = pd.DataFrame(checks)
checks.to_csv(out_folder / "injury_rate_checks.csv", index=False)
print("\nInjury rate checks:")
print(checks.to_string(index=False))
failed = checks[checks["pass"] == False]
assert failed.empty, failed
print("injury rate checks ok")

# injury rate plots (read back the csv so plot = csv)
RATE_NOTE = ("Injury rate = crashes with an alleged injury / crashes with known injury outcome (one row per crash). "
             "Error bars: Wilson 95% CI.\nAmong reported ADS crashes only: not a per-mile crash rate, "
             "and not evidence that newer vehicles or software are safer.")


def rate_points(ax, x, t, color, label, hollow, offset=0.0, show_n=True):
    x = np.asarray(x, dtype=float) + offset
    rate, lo, hi = t["injury_rate_pct"].values, t["ci95_low_pct"].values, t["ci95_high_pct"].values
    hollow = np.asarray(hollow)
    ax.errorbar(x, rate, yerr=[rate - lo, hi - rate], fmt="none", ecolor=color, elinewidth=1.2, capsize=3, alpha=0.8)
    ax.scatter(x[~hollow], rate[~hollow], color=color, s=40, zorder=3, label=label)
    ax.scatter(x[hollow], rate[hollow], facecolors="white", edgecolors=color, s=40, zorder=3,
               label=None if label is None else f"{label} (hollow)")
    if show_n:
        for xi, h, k, n in zip(x, hi, t["injured"], t["known_injury"]):
            ax.text(xi, h + 0.8, f"{k}/{n}", ha="center", fontsize=7, color=color)


# 1. incident year
t = pd.read_csv(out_folder / "injury_rate_by_incident_year.csv", keep_default_na=False)
overall = t[t["incident_year"] == "All"].iloc[0]
t = t[t["incident_year"] != "All"].copy()
t["incident_year"] = t["incident_year"].astype(int)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t["incident_year"], t["injury_rate_pct"], color="tab:blue", alpha=0.4)
rate_points(ax, t["incident_year"], t, "tab:blue", None, t["partial_year"])
ax.axhline(overall["injury_rate_pct"], color="gray", linestyle="--", linewidth=1,
           label=f"all years {overall['injury_rate_pct']:.1f}% ({overall['injured']}/{overall['known_injury']})")
ax.axvspan(2024.6, 2025.4, color="gray", alpha=0.1, label="2025: Third Amended SGO from Jun 16 (rules change mid-year)")
ax.scatter([], [], facecolors="white", edgecolors="tab:blue",
           label="partial year: " + ", ".join(f"{y} {n}" for y, n in zip(t["incident_year"], t["note"]) if "only" in n))
ax.set_xticks(t["incident_year"])
ax.set_xticklabels([f"{y}\n{u} unknown, {c} conflict" for y, u, c in zip(t["incident_year"], t["unknown"], t["conflict"])],
                   fontsize=8)
ax.set_xlabel("Incident year (labels: crashes left out of the rate)")
ax.set_ylabel("Injury rate (% of crashes with known injury)")
ax.set_ylim(0, max(t["ci95_high_pct"].max() + 4, 25))
ax.set_title("ADS crashes: injury rate by incident year (numbers = injured / known injury crashes)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
fig.text(0.01, 0.01, RATE_NOTE, fontsize=7.5, color="gray")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(fig_folder / "injury_rate_by_incident_year.png", dpi=150)
plt.close(fig)

# 2. model year
t = pd.read_csv(out_folder / "injury_rate_by_model_year.csv", dtype={"model_year_group": str})
overall = t[t["model_year_group"] == "All"].iloc[0]
unknown_my = t[t["model_year_group"] == "Unknown"]
t = t[~t["model_year_group"].isin(["All", "Unknown"])].copy()
t["model_year"] = t["model_year_group"].astype(int)
fig, ax = plt.subplots(figsize=(10, 6))
rate_points(ax, t["model_year"], t, "tab:purple", f"n >= {MIN_N}", t["small_sample"])
ax.axhline(overall["injury_rate_pct"], color="gray", linestyle="--", linewidth=1,
           label=f"all model years {overall['injury_rate_pct']:.1f}%")
for _, r in t[t["small_sample"]].iterrows():
    ax.text(r["model_year"], -2.5, "small n", ha="center", fontsize=7, color="tab:red")
ax.set_xticks(t["model_year"])
ax.set_xlabel("Vehicle model year (the car, not the ADS software version)")
ax.set_ylabel("Injury rate (% of crashes with known injury)")
ax.set_ylim(-4, t["ci95_high_pct"].max() + 6)
ax.set_title("ADS crashes: injury rate by vehicle model year (numbers = injured / known injury crashes)")
ax.grid(alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
labels = [l.replace(f"n >= {MIN_N} (hollow)", f"n < {MIN_N}: unstable") for l in labels]
ax.legend(handles, labels, loc="upper right", fontsize=8)
missing_note = ""
if len(unknown_my):
    u = unknown_my.iloc[0]
    missing_note = (f"Not shown: {u['total_crashes']} crashes with missing model year "
                    f"({u['injured']}/{u['known_injury']} injured). ")
fig.text(0.01, 0.01, missing_note + f"Unknown/conflict injury left out of the rate: {overall['unknown']} / "
         f"{overall['conflict']}. Model year comes from the crash's representative report.\n" + RATE_NOTE,
         fontsize=7.5, color="gray")
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(fig_folder / "injury_rate_by_model_year.png", dpi=150)
plt.close(fig)

# 3. All ADS vs Waymo vs non-Waymo
t = pd.read_csv(out_folder / "injury_rate_waymo_vs_all_by_year.csv")
t = t[(t["incident_year"] != "All") & t["group"].isin(["All ADS", "Waymo", "Non-Waymo"])].copy()
t["incident_year"] = t["incident_year"].astype(int)
partial_years = set(year_table.loc[year_table["partial_year"] == True, "incident_year"])
fig, ax = plt.subplots(figsize=(12, 6.5))
for g, color, off in [("All ADS", "black", -0.3), ("Waymo", "tab:green", 0.0), ("Non-Waymo", "tab:orange", 0.3)]:
    tg = t[t["group"] == g]
    n_all = op_all.loc[g, "total_crashes"]
    ax.plot(tg["incident_year"] + off, tg["injury_rate_pct"], color=color, alpha=0.35)
    rate_points(ax, tg["incident_year"], tg, color, f"{g} ({n_all} crashes)",
                tg["incident_year"].isin(partial_years), offset=off)
ax.axvspan(2024.6, 2025.4, color="gray", alpha=0.1, label="2025: Third Amended SGO from Jun 16")
ax.scatter([], [], facecolors="white", edgecolors="gray", label="hollow = partial year (2021 Jul-Dec, 2026 Jan-Aug)")
handles, labels = ax.get_legend_handles_labels()
keep = [i for i, l in enumerate(labels) if "(hollow)" not in l]
ax.legend([handles[i] for i in keep], [labels[i] for i in keep], loc="upper right", fontsize=8)
years = sorted(t["incident_year"].unique())
ax.set_xticks(years)
ax.set_xlabel("Incident year")
ax.set_ylabel("Injury rate (% of crashes with known injury)")
ax.set_ylim(0, t["ci95_high_pct"].max() + 8)
ax.set_title("ADS crashes: injury rate by incident year, All ADS vs Waymo vs non-Waymo operators\n"
             "(numbers = injured / known injury crashes)")
ax.grid(alpha=0.3)
others = ", ".join(f"{g}: {int(op_all.loc[g, 'total_crashes'])}" for g in ["Operator mixed", "Operator unknown"]
                   if g in op_all.index)
fig.text(0.01, 0.01, f"Operator = Operating Entity of every report of the crash (contains 'waymo', any case). "
         f"Only counted in All ADS: {others} crashes.\nWaymo is most of All ADS, so those two lines are not "
         f"independent.\n" + RATE_NOTE, fontsize=7.5, color="gray")
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(fig_folder / "injury_rate_waymo_vs_all.png", dpi=150)
plt.close(fig)
print("saved injury rate csv files and plots")
