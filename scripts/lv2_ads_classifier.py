import argparse
from math.integer import perm
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set up paths to the data and results folders relative to this script's location.
root = Path(__file__).resolve().parent.parent
data_folder = root / "data" / "processed"
results_folder = root / "results" / "models"
fig_folder = root / "docs" / "figures"

results_folder.mkdir(parents=True, exist_ok=True)
fig_folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE_TEST = 42
RANDOM_STATE_VALIDATION = 43
LABEL_ORDER = ["ads", "level2"]
LABEL_NAMES = ["ADS", "Level2"]

NUMERIC_FEATURES = [
    "SV Precrash Speed (MPH)",
    "Weather - Clear",
    "Weather - Cloudy",
    "Weather - Rain",
    "Weather - Severe Wind",
    "Weather - Snow",
    "Weather - Fog/Smoke/Haze Combined",
    "Wet Surface"
]

CATEGORICAL_FEATURES = [
    "Crash With",
    "Roadway Type",
    "SV Pre-Crash Movement"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Function for getting the features based on whether to include "Crash With" or not
def get_features(include_crash_with):
    """Return the predictors for the requested feature configuration."""
    categorical_features = CATEGORICAL_FEATURES.copy()
    if not include_crash_with:
        categorical_features.remove("Crash With")
    return NUMERIC_FEATURES.copy(), categorical_features

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ads", default="ads_model(1).csv", help="Path to ADS CSV")
    parser.add_argument(
        "--level2", default="level2_model(2).csv", help="Path to Level2 CSV"
    )
    return parser.parse_args()


def make_preprocessor(include_crash_with):
    # To include or exclude "Crash With" as a predictor
    numeric_features, categorical_features = get_features(include_crash_with)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def make_models():
    return {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE_TEST,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE_TEST,
            n_jobs=-1,
        ),
    }


def build_dataset():
    # Load the cleaned data for the ADS and LV2 models
    ads_df = pd.read_csv(data_folder / "ads_model.csv")
    level2_df = pd.read_csv(data_folder / "level2_model.csv")

    # Add a target column to each dataset to indicate the source of the data
    ads_df["target"] = "ads"
    level2_df["target"] = "level2"

    df = pd.concat([ads_df, level2_df], ignore_index=True)

    required_columns = {
        "Incident Date",
        "Report ID",
        "Same Incident ID",
        "target",
        "Crash With",
        "Roadway Type",
        "SV Precrash Speed (MPH)",
        "SV Pre-Crash Movement",
        "Weather - Clear",
        "Weather - Cloudy",
        "Weather - Rain",
        "Weather - Severe Wind",
        "Weather - Snow",
        "Weather - Fog/Smoke/Haze Combined",
        "Wet Surface",
        "Injury",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # A stable row order makes the fixed-seed split repeatable even if the CSVs
    # are loaded in a different row order.
    df = df.sort_values(["target", "Report ID"], kind="mergesort").reset_index(drop=True)

    # Parse the month-year date field and extract useful model features.
    incident_date = pd.to_datetime(
        df["Incident Date"].astype("string").str.title(),
        format="%b-%Y",
        errors="coerce",
    )
    df["Incident_Year"] = incident_date.dt.year
    df["Incident_Month"] = incident_date.dt.month.apply(
        lambda value: str(int(value)) if pd.notna(value) else np.nan
    )

    # Use Same Incident ID only to keep related rows in one split, never as a
    # predictor. Blank IDs get unique row-level groups rather than one shared
    # "blank" group.
    incident_ids = df["Same Incident ID"].fillna("").astype(str).str.strip()
    row_groups = "row:" + df["target"].astype(str) + ":" + df["Report ID"].astype(str)
    groups = pd.Series(
        np.where(incident_ids.ne(""), "incident:" + incident_ids, row_groups),
        index=df.index,
    )

    # Drop identifier columns from predictors to avoid memorizing record IDs.
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[all_features].copy()
    y = df["target"].copy()
    return X, y, groups


def print_class_distribution(name, labels):
    counts = labels.value_counts().reindex(LABEL_ORDER, fill_value=0)
    print(f"{name}: {len(labels):,} rows | ADS={counts['ads']:,} | Level2={counts['level2']:,}")

def make_pipeline(estimator, include_crash_with):
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(include_crash_with)),
            ("model", estimator),
        ]
    )

def pr_auc_range(y_true, probabilities, n=2000):
    # 95% range for PR-AUC
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    rng = np.random.default_rng(0)
    scores = []

    for _ in range(n):
        indices = rng.integers(0, len(y_true), len(y_true))
        scores.append(
            average_precision_score(
                y_true[indices],
                probabilities[indices],
            )
        )

    return np.percentile(scores, [2.5, 97.5])

X, y, groups = build_dataset()

# First reserve the untouched 20% test set. StratifiedGroupKFold keeps
# class proportions similar while preventing incident groups crossing sets.
test_splitter = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE_TEST
)
development_idx, test_idx = next(test_splitter.split(X, y, groups))

X_development = X.iloc[development_idx]
y_development = y.iloc[development_idx]
groups_development = groups.iloc[development_idx]
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# Split the remaining 80% into 60% train and 20% validation overall.
validation_splitter = StratifiedGroupKFold(
    n_splits=4, shuffle=True, random_state=RANDOM_STATE_VALIDATION
)
train_relative_idx, validation_relative_idx = next(
    validation_splitter.split(
        X_development, y_development, groups_development
    )
)

X_train = X_development.iloc[train_relative_idx]
y_train = y_development.iloc[train_relative_idx]
X_validation = X_development.iloc[validation_relative_idx]
y_validation = y_development.iloc[validation_relative_idx]

# Verify the group-aware split did its job.
train_groups = set(groups.iloc[development_idx[train_relative_idx]])
validation_groups = set(groups.iloc[development_idx[validation_relative_idx]])
test_groups = set(groups.iloc[test_idx])
assert train_groups.isdisjoint(validation_groups)
assert train_groups.isdisjoint(test_groups)
assert validation_groups.isdisjoint(test_groups)

print("\nDATA SPLIT")
print_class_distribution("Training", y_train)
print_class_distribution("Validation", y_validation)
print_class_distribution("Testing", y_test)
print("\nTest split is reproducible for the same input CSV contents.\n")

# Run model comparison separately with and without Crash With.
validation_rows = []
test_rows = []
report_rows = []
importance_rows = []

for include_crash_with in [True, False]:
    feature_label = "with Crash With" if include_crash_with else "without Crash With"
    print(f"\n{'=' * 72}\nRUNNING MODELS {feature_label.upper()}\n{'=' * 72}")

    fitted_models = {}
    validation_scores = {}

    for model_name, estimator in make_models().items():
        pipeline = make_pipeline(estimator, include_crash_with)
        pipeline.fit(X_train, y_train)

        validation_predictions = pipeline.predict(X_validation)
        macro_f1 = f1_score(
            y_validation,
            validation_predictions,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=0,
        )

        validation_scores[model_name] = macro_f1
        fitted_models[model_name] = pipeline
        validation_rows.append(
            {
                "feature configuration": feature_label,
                "model": model_name,
                "validation macro-F1": macro_f1,
            }
        )
        print(f"{model_name:22s} validation macro-F1 = {macro_f1:.4f}")

    # Choose the model using validation macro-F1 only.
    best_model_name = max(validation_scores, key=validation_scores.get)
    print(f"\nSelected model: {best_model_name}")

    validation_predictions = fitted_models[best_model_name].predict(X_validation)
    print(f"\nVALIDATION CLASSIFICATION REPORT — {feature_label} — {best_model_name}")
    print(
        classification_report(
            y_validation,
            validation_predictions,
            labels=LABEL_ORDER,
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    # Refit selected model using train + validation, then evaluate held-out test.
    X_train_validation = pd.concat([X_train, X_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)

    final_model = make_pipeline(
        make_models()[best_model_name], include_crash_with
    )
    final_model.fit(X_train_validation, y_train_validation)

    test_predictions = final_model.predict(X_test)
    test_report = classification_report(
        y_test,
        test_predictions,
        labels=LABEL_ORDER,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print(f"TEST CLASSIFICATION REPORT — {feature_label} — {best_model_name}")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=LABEL_ORDER,
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

        # Evaluate the selected model on the held-out test set.
    test_predictions = final_model.predict(X_test)

    # Treat Level2 as the positive class and ADS as the negative class.
    y_test_binary = (y_test == "level2").astype(int)

    # Get the probability assigned to Level2.
    model_classes = list(final_model.named_steps["model"].classes_)
    level2_index = model_classes.index("level2")
    test_probabilities = final_model.predict_proba(X_test)[:, level2_index]

    # Calculate PR-AUC and its bootstrap 95% range.
    pr_auc = average_precision_score(y_test_binary, test_probabilities)
    pr_low, pr_high = pr_auc_range(y_test_binary, test_probabilities)

    # Calculate ROC-AUC.
    roc_auc = roc_auc_score(y_test_binary, test_probabilities)

    test_report = classification_report(
        y_test,
        test_predictions,
        labels=LABEL_ORDER,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print(f"TEST CLASSIFICATION REPORT — {feature_label} — {best_model_name}")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=LABEL_ORDER,
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    print(f"PR-AUC: {pr_auc:.4f} ({pr_low:.4f}–{pr_high:.4f})")
    print(f"ROC-AUC: {roc_auc:.4f}")

    test_rows.append(
        {
            "feature configuration": feature_label,
            "selected model": best_model_name,
            "validation macro-F1": validation_scores[best_model_name],
            "test accuracy": accuracy_score(y_test, test_predictions),
            "test macro precision": test_report["macro avg"]["precision"],
            "test macro recall": test_report["macro avg"]["recall"],
            "test macro-F1": test_report["macro avg"]["f1-score"],
            "PR-AUC": pr_auc,
            "PR-AUC 95% low": pr_low,
            "PR-AUC 95% high": pr_high,
            "ROC-AUC": roc_auc,
        }
    )

    # Store per-class test metrics in a tidy CSV.
    for class_name in LABEL_NAMES:
        report_rows.append(
            {
                "feature configuration": feature_label,
                "selected model": best_model_name,
                "class": class_name,
                "precision": test_report[class_name]["precision"],
                "recall": test_report[class_name]["recall"],
                "f1-score": test_report[class_name]["f1-score"],
                "support": test_report[class_name]["support"],
            }
        )
    
    # --------------------------------
    # Random Forest permutation feature importance
    # --------------------------------

    # Fit a separate Random Forest for feature importance.
    rf_pipeline = make_pipeline(
        make_models()["Random Forest"], include_crash_with
    )
    rf_pipeline.fit(X_train_validation, y_train_validation)

    # Get the original input features for this configuration.
    numeric_features, categorical_features = get_features(include_crash_with)
    importance_features = numeric_features + categorical_features

    # Calculate permutation importance on the held-out test set.
    perm = permutation_importance(
        rf_pipeline,
        X_test[importance_features],
        y_test,
        scoring="f1_macro",
        n_repeats=20,
        random_state=0,
        n_jobs=-1,
    )

    # Store the results.
    for feature, mean_importance, std_importance in zip(
        importance_features,
        perm.importances_mean,
        perm.importances_std,
    ):
        importance_rows.append(
            {
                "feature configuration": feature_label,
                "feature": feature,
                "macro-F1 drop": mean_importance,
                "importance std": std_importance,
            }
        )

# Save all result tables.
validation_results = pd.DataFrame(validation_rows).sort_values(
    ["feature configuration", "validation macro-F1"],
    ascending=[True, False],
)
test_results = pd.DataFrame(test_rows)
classification_results = pd.DataFrame(report_rows)
feature_importance = pd.DataFrame(importance_rows).sort_values(
    ["feature configuration", "macro-F1 drop"],
    ascending=[True, False],
)

validation_results.to_csv(
    results_folder / "lv2_ads_validation_comparison.csv", index=False
)
test_results.to_csv(
    results_folder / "lv2_ads_model_results.csv", index=False
)
classification_results.to_csv(
    results_folder / "lv2_ads_test_classification_reports.csv", index=False
)
feature_importance.to_csv(
    results_folder / "lv2_ads_rf_feature_importance.csv", index=False
)

# Descriptive feature profiles show the direction of differences between sources.
# Unlike permutation importance, these tables show values/proportions by class.
profile_rows = []
for feature in NUMERIC_FEATURES:
    numeric_values = pd.to_numeric(X[feature], errors="coerce")
    for statistic, function in [
        ("mean", "mean"),
        ("median", "median"),
        ("missing proportion", "missing"),
    ]:
        if function == "missing":
            values = numeric_values.isna().groupby(y).mean()
        else:
            values = numeric_values.groupby(y).agg(function)
        profile_rows.append(
            {
                "feature": feature,
                "value/category": statistic,
                "ADS": values.get("ads", np.nan),
                "Level2": values.get("level2", np.nan),
                "profile type": "numeric summary",
            }
        )

for feature in CATEGORICAL_FEATURES:
    values = X[feature].astype("string").fillna("<MISSING>")
    proportions = pd.crosstab(y, values, normalize="index")
    for category in proportions.columns:
        profile_rows.append(
            {
                "feature": feature,
                "value/category": str(category),
                "ADS": proportions.loc["ads", category]
                if "ads" in proportions.index else 0.0,
                "Level2": proportions.loc["level2", category]
                if "level2" in proportions.index else 0.0,
                "profile type": "category proportion",
            }
        )

feature_profiles = pd.DataFrame(profile_rows)

print("\nTEST SUMMARY")
print(test_results.round(4).to_string(index=False))

print("\nRANDOM FOREST PERMUTATION IMPORTANCE")
print(feature_importance.round(5).to_string(index=False))

# Plot grouped feature importance for the two configurations.
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)
for ax, feature_label in zip(
    axes, ["with Crash With", "without Crash With"]
):
    subset = feature_importance[
        feature_importance["feature configuration"] == feature_label
    ].sort_values("macro-F1 drop")
    ax.barh(subset["feature"], subset["macro-F1 drop"])
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_title(feature_label)
    ax.set_xlabel("Decrease in test macro-F1 when shuffled")
    ax.grid(axis="x", alpha=0.3)

fig.suptitle("ADS vs Level2: Random Forest permutation feature importance")
fig.tight_layout()
fig.savefig(
    fig_folder / "lv2_ads_rf_feature_importance.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

print(
    "\nSaved results to results/models/: "
    "lv2_ads_validation_comparison.csv, "
    "lv2_ads_model_results.csv, "
    "lv2_ads_test_classification_reports.csv, "
    "lv2_ads_rf_feature_importance.csv, "
)
print("Saved feature-importance plot to docs/figures/lv2_ads_rf_feature_importance.png")

print("\nMARKDOWN RESULTS TABLE")

print("| Model | Crash With | PR-AUC (95% range) | ROC-AUC | Macro-F1 |")
print("|---|---|---|---|---|")

for _, row in test_results.iterrows():
    model = row["selected model"]
    crash_with = (
        "with" if row["feature configuration"] == "with Crash With"
        else "without"
    )
    pr_auc = (
        f'{row["PR-AUC"]:.3f} '
        f'({row["PR-AUC 95% low"]:.3f}–{row["PR-AUC 95% high"]:.3f})'
    )

    print(
        f'| {model} | {crash_with} | {pr_auc} | '
        f'{row["ROC-AUC"]:.3f} | {row["test macro-F1"]:.3f} |'
    )