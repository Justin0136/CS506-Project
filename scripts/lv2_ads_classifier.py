import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Locate the raw data folder
data_folder = Path(__file__).resolve().parent.parent / "data" / "processed"

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ads", default="ads_model(1).csv", help="Path to ADS CSV")
    parser.add_argument(
        "--level2", default="level2_model(2).csv", help="Path to Level2 CSV"
    )
    return parser.parse_args()


def make_preprocessor():
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
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
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
    X = df[FEATURES].copy()
    y = df["target"].copy()
    return X, y, groups


def print_class_distribution(name, labels):
    counts = labels.value_counts().reindex(LABEL_ORDER, fill_value=0)
    print(f"{name}: {len(labels):,} rows | ADS={counts['ads']:,} | Level2={counts['level2']:,}")



args = parse_args()
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

# Compare candidate models on validation data only.
fitted_models = {}
validation_scores = {}

print("VALIDATION MODEL COMPARISON (higher macro-F1 is better)")
for model_name, estimator in make_models().items():
    pipeline = Pipeline(
        steps=[("preprocess", make_preprocessor()), ("model", estimator)]
    )
    pipeline.fit(X_train, y_train)
    validation_predictions = pipeline.predict(X_validation)
    macro_f1 = f1_score(
        y_validation, validation_predictions, labels=LABEL_ORDER, average="macro"
    )
    validation_scores[model_name] = macro_f1
    fitted_models[model_name] = pipeline
    print(f"{model_name:22s} macro-F1 = {macro_f1:.4f}")

best_model_name = max(validation_scores, key=validation_scores.get)
best_validation_model = fitted_models[best_model_name]

# Print the validation report for the model selected using validation data.
validation_predictions = best_validation_model.predict(X_validation)
print(f"\nVALIDATION CLASSIFICATION REPORT — {best_model_name}")
print(
    classification_report(
        y_validation,
        validation_predictions,
        labels=LABEL_ORDER,
        target_names=LABEL_NAMES,
        zero_division=0,
    )
)

# Refit the chosen model on all non-test data, then evaluate the test set
# exactly once. The test set is not used for model selection.
final_model = Pipeline(
    steps=[
        ("preprocess", make_preprocessor()),
        ("model", make_models()[best_model_name]),
    ]
)
X_train_validation = pd.concat([X_train, X_validation], axis=0)
y_train_validation = pd.concat([y_train, y_validation], axis=0)
final_model.fit(X_train_validation, y_train_validation)

test_predictions = final_model.predict(X_test)
print(f"TEST CLASSIFICATION REPORT — final {best_model_name}")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=LABEL_ORDER,
        target_names=LABEL_NAMES,
        zero_division=0,
    )
)

print(f"Selected model: {best_model_name}")
print(f"Test accuracy: {(test_predictions == y_test).mean():.4f}")

