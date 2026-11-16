# Train and evaluate ADS injury models
#
# Models: baseline (always "no injury"), logistic regression, random forest.
# Each model is run with and without the Crash With feature.
# 1. pick settings on the 2025 validation set (trained on 2021-2024)
# 2. retrain on 2021-2025 and score once on the 2026 test set

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve

from model_features import build_preprocessor

from xgboost import XGBClassifier

root = Path(__file__).resolve().parent.parent
data_folder = root / "data" / "processed" / "model_input"
results_folder = root / "results" / "models"
fig_folder = root / "docs" / "figures"
results_folder.mkdir(parents=True, exist_ok=True)
fig_folder.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(data_folder / "train.csv")
validate = pd.read_csv(data_folder / "validate.csv")
test = pd.read_csv(data_folder / "test.csv")
train_val = pd.concat([train, validate], ignore_index=True)

# models and the settings to try for each
MODELS = {
    "Baseline (no injury)": (lambda: DummyClassifier(strategy="prior"), [{}]),
    "Logistic regression": (
        lambda **p: LogisticRegression(class_weight="balanced", max_iter=5000, **p),
        [{"C": c} for c in [0.01, 0.1, 1, 10]],
    ),
    "Random forest": (
        lambda **p: RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                           random_state=0, n_jobs=-1, **p),
        [{"min_samples_leaf": m, "max_depth": d} for m in [1, 5, 10, 20] for d in [None, 5, 10]],
    ),
    "XGBoost": (
        lambda **p: XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=500,
            tree_method="hist",
            random_state=0,
            n_jobs=-1,
            verbosity=0,
            **p,
        ),
        [
            {
                "max_depth": depth,
                "learning_rate": learning_rate,
                "min_child_weight": min_child_weight,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
            # Parameters to pick the best from on the 2025 validation set for XGBoost
            for depth in [3, 5]
            for learning_rate in [0.03, 0.1]
            for min_child_weight in [1, 5]
        ],
    ),
}


def make_pipeline(model, with_crash_partner):
    steps = [("prep", build_preprocessor(with_crash_partner))]
    if isinstance(model, LogisticRegression):
        # logistic regression needs features on a similar scale (e.g. model year ~2020 vs 0/1 flags)
        steps.append(("scale", StandardScaler(with_mean=False)))
    steps.append(("model", model))
    return Pipeline(steps)


def pr_auc_range(y, prob, n=2000):
    """95% range of PR-AUC when the test crashes are resampled (the test set has few injuries)."""
    y, prob = np.asarray(y), np.asarray(prob)
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        scores.append(average_precision_score(y[i], prob[i]))
    return np.percentile(scores, [2.5, 97.5])


def score(y, prob):
    """PR-AUC (main metric) with its 95% range, ROC-AUC, and macro-F1 at a 0.5 cutoff."""
    low, high = pr_auc_range(y, prob)
    return {
        "PR-AUC": average_precision_score(y, prob),
        "PR-AUC 95% low": low,
        "PR-AUC 95% high": high,
        "ROC-AUC": roc_auc_score(y, prob),
        "macro-F1": f1_score(y, (prob >= 0.5).astype(int), average="macro"),
    }

# Need to make a direct make_model function for XGBoost since it needs to calculate class weights from the training labels. 
# All other models are unaffected
def make_model_for_training(name, model_factory, settings, y_train):
    """Create a model, calculating XGBoost's class weight from this fit's labels."""
    params = dict(settings)
    # Only XGBoost uses this to weight the positive class when the training labels are imbalanced. The other models use class_weight="balanced" in their constructors.
    if name == "XGBoost":
        y_array = np.asarray(y_train)
        n_positive = np.count_nonzero(y_array == 1)
        n_negative = np.count_nonzero(y_array == 0)
        if n_positive == 0 or n_negative == 0:
            raise ValueError("XGBoost requires both injury (1) and no-injury (0) examples.")
        # Up-weight injury examples when the target classes are imbalanced.
        params["scale_pos_weight"] = n_negative / n_positive
    return model_factory(**params)

rows, curves, fitted = [], {}, {}
generalization_rows = []

for with_cp in [True, False]:
    label = "with Crash With" if with_cp else "without Crash With"

    for name, (make_model, settings) in MODELS.items():
        # 1. pick the setting with the best PR-AUC on 2025
        best = None
        for s in settings:
            # See comments above make_model_for_training() but TLDR: this is for XGBoost, other models are unaffected
            model = make_model_for_training(name, make_model, s, train["Injury"])
            pipe = make_pipeline(model, with_cp).fit(train, train["Injury"])
            val_prob = pipe.predict_proba(validate)[:, 1]
            val_pr = average_precision_score(validate["Injury"], val_prob)
            if best is None or val_pr > best[1]:
                best = (s, val_pr)

        # Refit the selected configuration on the training set to check for overfitting and generalization to the validation set. This is not used for model selection, which is done above.
        best_model = make_model_for_training(
            name, make_model, best[0], train["Injury"]
        )

        best_pipe = make_pipeline(best_model, with_cp).fit(
            train, train["Injury"]
        )

        # Calculate PR-AUC on the training data
        train_prob = best_pipe.predict_proba(train)[:, 1]
        train_pr = average_precision_score(train["Injury"], train_prob)

        # Calculate PR-AUC on the validation data
        val_prob = best_pipe.predict_proba(validate)[:, 1]
        val_pr = average_precision_score(validate["Injury"], val_prob)

        # Save results for plotting
        generalization_rows.append({
            "model": name,
            "features": label,
            "training PR-AUC": train_pr,
            "validation PR-AUC": val_pr
        })

        # 2. retrain on 2021-2025 with that setting, score once on 2026
            # See comments above make_model_for_training() but TLDR: this is for XGBoost, other models are unaffected
        final_model = make_model_for_training(name, make_model, best[0], train_val["Injury"])
        pipe = make_pipeline(final_model, with_cp).fit(train_val, train_val["Injury"])
        test_prob = pipe.predict_proba(test)[:, 1]
        rows.append({"model": name, "features": label, "best setting": str(best[0]),
                     "val PR-AUC": best[1], **score(test["Injury"], test_prob)})
        curves[(name, label)] = precision_recall_curve(test["Injury"], test_prob)
        fitted[(name, label)] = pipe

# Training vs validation PR-AUC plot

generalization_results = pd.DataFrame(generalization_rows)

fig, ax = plt.subplots(figsize=(12, 6))

# Create positions for grouped bars
x = np.arange(len(generalization_results))
width = 0.35

# Training PR-AUC bars
ax.bar(
    x - width / 2,
    generalization_results["training PR-AUC"],
    width,
    label="Training PR-AUC"
)

# Validation PR-AUC bars
ax.bar(
    x + width / 2,
    generalization_results["validation PR-AUC"],
    width,
    label="Validation PR-AUC"
)

# Label each model and feature configuration
labels = [
    f"{row['model']}\n{row['features']}"
    for _, row in generalization_results.iterrows()
]

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")

ax.set_ylabel("PR-AUC")
ax.set_title("ADS injury models: training vs validation PR-AUC")

ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

# Save in the existing figures folder
fig.savefig(
    fig_folder / "training_vs_validation_pr_auc.png",
    dpi=150
)

plt.close(fig)

results = pd.DataFrame(rows).round(3)
results.to_csv(results_folder / "model_results.csv", index=False)
print(results.to_string(index=False))

# precision-recall curves on 2026, one panel per feature set
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, label in zip(axes, ["with Crash With", "without Crash With"]):
    for name in MODELS:
        pr_auc = results.query("model == @name and features == @label")["PR-AUC"].item()
        if name.startswith("Baseline"):
            # the baseline gives every crash the same score, so its precision is flat at the injury rate
            ax.axhline(test["Injury"].mean(), color="gray", linestyle="--",
                       label=f"{name} (PR-AUC {pr_auc:.2f})")
            continue
        precision, recall, _ = curves[(name, label)]
        ax.plot(recall, precision, label=f"{name} (PR-AUC {pr_auc:.2f})")
    ax.set_title(f"2026 test set, {label}")
    ax.set_xlabel("Recall (share of injury crashes caught)")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("Precision (share of flagged crashes that were injuries)")
fig.suptitle("ADS injury models: precision-recall curves")
fig.tight_layout()
fig.savefig(fig_folder / "pr_curves.png", dpi=150)

# random forest feature importance (with Crash With), measured on 2026:
# how much PR-AUC drops when one feature's values are shuffled
rf = fitted[("Random forest", "with Crash With")]
perm = permutation_importance(rf, test, test["Injury"], scoring="average_precision",
                              n_repeats=20, random_state=0)
importance = pd.Series(perm.importances_mean, index=test.columns)
importance = importance.drop(["Injury"]).loc[lambda x: x != 0].sort_values()
importance.to_csv(results_folder / "rf_feature_importance.csv", header=["PR-AUC drop"])

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(importance.index, importance.values)
ax.axvline(0, color="gray", linewidth=1)
ax.set_title("Random forest feature importance on 2026 (with Crash With)")
ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
fig.tight_layout()
fig.savefig(fig_folder / "rf_feature_importance.png", dpi=150)
print("\nsaved results/models/, docs/figures/pr_curves.png, docs/figures/rf_feature_importance.png")
