from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# features used by injury models
# left out incident year since every 2026 test would have a year the model was not trained on
categorical_features = [
    "Crash With",
    "Roadway Type",
    "SV Pre-Crash Movement",
]

numerical_features = [
    "SV Precrash Speed (MPH)",
    "Weather - Clear",
    "Weather - Cloudy",
    "Weather - Rain",
    "Weather - Severe Wind",
    "Weather - Snow",
    "Weather - Fog/Smoke/Haze Combined",
    "Wet Surface",
    "Model Year",
    "Vehicle Age"
]

# encode features inside model pipeline so it only learns from training data
# !! with_crash_partner = False to leave out Crash With feature

def build_preprocessor(with_crash_partner=True):
    categorical = categorical_features
    if not with_crash_partner:
        categorical = [col for col in categorical_features if col != "Crash With"]
 
    # Fill missing values with "Unknown" and group categories with fewer than 10 crashes
    categorical_steps = Pipeline([
        ("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10))
    ])
 
    # Fill missing numbers with the median and add a column marking which were missing
    numerical_steps = SimpleImputer(strategy="median", add_indicator=True)
 
    return ColumnTransformer([
        ("categorical", categorical_steps, categorical),
        ("numeric", numerical_steps, numerical_features)
    ])