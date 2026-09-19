from pathlib import Path
import pandas as pd


# ==============================
# PATH
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = BASE_DIR / "ml" / "dataset" / "labeled"


# ==============================
# LOAD ALL LABELED DATA
# ==============================

files = list(INPUT_DIR.glob("*_labeled.csv"))

datasets = []

for file in files:

    df = pd.read_csv(file)

    # Identify source dataset
    df["source"] = file.stem.replace(
        "_features_labeled",
        ""
    )

    datasets.append(df)


if not datasets:
    raise FileNotFoundError(
        "No labeled datasets found."
    )


data = pd.concat(
    datasets,
    ignore_index=True
)


# ==============================
# BASIC INFORMATION
# ==============================

print("=" * 60)
print("LABELED DATASET ANALYSIS")
print("=" * 60)

print("Total samples:", len(data))
print("Total features:", len(data.columns))


# ==============================
# RISK DISTRIBUTION
# ==============================

print("\n" + "=" * 60)
print("RISK DISTRIBUTION")
print("=" * 60)

print(
    data["risk_state"]
    .value_counts()
)

print("\nNumerical labels:")

print(
    data["risk_label"]
    .value_counts()
    .sort_index()
)


# ==============================
# DISTRIBUTION BY DATASET
# ==============================

print("\n" + "=" * 60)
print("RISK DISTRIBUTION BY DATASET")
print("=" * 60)

print(
    pd.crosstab(
        data["source"],
        data["risk_state"]
    )
)


# ==============================
# BEHAVIOR SCORE BY RISK
# ==============================

print("\n" + "=" * 60)
print("BEHAVIOR SCORE BY RISK")
print("=" * 60)

score_summary = (
    data.groupby("risk_state")
    ["abnormal_behavior_score"]
    .agg(
        [
            "count",
            "mean",
            "min",
            "max",
            "std"
        ]
    )
)

print(score_summary)


# ==============================
# FEATURE COMPARISON
# ==============================

features = [
    "avg_speed_mean",
    "avg_acceleration_mean",
    "avg_displacement_mean",
    "avg_speed_change_mean",
    "avg_velocity_change_mean",
    "direction_alignment_mean",
    "directional_disorder_mean",
    "avg_density_mean",
]


print("\n" + "=" * 60)
print("FEATURE MEANS BY RISK STATE")
print("=" * 60)

available_features = [
    feature
    for feature in features
    if feature in data.columns
]

feature_summary = (
    data.groupby("risk_state")
    [available_features]
    .mean()
)

print(feature_summary)


# ==============================
# HIGH-RISK / CRITICAL WINDOWS
# ==============================

print("\n" + "=" * 60)
print("NON-NORMAL WINDOWS")
print("=" * 60)

non_normal = data[
    data["risk_label"] > 0
][
    [
        "source",
        "window_start",
        "window_end",
        "abnormal_behavior_score",
        "risk_state",
    ]
]

print(
    non_normal.to_string(index=False)
)


print("\n" + "=" * 60)
print("✓ ANALYSIS COMPLETE")
print("=" * 60)