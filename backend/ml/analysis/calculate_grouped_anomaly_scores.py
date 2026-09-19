import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

BEHAVIORAL_DIR = Path("ml/dataset/behavioral")

REFERENCE_FILE = Path(
    "ml/dataset/reference/normal_reference_stats.csv"
)

OUTPUT_DIR = Path("ml/dataset/anomaly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR / "grouped_anomaly_scores.csv"
)


# ==========================================
# FEATURE GROUPS
# ==========================================

# Movement features
MOVEMENT_FEATURES = [
    "avg_speed_mean",
    "avg_speed_std",
    "avg_speed_change_mean",
    "avg_velocity_change_mean",
]

# Structure / crowd organization features
STRUCTURE_FEATURES = [
    "people_count_mean",
    "people_count_std",
    "avg_density_mean",
    "direction_alignment_mean",
    "directional_disorder_mean",
]


# ==========================================
# LOAD REFERENCE
# ==========================================

print("\n==========================================")
print("    GROUPED ANOMALY SCORE CALCULATION")
print("==========================================")

if not REFERENCE_FILE.exists():
    raise FileNotFoundError(
        f"Reference file not found: {REFERENCE_FILE}"
    )

reference = pd.read_csv(
    REFERENCE_FILE
).set_index("feature")


# ==========================================
# LOAD BEHAVIORAL DATA
# ==========================================

exclude_files = {
    "training_behavioral.csv",
    "crowd_features_behavioral.csv",
    "stampede_video_features_behavioral.csv",
    "training_features_behavioral.csv",
}

files = [
    f
    for f in BEHAVIORAL_DIR.glob("*_behavioral.csv")
    if f.name not in exclude_files
]

if not files:
    raise RuntimeError(
        "No behavioral datasets found."
    )

datasets = []

print("\n===== LOADING DATASETS =====")

for file in sorted(files):

    data = pd.read_csv(file)

    data["source"] = file.stem

    datasets.append(data)

    print(
        f"{file.name:<50}"
        f"{len(data):>3} windows"
    )


df = pd.concat(
    datasets,
    ignore_index=True
)

print("\nTotal windows:", len(df))


# ==========================================
# FEATURE CHECK
# ==========================================

ALL_FEATURES = (
    MOVEMENT_FEATURES
    + STRUCTURE_FEATURES
)

missing_features = [
    feature
    for feature in ALL_FEATURES
    if feature not in df.columns
]

if missing_features:

    print("\nMissing features:")

    for feature in missing_features:
        print(" -", feature)

    raise RuntimeError(
        "Required features are missing."
    )


# ==========================================
# ROBUST DEVIATION FUNCTION
# ==========================================

def calculate_deviation(data, features):

    deviations = []

    for feature in features:

        median = reference.loc[
            feature,
            "median"
        ]

        iqr = reference.loc[
            feature,
            "iqr"
        ]

        values = pd.to_numeric(
            data[feature],
            errors="coerce"
        )

        deviation = (
            (values - median).abs()
            / iqr
        )

        deviations.append(
            deviation.rename(feature)
        )

    result = pd.concat(
        deviations,
        axis=1
    )

    return result


# ==========================================
# MOVEMENT SCORE
# ==========================================

print("\n===== MOVEMENT SCORE =====")

movement_deviation = calculate_deviation(
    df,
    MOVEMENT_FEATURES
)

df["movement_anomaly_score"] = (
    movement_deviation.mean(axis=1)
)


# ==========================================
# STRUCTURE SCORE
# ==========================================

print("\n===== STRUCTURE SCORE =====")

structure_deviation = calculate_deviation(
    df,
    STRUCTURE_FEATURES
)

df["structure_anomaly_score"] = (
    structure_deviation.mean(axis=1)
)


# ==========================================
# FINAL GROUPED SCORE
# ==========================================

# Equal importance between the two groups.
df["grouped_anomaly_score"] = (
    0.5 * df["movement_anomaly_score"]
    + 0.5 * df["structure_anomaly_score"]
)


# ==========================================
# CLEAN INVALID VALUES
# ==========================================

score_columns = [
    "movement_anomaly_score",
    "structure_anomaly_score",
    "grouped_anomaly_score",
]

df[score_columns] = (
    df[score_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ==========================================
# SORT
# ==========================================

df = df.sort_values(
    "grouped_anomaly_score",
    ascending=False
).reset_index(drop=True)


# ==========================================
# TOP ANOMALIES
# ==========================================

print("\n==========================================")
print("       TOP GROUPED ANOMALIES")
print("==========================================")

display_columns = [
    "source",
    "window_id",
    "people_count_mean",
    "avg_speed_mean",
    "directional_disorder_mean",
    "movement_anomaly_score",
    "structure_anomaly_score",
    "grouped_anomaly_score",
]

print(
    df[display_columns]
    .head(25)
    .round(3)
    .to_string(index=False)
)


# ==========================================
# DATASET SUMMARY
# ==========================================

print("\n==========================================")
print("      SCORE BY DATASET")
print("==========================================")

summary = (
    df.groupby("source")
    [
        [
            "movement_anomaly_score",
            "structure_anomaly_score",
            "grouped_anomaly_score",
        ]
    ]
    .agg(
        ["mean", "median", "max"]
    )
)

print(
    summary
    .round(3)
    .to_string()
)


# ==========================================
# SAVE
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==========================================")
print("       GROUPED SCORING COMPLETE")
print("==========================================")

print("Saved to:")
print(OUTPUT_FILE)

print("==========================================")