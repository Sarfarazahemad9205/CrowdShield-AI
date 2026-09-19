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

OUTPUT_FILE = OUTPUT_DIR / "all_anomaly_scores.csv"


# ==========================================
# FEATURES
# ==========================================

FEATURES = [
    "people_count_mean",
    "people_count_std",
    "avg_density_mean",
    "avg_speed_mean",
    "avg_speed_std",
    "avg_displacement_mean",
    "direction_alignment_mean",
    "directional_disorder_mean",
    "avg_speed_change_mean",
    "avg_velocity_change_mean",
]


# ==========================================
# LOAD REFERENCE
# ==========================================

print("\n==========================================")
print("       CALCULATING ANOMALY SCORES")
print("==========================================")

if not REFERENCE_FILE.exists():
    raise FileNotFoundError(
        f"Reference file not found: {REFERENCE_FILE}"
    )

reference = pd.read_csv(REFERENCE_FILE)

reference = reference.set_index("feature")


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

    df = pd.read_csv(file)

    df["source"] = file.stem

    datasets.append(df)

    print(
        f"{file.name:<50} "
        f"{len(df):>3} windows"
    )


df = pd.concat(
    datasets,
    ignore_index=True
)

print("\nTotal windows:", len(df))


# ==========================================
# CHECK FEATURES
# ==========================================

missing = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing:
    print("\nMissing features:")

    for feature in missing:
        print(" -", feature)

    raise RuntimeError(
        "Required features are missing."
    )


# ==========================================
# CALCULATE ROBUST DEVIATIONS
# ==========================================

print("\n===== CALCULATING FEATURE DEVIATIONS =====")

deviation_columns = []

for feature in FEATURES:

    median = reference.loc[feature, "median"]
    iqr = reference.loc[feature, "iqr"]

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    deviation = (
        (values - median).abs()
        / iqr
    )

    column_name = f"{feature}_deviation"

    df[column_name] = deviation

    deviation_columns.append(column_name)


# ==========================================
# HANDLE INVALID VALUES
# ==========================================

df[deviation_columns] = (
    df[deviation_columns]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ==========================================
# ANOMALY SCORE
# ==========================================

df["anomaly_score"] = (
    df[deviation_columns]
    .mean(axis=1)
)


# ==========================================
# SORT BY ANOMALY
# ==========================================

df = df.sort_values(
    "anomaly_score",
    ascending=False
).reset_index(drop=True)


# ==========================================
# DISPLAY TOP ANOMALIES
# ==========================================

print("\n==========================================")
print("          TOP ANOMALOUS WINDOWS")
print("==========================================")

display_columns = [
    "source",
    "window_id",
    "people_count_mean",
    "avg_speed_mean",
    "avg_displacement_mean",
    "directional_disorder_mean",
    "anomaly_score",
]

print(
    df[display_columns]
    .head(20)
    .round(3)
    .to_string(index=False)
)


# ==========================================
# DATASET SUMMARY
# ==========================================

print("\n==========================================")
print("        ANOMALY SCORE BY DATASET")
print("==========================================")

summary = (
    df.groupby("source")["anomaly_score"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        max="max",
    )
    .sort_values(
        "mean",
        ascending=False
    )
)

print(
    summary
    .round(3)
    .to_string()
)


# ==========================================
# SCORE DISTRIBUTION
# ==========================================

print("\n==========================================")
print("          SCORE DISTRIBUTION")
print("==========================================")

print(
    df["anomaly_score"]
    .describe()
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
print("       ANOMALY SCORING COMPLETE")
print("==========================================")

print("Saved to:")
print(OUTPUT_FILE)

print("==========================================")