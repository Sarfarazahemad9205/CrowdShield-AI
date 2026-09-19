from pathlib import Path
import pandas as pd
import numpy as np


# ==============================
# PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = BASE_DIR / "ml" / "dataset" / "windowed" / "training_windowed.csv"

OUTPUT_DIR = BASE_DIR / "ml" / "dataset" / "behavioral"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "training_behavioral.csv"


# ==============================
# BEHAVIORAL FEATURES
# ==============================

BEHAVIOR_FEATURES = [
    "avg_speed_mean",
    "avg_acceleration_mean",
    "avg_displacement_mean",
    "avg_speed_change_mean",
    "avg_velocity_change_mean",
]


# ==============================
# LOAD DATA
# ==============================

print("Loading training window data...")

df = pd.read_csv(INPUT_FILE)

print(f"Total windows: {len(df)}")
print(f"Total columns: {len(df.columns)}")


# ==============================
# CHECK REQUIRED COLUMNS
# ==============================

required_columns = ["video_id"] + BEHAVIOR_FEATURES

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ==============================
# SORT DATA
# ==============================

df = df.sort_values(
    ["video_id", "window_start"]
).reset_index(drop=True)


# ==============================
# CREATE BEHAVIORAL FEATURES
# ==============================

result = []

for video_id, video_df in df.groupby("video_id"):

    video_df = video_df.copy()

    deviations = []

    for feature in BEHAVIOR_FEATURES:

        median = video_df[feature].median()

        q1 = video_df[feature].quantile(0.25)
        q3 = video_df[feature].quantile(0.75)

        iqr = q3 - q1

        # Prevent division by zero
        if iqr == 0 or pd.isna(iqr):
            iqr = 1.0

        deviation = (
            video_df[feature] - median
        ) / iqr

        # Only positive deviation represents
        # abnormal increase
        positive_deviation = deviation.clip(lower=0)

        column_name = f"{feature}_deviation"

        video_df[column_name] = positive_deviation

        deviations.append(positive_deviation)

    # ==============================
    # ABNORMAL BEHAVIOR SCORE
    # ==============================

    deviation_df = pd.concat(
        deviations,
        axis=1
    )

    video_df["abnormal_behavior_score"] = (
        deviation_df.mean(axis=1)
    )

    # ==============================
    # TEMPORAL TRENDS
    # ==============================

    video_df["speed_trend"] = (
        video_df["avg_speed_mean"]
        .diff()
        .fillna(0)
    )

    video_df["acceleration_trend"] = (
        video_df["avg_acceleration_mean"]
        .diff()
        .fillna(0)
    )

    video_df["displacement_trend"] = (
        video_df["avg_displacement_mean"]
        .diff()
        .fillna(0)
    )

    result.append(video_df)


# ==============================
# COMBINE VIDEOS
# ==============================

behavioral_df = pd.concat(
    result,
    ignore_index=True
)


# ==============================
# CLEAN VALUES
# ==============================

behavioral_df = behavioral_df.replace(
    [np.inf, -np.inf],
    np.nan
)

behavioral_df = behavioral_df.fillna(0)


# ==============================
# SAVE
# ==============================

behavioral_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==============================
# SUMMARY
# ==============================

print()
print("====================================")
print("BEHAVIORAL FEATURE EXTRACTION")
print("====================================")

print(f"Total windows: {len(behavioral_df)}")
print(f"Total videos: {behavioral_df['video_id'].nunique()}")
print(f"Total columns: {len(behavioral_df.columns)}")

print()
print("Abnormal behavior score:")
print(
    behavioral_df["abnormal_behavior_score"].describe()
)

print()
print("Output:")
print(OUTPUT_FILE)

print()
print("✓ Behavioral features created successfully")