import pandas as pd
from pathlib import Path

from ml.features.basic_features import calculate_basic_features
from ml.features.temporal_features import calculate_temporal_features
from ml.features.directional_features import calculate_directional_features


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "processed/stampede_video_processed_data.csv"

OUTPUT_DIR = Path("ml/dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "crowd_features.csv"


# ============================================================
# LOAD RAW TRACKING DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("===== RAW TRACKING DATA =====")
print("Total rows:", len(df))
print("Total frames:", df["frame"].nunique())


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "frame",
    "timestamp",
    "x",
    "y",
    "width",
    "height",
    "movement_x",
    "movement_y",
    "displacement",
    "velocity_x",
    "velocity_y",
    "speed",
    "direction",
    "density",
]


print("\n===== MISSING VALUES AFTER CONVERSION =====")
print(df.isna().sum())
# ============================================================
# CHECK DATA TYPES
# ============================================================
print("\n===== DIRECTION VALUES =====")
print(df["direction"].head(20).tolist())
print("Direction dtype:", df["direction"].dtype)
# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=[
        "person_id",
        "frame",
        "timestamp",
        "speed",
        "displacement",
        "velocity_x",
        "velocity_y",
        "direction",
        "density",
    ]
)


print("\n===== AFTER CLEANING =====")
print("Rows remaining:", len(df))
print("Frames remaining:", df["frame"].nunique())


# ============================================================
# BASIC FEATURES
# ============================================================

print("\n===== BASIC FEATURES =====")

basic_features = calculate_basic_features(df)

print(
    "Basic feature columns:",
    len(basic_features.columns)
)


# ============================================================
# TEMPORAL FEATURES
# ============================================================

print("\n===== TEMPORAL FEATURES =====")

temporal_features = calculate_temporal_features(df)

print(
    "Temporal feature columns:",
    len(temporal_features.columns)
)


# ============================================================
# DIRECTIONAL FEATURES
# ============================================================

print("\n===== DIRECTIONAL FEATURES =====")

directional_features = calculate_directional_features(df)

print(
    "Directional feature columns:",
    len(directional_features.columns)
)


# ============================================================
# MERGE ALL FEATURES
# ============================================================

features = basic_features.merge(
    temporal_features,
    on="frame",
    how="left"
)

features = features.merge(
    directional_features,
    on="frame",
    how="left"
)


# ============================================================
# ADD TIMESTAMP
# ============================================================

timestamps = (
    df.groupby("frame")["timestamp"]
    .first()
    .reset_index()
)

features = features.merge(
    timestamps,
    on="frame",
    how="left"
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

features = features.fillna(0)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

features = features[
    [
        "frame",
        "timestamp",

        # Basic crowd features
        "people_count",
        "avg_speed",
        "max_speed",
        "speed_std",
        "avg_displacement",
        "max_displacement",
        "avg_velocity_x",
        "avg_velocity_y",

        # Temporal features
        "avg_acceleration",
        "max_acceleration",
        "acceleration_std",
        "avg_speed_change",
        "max_speed_change",
        "avg_velocity_change",
        "max_velocity_change",
        "avg_displacement_change",

        # Directional features
        "direction_alignment",
        "directional_disorder",

        # Appearance / density
        "avg_width",
        "avg_height",
        "avg_density",
    ]
]


# ============================================================
# SAVE FEATURE DATASET
# ============================================================

features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n===== FINAL CROWD FEATURES =====")

print(
    "Total feature rows:",
    len(features)
)

print(
    "Total feature columns:",
    len(features.columns)
)


print("\n===== FEATURE COLUMNS =====")

print(
    features.columns.tolist()
)


print("\n===== FIRST 10 ROWS =====")

print(
    features.head(10).to_string(
        index=False
    )
)


print("\n===== BASIC STATISTICS =====")

print(
    features.describe().to_string()
)


print("\n==========================================")
print("Feature extraction completed successfully")
print("Saved to:", OUTPUT_FILE)
print("==========================================")