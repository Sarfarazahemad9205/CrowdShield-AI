import pandas as pd
from pathlib import Path

from ml.features.basic_features import calculate_basic_features
from ml.features.temporal_features import calculate_temporal_features
from ml.features.directional_features import calculate_directional_features
from ml.features.tracking_cleaning import detect_tracking_anomalies


# ============================================================
# DIRECTORIES
# ============================================================

PROCESSED_DIR = Path("processed")
OUTPUT_DIR = Path("ml/dataset")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(input_file):

    print("\n")
    print("==========================================")
    print("        FEATURE EXTRACTION")
    print("==========================================")

    print("Input file:", input_file)

    # ========================================================
    # LOAD RAW TRACKING DATA
    # ========================================================

    df = pd.read_csv(input_file)

    print("\n===== RAW TRACKING DATA =====")

    print("Total rows:", len(df))
    print("Total frames:", df["frame"].nunique())

    # ========================================================
    # CONVERT NUMERIC COLUMNS
    # ========================================================

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
        "density",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ========================================================
    # CHECK MISSING VALUES
    # ========================================================

    print("\n===== MISSING VALUES AFTER CONVERSION =====")

    print(df.isna().sum())

    # ========================================================
    # CHECK DIRECTIONS
    # ========================================================

    print("\n===== DIRECTION VALUES =====")

    print(
        df["direction"]
        .head(20)
        .tolist()
    )

    print(
        "Direction dtype:",
        df["direction"].dtype
    )

    # ========================================================
    # REMOVE INVALID ROWS
    # ========================================================

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
    ).copy()

    print("\n===== AFTER BASIC CLEANING =====")

    print("Rows remaining:", len(df))
    print("Frames remaining:", df["frame"].nunique())

    # ========================================================
    # TRACKING ANOMALY DETECTION
    # ========================================================

    print("\n===== TRACKING QUALITY =====")

    df = detect_tracking_anomalies(df)

    anomaly_count = int(
        df["tracking_anomaly"].sum()
    )

    anomaly_percentage = (
        anomaly_count / len(df) * 100
        if len(df) > 0
        else 0
    )

    print(
        "Tracking anomalies:",
        anomaly_count
    )

    print(
        "Tracking anomaly percentage:",
        round(anomaly_percentage, 2),
        "%"
    )

    # ========================================================
    # CLEAN MOVEMENT DATA
    # ========================================================
    #
    # Keep ALL detections in df for people_count.
    #
    # Exclude suspicious tracking observations from
    # movement-related statistics.
    #
    # ========================================================

    movement_df = df[
        ~df["tracking_anomaly"]
    ].copy()

    print(
        "Movement-valid rows:",
        len(movement_df)
    )

    # ========================================================
    # BASIC FEATURES
    # ========================================================

    print("\n===== BASIC FEATURES =====")

    basic_features = calculate_basic_features(
        df,
        movement_df
    )

    print(
        "Basic feature columns:",
        len(basic_features.columns)
    )

    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================

    print("\n===== TEMPORAL FEATURES =====")

    temporal_features = calculate_temporal_features(df)

    print(
        "Temporal feature columns:",
        len(temporal_features.columns)
    )

    # ========================================================
    # DIRECTIONAL FEATURES
    # ========================================================

    print("\n===== DIRECTIONAL FEATURES =====")

    directional_features = calculate_directional_features(
        movement_df
    )

    print(
        "Directional feature columns:",
        len(directional_features.columns)
    )

    # ========================================================
    # MERGE FEATURES
    # ========================================================

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

    # ========================================================
    # ADD TIMESTAMP
    # ========================================================

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

    # ========================================================
    # HANDLE MISSING VALUES
    # ========================================================

    features = features.fillna(0)

    # ========================================================
    # FINAL COLUMN ORDER
    # ========================================================

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

    # ========================================================
    # OUTPUT FILE
    # ========================================================

    video_name = input_file.stem.replace(
        "_processed_data",
        ""
    )

    output_file = (
        OUTPUT_DIR /
        f"{video_name}_features.csv"
    )

    # ========================================================
    # SAVE
    # ========================================================

    features.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

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
    print("Saved to:", output_file)
    print("==========================================")


# ============================================================
# FIND ALL TRACKING CSV FILES
# ============================================================

tracking_files = sorted(
    PROCESSED_DIR.glob("*_processed_data.csv")
)


# ============================================================
# CHECK FILES
# ============================================================

print("\n==========================================")
print("       TRACKING DATASETS FOUND")
print("==========================================")

if not tracking_files:

    print("No *_processed_data.csv files found.")

else:

    for file in tracking_files:
        print(file.name)

    print(
        "\nTotal datasets:",
        len(tracking_files)
    )

    # ========================================================
    # PROCESS ALL VIDEOS
    # ========================================================

    for file in tracking_files:

        extract_features(file)


print("\n")
print("==========================================")
print("   ALL FEATURE EXTRACTION COMPLETED")
print("==========================================")