import pandas as pd
import numpy as np
from pathlib import Path

from ml.features.tracking_cleaning import detect_tracking_anomalies


# ==========================================
# PATHS
# ==========================================

PROCESSED_DIR = Path("processed")

OUTPUT_DIR = Path("ml/dataset/reliability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR / "tracking_reliability.csv"
)


# ==========================================
# PROCESS DATASETS
# ==========================================

tracking_files = sorted(
    PROCESSED_DIR.glob("*_processed_data.csv")
)

if not tracking_files:
    raise RuntimeError(
        "No processed tracking files found."
    )


results = []


print("\n==========================================")
print("      TRACKING RELIABILITY ANALYSIS")
print("==========================================")


for file in tracking_files:

    print("\nProcessing:", file.name)

    df = pd.read_csv(file)

    # --------------------------------------
    # Numeric conversion
    # --------------------------------------

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

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.dropna(
        subset=[
            "person_id",
            "frame",
            "timestamp",
            "x",
            "y",
        ]
    ).copy()

    # --------------------------------------
    # Detect tracking anomalies
    # --------------------------------------

    df = detect_tracking_anomalies(df)

    # --------------------------------------
    # Calculate reliability statistics
    # --------------------------------------

    total_rows = len(df)

    anomaly_rows = int(
        df["tracking_anomaly"].sum()
    )

    anomaly_percentage = (
        anomaly_rows / total_rows * 100
        if total_rows > 0
        else 0
    )

    # Large raw movement
    large_jump_rows = int(
        (df["trajectory_displacement"] > 100)
        .sum()
    )

    large_jump_percentage = (
        large_jump_rows / total_rows * 100
        if total_rows > 0
        else 0
    )

    # --------------------------------------
    # Frame statistics
    # --------------------------------------

    total_frames = df["frame"].nunique()

    people_per_frame = (
        df.groupby("frame")["person_id"]
        .nunique()
    )

    average_people = (
        people_per_frame.mean()
        if len(people_per_frame) > 0
        else 0
    )

    minimum_people = (
        people_per_frame.min()
        if len(people_per_frame) > 0
        else 0
    )

    # --------------------------------------
    # Reliability score
    # --------------------------------------

    # Start from perfect reliability.
    reliability = 1.0

    # Penalize tracking anomalies.
    reliability -= min(
        anomaly_percentage / 20,
        0.50
    )

    # Penalize large jumps.
    reliability -= min(
        large_jump_percentage / 10,
        0.30
    )

    # Keep within [0, 1].
    reliability = max(
        0.0,
        min(1.0, reliability)
    )

    # --------------------------------------
    # Save result
    # --------------------------------------

    video_name = file.stem.replace(
        "_processed_data",
        ""
    )

    results.append(
        {
            "source": video_name,
            "total_rows": total_rows,
            "total_frames": total_frames,
            "tracking_anomaly_rows": anomaly_rows,
            "tracking_anomaly_percentage": anomaly_percentage,
            "large_jump_rows": large_jump_rows,
            "large_jump_percentage": large_jump_percentage,
            "average_people_per_frame": average_people,
            "minimum_people_per_frame": minimum_people,
            "tracking_reliability": reliability,
        }
    )


# ==========================================
# FINAL TABLE
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "tracking_reliability"
)


print("\n==========================================")
print("       TRACKING RELIABILITY RESULTS")
print("==========================================")

print(
    results_df.round(3)
    .to_string(index=False)
)


# ==========================================
# SAVE
# ==========================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==========================================")
print("       RELIABILITY ANALYSIS COMPLETE")
print("==========================================")

print("Saved to:")
print(OUTPUT_FILE)

print("==========================================")
