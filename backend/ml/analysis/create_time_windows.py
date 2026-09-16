from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATASET_DIR = Path("ml/dataset")
OUTPUT_DIR = DATASET_DIR / "windowed"

WINDOW_SECONDS = 5


# --------------------------------------------------
# Features used for window-level analysis
# --------------------------------------------------

NUMERIC_FEATURES = [
    "people_count",
    "avg_speed",
    "max_speed",
    "speed_std",
    "avg_displacement",
    "max_displacement",
    "avg_velocity_x",
    "avg_velocity_y",
    "avg_acceleration",
    "max_acceleration",
    "acceleration_std",
    "avg_speed_change",
    "max_speed_change",
    "avg_velocity_change",
    "max_velocity_change",
    "avg_displacement_change",
    "direction_alignment",
    "directional_disorder",
    "avg_width",
    "avg_height",
    "avg_density",
]


# --------------------------------------------------
# Create time windows
# --------------------------------------------------

def create_time_windows(df, window_seconds=5):
    """
    Convert frame-level features into fixed time windows.

    Each output row represents one time window.
    """

    data = df.copy()

    # Make sure data is sorted by time
    data = data.sort_values("timestamp").reset_index(drop=True)

    # Create window number
    data["window_id"] = (
        data["timestamp"] // window_seconds
    ).astype(int)

    # Window start/end
    data["window_start"] = (
        data["window_id"] * window_seconds
    )

    data["window_end"] = (
        data["window_start"] + window_seconds
    )

    # Aggregate features inside each window
    aggregations = {}

    for feature in NUMERIC_FEATURES:

        if feature in data.columns:

            aggregations[feature] = [
                "mean",
                "std",
                "min",
                "max",
            ]

    windowed = (
        data.groupby(
            [
                "window_id",
                "window_start",
                "window_end",
            ]
        )
        .agg(aggregations)
        .reset_index()
    )

    # Flatten multi-level column names
    new_columns = []

    for column in windowed.columns:

        if isinstance(column, tuple):

            if column[1] == "":
                new_columns.append(column[0])
            else:
                new_columns.append(
                    f"{column[0]}_{column[1]}"
                )

        else:
            new_columns.append(column)

    windowed.columns = new_columns

    return windowed


# --------------------------------------------------
# Process all feature datasets
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_files = sorted(
        DATASET_DIR.glob("*_features.csv")
    )

    if not feature_files:

        print(
            "No feature CSV files found in:",
            DATASET_DIR
        )

        return

    print("=" * 60)
    print("TIME-WINDOW FEATURE GENERATION")
    print("=" * 60)

    for file in feature_files:

        print(f"\nProcessing: {file.name}")

        df = pd.read_csv(file)

        if "timestamp" not in df.columns:

            print(
                "Skipping because timestamp column is missing."
            )

            continue

        windowed = create_time_windows(
            df,
            WINDOW_SECONDS
        )

        output_file = (
            OUTPUT_DIR /
            f"{file.stem}_windows.csv"
        )

        windowed.to_csv(
            output_file,
            index=False
        )

        duration = df["timestamp"].max()

        print(
            f"Original frames : {len(df)}"
        )

        print(
            f"Video duration   : {duration:.2f} seconds"
        )

        print(
            f"Window size      : {WINDOW_SECONDS} seconds"
        )

        print(
            f"Windows created  : {len(windowed)}"
        )

        print(
            f"Saved to         : {output_file}"
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()