from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_DIR = Path("ml/dataset/windowed")
OUTPUT_DIR = Path("ml/dataset/normalized")


# --------------------------------------------------
# Features to normalize
# --------------------------------------------------

FEATURES_TO_NORMALIZE = [
    "avg_speed_mean",
    "max_speed_mean",
    "speed_std_mean",
    "avg_displacement_mean",
    "max_displacement_mean",
    "avg_acceleration_mean",
    "max_acceleration_mean",
    "acceleration_std_mean",
    "avg_speed_change_mean",
    "max_speed_change_mean",
    "avg_velocity_change_mean",
    "max_velocity_change_mean",
    "avg_displacement_change_mean",
]


# --------------------------------------------------
# Percentile normalization
# --------------------------------------------------

def percentile_score(series):
    """
    Convert values into percentile ranks from 0 to 1.

    0.0 = relatively low value in this video
    1.0 = relatively high value in this video
    """

    values = series.copy()

    if values.nunique() <= 1:
        return pd.Series(
            np.full(len(values), 0.5),
            index=values.index
        )

    return values.rank(
        method="average",
        pct=True
    )


# --------------------------------------------------
# Normalize one dataset
# --------------------------------------------------

def normalize_dataset(df):
    """
    Add video-relative percentile features.

    Original features are preserved.
    """

    data = df.copy()

    for feature in FEATURES_TO_NORMALIZE:

        if feature not in data.columns:
            continue

        normalized_name = (
            feature.replace("_mean", "")
            + "_percentile"
        )

        data[normalized_name] = percentile_score(
            data[feature]
        )

    return data


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        INPUT_DIR.glob("*_windows.csv")
    )

    if not files:

        print(
            "No windowed datasets found in:",
            INPUT_DIR
        )

        return

    print("=" * 60)
    print("VIDEO-RELATIVE FEATURE NORMALIZATION")
    print("=" * 60)

    for file in files:

        print(f"\nProcessing: {file.name}")

        df = pd.read_csv(file)

        normalized = normalize_dataset(df)

        output_file = (
            OUTPUT_DIR /
            file.name.replace(
                "_windows.csv",
                "_normalized.csv"
            )
        )

        normalized.to_csv(
            output_file,
            index=False
        )

        print(
            f"Input rows  : {len(df)}"
        )

        print(
            f"Output rows : {len(normalized)}"
        )

        print(
            f"Output file : {output_file}"
        )

    print("\n" + "=" * 60)
    print("NORMALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()