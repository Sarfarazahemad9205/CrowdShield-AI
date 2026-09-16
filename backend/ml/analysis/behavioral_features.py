from pathlib import Path
import numpy as np
import pandas as pd


INPUT_DIR = Path("ml/dataset/windowed")
OUTPUT_DIR = Path("ml/dataset/behavioral")


FEATURES = [
    "avg_speed_mean",
    "avg_acceleration_mean",
    "avg_displacement_mean",
    "avg_speed_change_mean",
    "avg_velocity_change_mean",
]


def calculate_robust_deviation(series):
    """
    Calculate how far a value is from the median
    using the Interquartile Range (IQR).

    Formula:
        deviation = (value - median) / IQR
    """

    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    if iqr == 0:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (series - median) / iqr


def calculate_trend(series):
    """
    Calculate the change between consecutive windows.

    Positive value  -> feature is increasing
    Negative value  -> feature is decreasing
    Zero             -> no change
    """

    return series.diff()


def calculate_behavioral_features(df):
    """
    Calculate behavioral abnormality features
    from 5-second crowd windows.
    """

    data = df.copy()

    # -------------------------------------------------
    # 1. Robust deviation features
    # -------------------------------------------------

    if "avg_speed_mean" in data.columns:
        data["speed_deviation"] = calculate_robust_deviation(
            data["avg_speed_mean"]
        )

    if "avg_acceleration_mean" in data.columns:
        data["acceleration_deviation"] = calculate_robust_deviation(
            data["avg_acceleration_mean"]
        )

    if "avg_displacement_mean" in data.columns:
        data["displacement_deviation"] = calculate_robust_deviation(
            data["avg_displacement_mean"]
        )

    if "avg_speed_change_mean" in data.columns:
        data["speed_change_deviation"] = calculate_robust_deviation(
            data["avg_speed_change_mean"]
        )

    if "avg_velocity_change_mean" in data.columns:
        data["velocity_change_deviation"] = calculate_robust_deviation(
            data["avg_velocity_change_mean"]
        )

    # -------------------------------------------------
    # 2. Temporal trends
    # -------------------------------------------------

    if "avg_speed_mean" in data.columns:
        data["speed_trend"] = calculate_trend(
            data["avg_speed_mean"]
        )

    if "avg_acceleration_mean" in data.columns:
        data["acceleration_trend"] = calculate_trend(
            data["avg_acceleration_mean"]
        )

    # -------------------------------------------------
    # 3. Combined abnormal behavior score
    # -------------------------------------------------

    deviation_columns = [
        "speed_deviation",
        "acceleration_deviation",
        "displacement_deviation",
        "speed_change_deviation",
        "velocity_change_deviation",
    ]

    existing_columns = [
        column
        for column in deviation_columns
        if column in data.columns
    ]

    if existing_columns:
        data["abnormal_behavior_score"] = (
            data[existing_columns]
            .clip(lower=0)
            .mean(axis=1)
        )

    return data


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
    print("BEHAVIORAL FEATURE EXTRACTION")
    print("=" * 60)

    for file in files:

        print(f"\nProcessing: {file.name}")

        df = pd.read_csv(file)

        behavioral = calculate_behavioral_features(df)

        output_file = (
            OUTPUT_DIR /
            file.name.replace(
                "_windows.csv",
                "_behavioral.csv"
            )
        )

        behavioral.to_csv(
            output_file,
            index=False
        )

        print("Input rows  :", len(df))
        print("Output rows :", len(behavioral))
        print("Output file :", output_file)

    print("\n" + "=" * 60)
    print("BEHAVIORAL FEATURE EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()