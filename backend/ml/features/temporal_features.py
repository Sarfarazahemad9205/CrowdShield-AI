import numpy as np
import pandas as pd

from ml.features.tracking_cleaning import detect_tracking_anomalies


def calculate_temporal_features(df):
    """
    Calculate temporal movement features from individual
    person trajectories.

    Tracking anomalies are excluded from temporal calculations
    so that sudden tracking jumps do not create artificial
    acceleration or velocity-change values.
    """

    data = df.copy()

    # --------------------------------------------------
    # Detect tracking anomalies
    # --------------------------------------------------

    data = detect_tracking_anomalies(data)

    # Make sure each person's trajectory is in chronological order
    data = data.sort_values(
        ["person_id", "frame"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Time difference
    # --------------------------------------------------

    data["time_change"] = (
        data.groupby("person_id")["timestamp"].diff()
    )

    data.loc[
        data["time_change"] <= 0,
        "time_change"
    ] = np.nan

    # --------------------------------------------------
    # Prevent anomaly transitions from affecting features
    # --------------------------------------------------

    previous_anomaly = (
        data.groupby("person_id")["tracking_anomaly"]
        .shift(fill_value=False)
    )

    temporal_invalid = (
        data["tracking_anomaly"] |
        previous_anomaly
    )

    # --------------------------------------------------
    # Speed change
    # --------------------------------------------------

    data["speed_change"] = (
        data.groupby("person_id")["speed"].diff()
    )

    data.loc[
        temporal_invalid,
        "speed_change"
    ] = np.nan

    data["abs_speed_change"] = (
        data["speed_change"].abs()
    )

    # --------------------------------------------------
    # Acceleration
    # --------------------------------------------------

    data["acceleration"] = (
        data["speed_change"] /
        data["time_change"]
    )

    data["abs_acceleration"] = (
        data["acceleration"].abs()
    )

    # --------------------------------------------------
    # Velocity change
    # --------------------------------------------------

    data["velocity_change_x"] = (
        data.groupby("person_id")["velocity_x"].diff()
    )

    data["velocity_change_y"] = (
        data.groupby("person_id")["velocity_y"].diff()
    )

    data["velocity_change"] = np.sqrt(
        data["velocity_change_x"] ** 2 +
        data["velocity_change_y"] ** 2
    )

    data.loc[
        temporal_invalid,
        "velocity_change"
    ] = np.nan

    # --------------------------------------------------
    # Displacement change
    # --------------------------------------------------

    data["displacement_change"] = (
        data.groupby("person_id")["displacement"].diff()
    )

    data.loc[
        temporal_invalid,
        "displacement_change"
    ] = np.nan

    data["abs_displacement_change"] = (
        data["displacement_change"].abs()
    )

    # --------------------------------------------------
    # Aggregate temporal features per frame
    # --------------------------------------------------

    temporal_features = (
        data.groupby("frame")
        .agg(
            avg_acceleration=("abs_acceleration", "mean"),
            max_acceleration=("abs_acceleration", "max"),
            acceleration_std=("abs_acceleration", "std"),

            avg_speed_change=("abs_speed_change", "mean"),
            max_speed_change=("abs_speed_change", "max"),

            avg_velocity_change=("velocity_change", "mean"),
            max_velocity_change=("velocity_change", "max"),

            avg_displacement_change=(
                "abs_displacement_change",
                "mean"
            ),
        )
        .reset_index()
    )

    return temporal_features