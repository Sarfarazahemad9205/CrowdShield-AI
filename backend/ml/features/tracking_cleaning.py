import numpy as np
import pandas as pd


def detect_tracking_anomalies(df):
    """
    Detect suspicious jumps in individual person trajectories.

    The raw tracking data is not modified or deleted.
    Instead, suspicious observations are marked with
    the 'tracking_anomaly' column.
    """

    data = df.copy()

    # Sort each person's trajectory chronologically
    data = data.sort_values(
        ["person_id", "frame"]
    ).reset_index(drop=True)

    # Time difference between consecutive observations
    data["time_change"] = (
        data.groupby("person_id")["timestamp"].diff()
    )

    data.loc[
        data["time_change"] <= 0,
        "time_change"
    ] = np.nan

    # Position changes
    data["dx"] = (
        data.groupby("person_id")["x"].diff()
    )

    data["dy"] = (
        data.groupby("person_id")["y"].diff()
    )

    # Distance travelled between observations
    data["trajectory_displacement"] = np.sqrt(
        data["dx"] ** 2 +
        data["dy"] ** 2
    )

    # Movement speed based on trajectory
    data["trajectory_speed"] = (
        data["trajectory_displacement"]
        / data["time_change"]
    )

    # Calculate robust statistics for each person
    median_speed = (
        data.groupby("person_id")["trajectory_speed"]
        .transform("median")
    )

    q1 = (
        data.groupby("person_id")["trajectory_speed"]
        .transform(lambda x: x.quantile(0.25))
    )

    q3 = (
        data.groupby("person_id")["trajectory_speed"]
        .transform(lambda x: x.quantile(0.75))
    )

    iqr = q3 - q1

    # Upper limit for unusually large movement
    upper_limit = q3 + 3 * iqr

    # Detect suspicious movement
    data["tracking_anomaly"] = (
        data["trajectory_speed"] > upper_limit
    )

    # First observation of each person cannot be compared
    first_observation = (
        data.groupby("person_id").cumcount() == 0
    )

    data.loc[
        first_observation,
        "tracking_anomaly"
    ] = False

    return data