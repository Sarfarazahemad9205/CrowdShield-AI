import numpy as np
import pandas as pd


def detect_tracking_anomalies(df):
    """
    Detect suspicious jumps in individual person trajectories.

    The raw tracking data is not modified or deleted.
    Suspicious observations are marked with
    the 'tracking_anomaly' column.

    A movement transition is considered suspicious if:
    1. It exceeds the robust per-person IQR threshold, OR
    2. Its displacement exceeds 100 pixels in one frame transition.
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
    q1 = (
        data.groupby("person_id")["trajectory_speed"]
        .transform(lambda x: x.quantile(0.25))
    )

    q3 = (
        data.groupby("person_id")["trajectory_speed"]
        .transform(lambda x: x.quantile(0.75))
    )

    iqr = q3 - q1

    # Robust upper limit
    upper_limit = q3 + 3 * iqr

    # Existing statistical anomaly rule
    statistical_anomaly = (
        data["trajectory_speed"] > upper_limit
    )

    # Hard movement-quality gate
    displacement_anomaly = (
        data["trajectory_displacement"] > 100
    )

    # Either condition marks the transition as suspicious
    data["tracking_anomaly"] = (
        statistical_anomaly |
        displacement_anomaly
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