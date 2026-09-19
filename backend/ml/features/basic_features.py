import pandas as pd


def calculate_basic_features(df, movement_df=None):
    """
    Calculate basic crowd-level features.

    people_count is calculated from all detections.

    Movement-related features are calculated only from
    tracking-valid observations.
    """

    if movement_df is None:
        movement_df = df

    # --------------------------------------------------
    # Crowd count from ALL detections
    # --------------------------------------------------

    count_features = (
        df.groupby("frame")
        .agg(
            people_count=("person_id", "nunique")
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Movement features from CLEAN observations
    # --------------------------------------------------

    movement_features = (
        movement_df.groupby("frame")
        .agg(
            avg_speed=("speed", "mean"),
            max_speed=("speed", "max"),
            speed_std=("speed", "std"),

            avg_displacement=("displacement", "mean"),
            max_displacement=("displacement", "max"),

            avg_velocity_x=("velocity_x", "mean"),
            avg_velocity_y=("velocity_y", "mean"),

            avg_width=("width", "mean"),
            avg_height=("height", "mean"),
            avg_density=("density", "mean"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Combine
    # --------------------------------------------------

    features = count_features.merge(
        movement_features,
        on="frame",
        how="left"
    )

    return features