import pandas as pd


def calculate_basic_features(df):
    """
    Calculate basic crowd-level features.

    One output row represents one video frame.
    """

    features = (
        df.groupby("frame")
        .agg(
            people_count=("person_id", "nunique"),
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

    return features