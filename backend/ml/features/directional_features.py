from collections import Counter
import pandas as pd


def calculate_directional_features(df):
    """
    Calculate crowd directional alignment and disorder.

    Direction values are categorical:
    LEFT, RIGHT, UP, DOWN, STATIONARY.

    STATIONARY people are excluded from directional
    movement calculations.
    """

    results = []

    for frame, group in df.groupby("frame"):

        # Keep only people who are moving
        moving_group = group[
            group["direction"].str.upper() != "STATIONARY"
        ]

        # If nobody is moving
        if len(moving_group) == 0:
            results.append(
                {
                    "frame": frame,
                    "direction_alignment": 0.0,
                    "directional_disorder": 0.0,
                }
            )
            continue

        directions = (
            moving_group["direction"]
            .str.upper()
            .tolist()
        )

        # Count each direction
        counts = Counter(directions)

        # Number of moving people
        total = len(directions)

        # Most common direction
        dominant_count = counts.most_common(1)[0][1]

        # Direction alignment
        direction_alignment = (
            dominant_count / total
        )

        # Directional disorder
        directional_disorder = (
            1.0 - direction_alignment
        )

        results.append(
            {
                "frame": frame,
                "direction_alignment": direction_alignment,
                "directional_disorder": directional_disorder,
            }
        )

    return pd.DataFrame(results)