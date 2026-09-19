from pathlib import Path
import pandas as pd


# ==============================
# PATH
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = (
    BASE_DIR
    / "ml"
    / "dataset"
    / "behavioral"
    / "training_behavioral.csv"
)


# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_FILE)

print("====================================")
print("BEHAVIORAL SCORE ANALYSIS")
print("====================================")

print(f"Total windows: {len(df)}")
print(f"Total videos: {df['video_id'].nunique()}")


# ==============================
# SCORE DISTRIBUTION
# ==============================

print()
print("====================================")
print("SCORE DISTRIBUTION")
print("====================================")

print(
    df["abnormal_behavior_score"].describe()
)


# ==============================
# TOP 10 ABNORMAL WINDOWS
# ==============================

print()
print("====================================")
print("TOP 10 ABNORMAL WINDOWS")
print("====================================")

top_windows = (
    df[
        [
            "video_id",
            "window_start",
            "window_end",
            "abnormal_behavior_score",
            "avg_speed_mean",
            "avg_acceleration_mean",
            "avg_displacement_mean",
            "avg_speed_change_mean",
            "avg_velocity_change_mean",
        ]
    ]
    .sort_values(
        "abnormal_behavior_score",
        ascending=False
    )
    .head(10)
)

print(
    top_windows.to_string(index=False)
)


# ==============================
# SCORE BY VIDEO
# ==============================

print()
print("====================================")
print("SCORE BY VIDEO")
print("====================================")

video_summary = (
    df.groupby("video_id")[
        "abnormal_behavior_score"
    ]
    .agg(
        [
            "count",
            "mean",
            "max",
            "std",
        ]
    )
    .sort_values(
        "max",
        ascending=False
    )
)

print(video_summary)


# ==============================
# WINDOWS ABOVE THRESHOLDS
# ==============================

print()
print("====================================")
print("WINDOW COUNT BY SCORE")
print("====================================")

thresholds = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
]

for threshold in thresholds:

    count = (
        df["abnormal_behavior_score"]
        >= threshold
    ).sum()

    percentage = (
        count / len(df)
    ) * 100

    print(
        f"Score >= {threshold:.2f}: "
        f"{count} windows "
        f"({percentage:.2f}%)"
    )


# ==============================
# HIGHEST SCORE PER VIDEO
# ==============================

print()
print("====================================")
print("HIGHEST SCORE PER VIDEO")
print("====================================")

highest_per_video = (
    df.loc[
        df.groupby("video_id")[
            "abnormal_behavior_score"
        ].idxmax()
    ]
    [
        [
            "video_id",
            "window_start",
            "window_end",
            "abnormal_behavior_score",
        ]
    ]
    .sort_values(
        "abnormal_behavior_score",
        ascending=False
    )
)

print(
    highest_per_video.to_string(index=False)
)


print()
print("✓ Behavioral score analysis completed")