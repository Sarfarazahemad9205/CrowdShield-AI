import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = BASE_DIR / "ml" / "dataset" / "training_features.csv"

OUTPUT_DIR = BASE_DIR / "ml" / "dataset" / "windowed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "training_windowed.csv"


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SECONDS = 5


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CREATING 5-SECOND TRAINING WINDOWS")
print("=" * 60)

print("\nInput file:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("\nTotal frame rows:", len(df))
print("Total videos:", df["video_id"].nunique())


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    ["video_id", "timestamp"]
).reset_index(drop=True)


# ============================================================
# CREATE WINDOW ID
# ============================================================

df["window_id"] = (
    df["timestamp"] // WINDOW_SECONDS
).astype(int)


# ============================================================
# CREATE WINDOW START / END
# ============================================================

df["window_start"] = (
    df["window_id"] * WINDOW_SECONDS
)

df["window_end"] = (
    df["window_start"] + WINDOW_SECONDS
)


# ============================================================
# FEATURES TO AGGREGATE
# ============================================================

exclude_columns = {
    "video_id",
    "frame",
    "timestamp",
    "window_id",
    "window_start",
    "window_end"
}


feature_columns = [
    column
    for column in df.columns
    if column not in exclude_columns
    and pd.api.types.is_numeric_dtype(df[column])
]


# ============================================================
# AGGREGATE EACH 5-SECOND WINDOW
# ============================================================

print("\nFeature columns being aggregated:")
print(feature_columns)


windowed = (
    df.groupby(
        [
            "video_id",
            "window_id",
            "window_start",
            "window_end"
        ],
        as_index=False
    )[feature_columns]
    .agg(["mean", "std", "min", "max"])
)


# ============================================================
# FLATTEN MULTI-LEVEL COLUMNS
# ============================================================

windowed.columns = [
    "_".join(column).strip("_")
    if isinstance(column, tuple)
    else column
    for column in windowed.columns
]


# ============================================================
# ADD FRAME COUNT
# ============================================================

frame_counts = (
    df.groupby(
        [
            "video_id",
            "window_id"
        ]
    )
    .size()
    .reset_index(name="frame_count")
)


windowed = windowed.merge(
    frame_counts,
    on=["video_id", "window_id"],
    how="left"
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

windowed = windowed.fillna(0)


# ============================================================
# SORT
# ============================================================

windowed = windowed.sort_values(
    ["video_id", "window_id"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

windowed.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("WINDOW CREATION COMPLETED")
print("=" * 60)

print("\nTotal windows:", len(windowed))

print(
    "Total videos:",
    windowed["video_id"].nunique()
)

print("\nWindows per video:")

print(
    windowed["video_id"]
    .value_counts()
    .sort_index()
)

print("\nTotal columns:", len(windowed.columns))

print("\nOutput:")
print(OUTPUT_FILE)

print("=" * 60)