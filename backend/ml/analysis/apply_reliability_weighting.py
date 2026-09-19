import pandas as pd
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

ANOMALY_FILE = Path(
    "ml/dataset/anomaly/grouped_anomaly_scores.csv"
)

RELIABILITY_FILE = Path(
    "ml/dataset/reliability/tracking_reliability.csv"
)

OUTPUT_DIR = Path("ml/dataset/anomaly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR / "reliability_weighted_scores.csv"
)


# ==========================================
# LOAD DATA
# ==========================================

print("\n==========================================")
print("    RELIABILITY WEIGHTED ANOMALY SCORE")
print("==========================================")

anomaly_df = pd.read_csv(ANOMALY_FILE)

reliability_df = pd.read_csv(
    RELIABILITY_FILE
)


# ==========================================
# NORMALIZE SOURCE NAMES
# ==========================================

# Behavioral source:
# Bengaluru_Crowd_stampede_features_behavioral
#
# Reliability source:
# Bengaluru_Crowd_stampede
#
# Convert behavioral source names into
# the same naming convention.

anomaly_df["reliability_source"] = (
    anomaly_df["source"]
    .str.replace(
        "_features_behavioral",
        "",
        regex=False
    )
)


# ==========================================
# MERGE
# ==========================================

df = anomaly_df.merge(
    reliability_df[
        [
            "source",
            "tracking_reliability",
            "tracking_anomaly_percentage",
            "large_jump_percentage",
        ]
    ],
    left_on="reliability_source",
    right_on="source",
    how="left",
    suffixes=("", "_reliability")
)


# ==========================================
# CHECK MERGE
# ==========================================

missing_reliability = (
    df["tracking_reliability"]
    .isna()
    .sum()
)

print(
    "\nWindows without reliability:",
    missing_reliability
)

if missing_reliability > 0:

    print("\nSources with missing reliability:")

    print(
        df.loc[
            df["tracking_reliability"].isna(),
            "source"
        ]
        .unique()
    )

    raise RuntimeError(
        "Some datasets could not be matched "
        "with tracking reliability."
    )


# ==========================================
# RELIABILITY-WEIGHTED MOVEMENT
# ==========================================

df["weighted_movement_score"] = (
    df["movement_anomaly_score"]
    * df["tracking_reliability"]
)


# ==========================================
# FINAL WEIGHTED SCORE
# ==========================================

df["reliability_weighted_score"] = (
    0.5 * df["weighted_movement_score"]
    +
    0.5 * df["structure_anomaly_score"]
)


# ==========================================
# SORT
# ==========================================

df = df.sort_values(
    "reliability_weighted_score",
    ascending=False
).reset_index(drop=True)


# ==========================================
# TOP WINDOWS
# ==========================================

print("\n==========================================")
print("       TOP WEIGHTED ANOMALIES")
print("==========================================")

display_columns = [
    "source",
    "window_id",
    "tracking_reliability",
    "movement_anomaly_score",
    "weighted_movement_score",
    "structure_anomaly_score",
    "grouped_anomaly_score",
    "reliability_weighted_score",
]

print(
    df[display_columns]
    .head(25)
    .round(3)
    .to_string(index=False)
)


# ==========================================
# DATASET SUMMARY
# ==========================================

print("\n==========================================")
print("       DATASET SCORE COMPARISON")
print("==========================================")

summary = (
    df.groupby("source")
    [
        [
            "tracking_reliability",
            "movement_anomaly_score",
            "structure_anomaly_score",
            "grouped_anomaly_score",
            "reliability_weighted_score",
        ]
    ]
    .mean()
    .sort_values(
        "reliability_weighted_score",
        ascending=False
    )
)

print(
    summary
    .round(3)
    .to_string()
)


# ==========================================
# SAVE
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==========================================")
print("     RELIABILITY WEIGHTING COMPLETE")
print("==========================================")

print("Saved to:")
print(OUTPUT_FILE)

print("==========================================")