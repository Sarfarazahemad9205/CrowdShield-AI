import pandas as pd
from pathlib import Path


# ==========================================
# PATHS
# ==========================================

INPUT_FILE = Path(
    "ml/dataset/anomaly/reliability_weighted_scores.csv"
)


# ==========================================
# LOAD DATA
# ==========================================

print("\n==========================================")
print("          PERSISTENCE ANALYSIS")
print("==========================================")

df = pd.read_csv(INPUT_FILE)


# ==========================================
# EXTRACT NUMERIC WINDOW ID
# ==========================================

df["window_id_numeric"] = pd.to_numeric(
    df["window_id"],
    errors="coerce"
)


# ==========================================
# SORT
# ==========================================

df = df.sort_values(
    [
        "source",
        "window_id_numeric"
    ]
).reset_index(drop=True)


# ==========================================
# TEMPORARY ANOMALY THRESHOLD
# ==========================================

# IMPORTANT:
# This is NOT our final risk threshold.
# It is only used to study persistence.

PERSISTENCE_THRESHOLD = 2.0


df["above_threshold"] = (
    df["reliability_weighted_score"]
    >= PERSISTENCE_THRESHOLD
)


# ==========================================
# CALCULATE CONSECUTIVE RUNS
# ==========================================

df["run_group"] = (
    df.groupby("source")["above_threshold"]
    .transform(
        lambda x: x.ne(x.shift()).cumsum()
    )
)


df["consecutive_count"] = (
    df.groupby(
        ["source", "run_group"]
    )
    .cumcount()
    + 1
)


# ==========================================
# KEEP ONLY ABNORMAL WINDOWS
# ==========================================

abnormal = df[
    df["above_threshold"]
].copy()


# ==========================================
# FIND MAX PERSISTENCE
# ==========================================

persistence_summary = (
    abnormal
    .groupby("source")
    .agg(
        abnormal_windows=(
            "window_id",
            "count"
        ),
        maximum_consecutive_windows=(
            "consecutive_count",
            "max"
        ),
        maximum_score=(
            "reliability_weighted_score",
            "max"
        ),
        mean_score=(
            "reliability_weighted_score",
            "mean"
        ),
    )
    .sort_values(
        "maximum_consecutive_windows",
        ascending=False
    )
)


# ==========================================
# DISPLAY
# ==========================================

print("\n==========================================")
print("      PERSISTENCE SUMMARY")
print("==========================================")

print(
    persistence_summary
    .round(3)
    .to_string()
)


# ==========================================
# DETAILED ABNORMAL WINDOWS
# ==========================================

print("\n==========================================")
print("     ABNORMAL WINDOWS IN ORDER")
print("==========================================")

display_columns = [
    "source",
    "window_id",
    "reliability_weighted_score",
    "consecutive_count",
]

print(
    abnormal[
        display_columns
    ]
    .round(3)
    .to_string(index=False)
)


print("\n==========================================")
print("       PERSISTENCE ANALYSIS COMPLETE")
print("==========================================")