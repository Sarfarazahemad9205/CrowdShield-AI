import pandas as pd
import glob
import os
import numpy as np

# --------------------------------------------------
# 1. Load source behavioral files
# --------------------------------------------------

exclude = {
    "training_behavioral.csv",
    "crowd_features_behavioral.csv",
    "stampede_video_features_behavioral.csv",
    "training_features_behavioral.csv"
}

files = [
    f for f in glob.glob("ml/dataset/behavioral/*.csv")
    if os.path.basename(f) not in exclude
]

df = pd.concat(
    [
        pd.read_csv(f).assign(
            video_id=os.path.splitext(os.path.basename(f))[0]
        )
        for f in files
    ],
    ignore_index=True
)

# --------------------------------------------------
# 2. Features used for normal-behavior baseline
# --------------------------------------------------

features = [
    "people_count_mean",
    "avg_speed_mean",
    "avg_acceleration_mean",
    "avg_displacement_mean",
    "avg_speed_change_mean",
    "avg_velocity_change_mean",
    "directional_disorder_mean",
    "avg_density_mean"
]

# --------------------------------------------------
# 3. Build initial baseline from lowest 25%
# --------------------------------------------------

df = df.sort_values("abnormal_behavior_score")

n = max(1, int(len(df) * 0.25))

baseline_samples = df.head(n)

baseline_median = baseline_samples[features].median()
baseline_iqr = (
    baseline_samples[features].quantile(0.75)
    - baseline_samples[features].quantile(0.25)
)

# Prevent division by zero
baseline_iqr = baseline_iqr.replace(0, 1e-9)

# --------------------------------------------------
# 4. Calculate robust deviations
# --------------------------------------------------

deviations = pd.DataFrame(index=df.index)

for feature in features:
    deviations[feature] = (
        (df[feature] - baseline_median[feature]).abs()
        / baseline_iqr[feature]
    )

# --------------------------------------------------
# 5. Overall anomaly score
# --------------------------------------------------

df["baseline_anomaly_score"] = deviations.mean(axis=1)

# --------------------------------------------------
# 6. Show distribution
# --------------------------------------------------

print("\n========================================")
print("NORMAL BEHAVIOR BASELINE TEST")
print("========================================")

print("\nTotal windows:", len(df))
print("Baseline windows:", len(baseline_samples))

print("\nBaseline:")
print(
    pd.DataFrame({
        "median": baseline_median,
        "IQR": baseline_iqr
    }).to_string()
)

print("\nAnomaly score distribution:")
print(
    df["baseline_anomaly_score"]
    .describe()
    .to_string()
)

# --------------------------------------------------
# 7. Show highest anomaly windows
# --------------------------------------------------

cols = [
    "video_id",
    "window_id",
    "baseline_anomaly_score",
    "abnormal_behavior_score"
]

print("\nTop 20 most anomalous windows:")
print(
    df[cols]
    .sort_values("baseline_anomaly_score", ascending=False)
    .head(20)
    .to_string(index=False)
)

# --------------------------------------------------
# 8. Analyze each video
# --------------------------------------------------

video_summary = (
    df.groupby("video_id")["baseline_anomaly_score"]
    .agg(["count", "mean", "median", "max"])
    .sort_values("mean", ascending=False)
)

print("\n========================================")
print("VIDEO-LEVEL ANOMALY SUMMARY")
print("========================================")

print(video_summary.to_string())