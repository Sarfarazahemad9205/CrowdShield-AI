from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = Path("ml/dataset")

DATASETS = {
    "UMN Crowd": FEATURE_DIR / "umn_crowd_features.csv",
    "Railway Crowd": FEATURE_DIR / "railway_crowd_features.csv",
    "Bengaluru Stampede": FEATURE_DIR / "Bengaluru_Crowd_stampede_features.csv",
}


# Features we are most interested in for crowd-risk analysis
KEY_FEATURES = [
    "people_count",
    "avg_speed",
    "max_speed",
    "speed_std",
    "avg_displacement",
    "max_displacement",
    "avg_acceleration",
    "max_acceleration",
    "acceleration_std",
    "avg_speed_change",
    "max_speed_change",
    "avg_velocity_change",
    "max_velocity_change",
    "direction_alignment",
    "directional_disorder",
    "avg_density",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_datasets():

    datasets = {}

    print("=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    for name, path in DATASETS.items():

        if not path.exists():
            print(f"\nWARNING: File not found: {path}")
            continue

        df = pd.read_csv(path)

        datasets[name] = df

        print(f"\n{name}")
        print(f"File   : {path}")
        print(f"Rows   : {len(df)}")
        print(f"Columns: {len(df.columns)}")

        if "timestamp" in df.columns:
            print(
                f"Time   : "
                f"{df['timestamp'].min():.2f} - "
                f"{df['timestamp'].max():.2f} sec"
            )

    return datasets


# ============================================================
# BASIC INFORMATION
# ============================================================

def basic_comparison(datasets):

    print("\n")
    print("=" * 70)
    print("BASIC DATASET COMPARISON")
    print("=" * 70)

    rows = []

    for name, df in datasets.items():

        duration = (
            df["timestamp"].max()
            if "timestamp" in df.columns
            else np.nan
        )

        rows.append({
            "dataset": name,
            "frames": len(df),
            "duration_sec": duration,
            "avg_people": df["people_count"].mean(),
            "max_people": df["people_count"].max(),
        })

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    return result


# ============================================================
# FEATURE STATISTICS
# ============================================================

def feature_statistics(datasets):

    print("\n")
    print("=" * 70)
    print("FEATURE STATISTICS")
    print("=" * 70)

    all_results = []

    for name, df in datasets.items():

        for feature in KEY_FEATURES:

            if feature not in df.columns:
                continue

            values = df[feature].dropna()

            if len(values) == 0:
                continue

            result = {
                "dataset": name,
                "feature": feature,
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "min": values.min(),
                "p95": values.quantile(0.95),
                "max": values.max(),
            }

            all_results.append(result)

    result_df = pd.DataFrame(all_results)

    print(result_df.to_string(index=False))

    return result_df


# ============================================================
# SIDE-BY-SIDE MEAN COMPARISON
# ============================================================

def mean_comparison(datasets):

    print("\n")
    print("=" * 70)
    print("SIDE-BY-SIDE MEAN COMPARISON")
    print("=" * 70)

    rows = []

    for feature in KEY_FEATURES:

        row = {"feature": feature}

        for name, df in datasets.items():

            if feature in df.columns:
                row[name] = df[feature].mean()
            else:
                row[name] = np.nan

        rows.append(row)

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    return result


# ============================================================
# PERCENTILE COMPARISON
# ============================================================

def percentile_comparison(datasets):

    print("\n")
    print("=" * 70)
    print("95th PERCENTILE COMPARISON")
    print("=" * 70)

    rows = []

    for feature in KEY_FEATURES:

        row = {"feature": feature}

        for name, df in datasets.items():

            if feature in df.columns:
                row[name] = df[feature].quantile(0.95)
            else:
                row[name] = np.nan

        rows.append(row)

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    return result


# ============================================================
# TEMPORAL ANALYSIS
# ============================================================

def temporal_analysis(datasets):

    print("\n")
    print("=" * 70)
    print("TEMPORAL ANALYSIS")
    print("=" * 70)

    results = []

    # Divide each video into 10 normalized time segments
    for name, df in datasets.items():

        df = df.copy()

        df["time_bin"] = pd.qcut(
            df.index,
            q=10,
            labels=False,
            duplicates="drop"
        )

        grouped = df.groupby("time_bin")

        for bin_number, group in grouped:

            row = {
                "dataset": name,
                "time_bin": int(bin_number) + 1,
                "people_count": group["people_count"].mean(),
                "avg_speed": group["avg_speed"].mean(),
                "speed_std": group["speed_std"].mean(),
                "avg_acceleration": group["avg_acceleration"].mean(),
                "directional_disorder":
                    group["directional_disorder"].mean(),
                "avg_displacement":
                    group["avg_displacement"].mean(),
            }

            results.append(row)

    result = pd.DataFrame(results)

    print(result.to_string(index=False))

    return result


# ============================================================
# EXTREME FRAMES
# ============================================================

def extreme_frames(datasets):

    print("\n")
    print("=" * 70)
    print("EXTREME FRAME ANALYSIS")
    print("=" * 70)

    results = []

    extreme_features = [
        "avg_speed",
        "speed_std",
        "avg_acceleration",
        "directional_disorder",
        "people_count",
    ]

    for name, df in datasets.items():

        for feature in extreme_features:

            if feature not in df.columns:
                continue

            idx = df[feature].idxmax()

            row = df.loc[idx]

            results.append({
                "dataset": name,
                "feature": feature,
                "frame": row.get("frame", np.nan),
                "timestamp": row.get("timestamp", np.nan),
                "value": row[feature],
                "people_count": row.get(
                    "people_count", np.nan
                ),
            })

    result = pd.DataFrame(results)

    print(result.to_string(index=False))

    return result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    basic,
    statistics,
    means,
    percentiles,
    temporal,
    extremes
):

    output_dir = FEATURE_DIR

    basic.to_csv(
        output_dir / "comparison_basic.csv",
        index=False
    )

    statistics.to_csv(
        output_dir / "comparison_statistics.csv",
        index=False
    )

    means.to_csv(
        output_dir / "comparison_means.csv",
        index=False
    )

    percentiles.to_csv(
        output_dir / "comparison_percentiles.csv",
        index=False
    )

    temporal.to_csv(
        output_dir / "temporal_comparison.csv",
        index=False
    )

    extremes.to_csv(
        output_dir / "extreme_frames.csv",
        index=False
    )

    print("\n")
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print("\nFiles created:")

    print("ml/dataset/comparison_basic.csv")
    print("ml/dataset/comparison_statistics.csv")
    print("ml/dataset/comparison_means.csv")
    print("ml/dataset/comparison_percentiles.csv")
    print("ml/dataset/temporal_comparison.csv")
    print("ml/dataset/extreme_frames.csv")


# ============================================================
# MAIN
# ============================================================

def main():

    datasets = load_datasets()

    if len(datasets) == 0:
        print("\nNo datasets found.")
        return

    basic = basic_comparison(datasets)

    statistics = feature_statistics(datasets)

    means = mean_comparison(datasets)

    percentiles = percentile_comparison(datasets)

    temporal = temporal_analysis(datasets)

    extremes = extreme_frames(datasets)

    save_results(
        basic,
        statistics,
        means,
        percentiles,
        temporal,
        extremes
    )

    print("\n")
    print("=" * 70)
    print("COMPARATIVE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()