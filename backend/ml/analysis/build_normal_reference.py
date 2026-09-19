import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

BEHAVIORAL_DIR = Path("ml/dataset/behavioral")
OUTPUT_DIR = Path("ml/dataset/reference")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "normal_reference_stats.csv"


# ==========================================
# REFERENCE DATASETS
# ==========================================

# Avenue 01-16
avenue_files = sorted(
    BEHAVIORAL_DIR.glob("*_features_behavioral.csv")
)

avenue_files = [
    f for f in avenue_files
    if f.stem.split("_")[0].isdigit()
    and 1 <= int(f.stem.split("_")[0]) <= 16
]

# UMN
umn_file = BEHAVIORAL_DIR / "umn_crowd_features_behavioral.csv"


# ==========================================
# LOAD DATA
# ==========================================

datasets = []

print("\n==========================================")
print("      BUILDING NORMAL REFERENCE")
print("==========================================")

print("\n===== AVENUE DATASETS =====")

for file in avenue_files:
    df = pd.read_csv(file)

    print(
        f"{file.name:<45} "
        f"{len(df):>3} windows"
    )

    df["source"] = file.stem
    datasets.append(df)


print("\n===== UMN DATASET =====")

if umn_file.exists():

    df = pd.read_csv(umn_file)

    print(
        f"{umn_file.name:<45} "
        f"{len(df):>3} windows"
    )

    df["source"] = umn_file.stem
    datasets.append(df)

else:
    print("WARNING: UMN behavioral file not found.")


# ==========================================
# COMBINE
# ==========================================

if not datasets:
    raise RuntimeError(
        "No reference datasets were found."
    )

reference_df = pd.concat(
    datasets,
    ignore_index=True
)

print("\n==========================================")
print("REFERENCE DATASET")
print("==========================================")

print("Total windows:", len(reference_df))
print("Total datasets:", len(datasets))

print("\nWindows by source:")
print(
    reference_df["source"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ==========================================
# FEATURES
# ==========================================

FEATURES = [
    # Crowd structure
    "people_count_mean",
    "people_count_std",
    "avg_density_mean",

    # Movement
    "avg_speed_mean",
    "avg_speed_std",
    "avg_displacement_mean",

    # Crowd coordination
    "direction_alignment_mean",
    "directional_disorder_mean",

    # Behavioral changes
    "avg_speed_change_mean",
    "avg_velocity_change_mean",
]


# ==========================================
# CHECK FEATURES
# ==========================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in reference_df.columns
]

if missing_features:
    print("\nMissing features:")
    for feature in missing_features:
        print(" -", feature)

    raise RuntimeError(
        "Required features are missing."
    )


# ==========================================
# CLEAN VALUES
# ==========================================

data = reference_df[FEATURES].copy()

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

before = len(data)

data = data.dropna()

after = len(data)

print("\n===== CLEANING =====")
print("Rows before cleaning:", before)
print("Rows after cleaning :", after)
print("Rows removed        :", before - after)


# ==========================================
# ROBUST STATISTICS
# ==========================================

statistics = []

for feature in FEATURES:

    values = data[feature]

    median = values.median()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    # Prevent zero IQR from causing
    # division-by-zero later.
    if iqr == 0:
        iqr = 1e-9

    statistics.append(
        {
            "feature": feature,
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "min": values.min(),
            "max": values.max(),
        }
    )


stats_df = pd.DataFrame(statistics)


# ==========================================
# DISPLAY
# ==========================================

print("\n==========================================")
print("NORMAL REFERENCE STATISTICS")
print("==========================================")

print(
    stats_df.to_string(index=False)
)


# ==========================================
# SAVE
# ==========================================

stats_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==========================================")
print("REFERENCE CREATED SUCCESSFULLY")
print("==========================================")

print("Saved to:")
print(OUTPUT_FILE)

print("==========================================")