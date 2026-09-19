from pathlib import Path
import pandas as pd


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = BASE_DIR / "ml" / "dataset"

OUTPUT_FILE = DATASET_DIR / "training_features.csv"


# ==========================================
# FIND TRAINING FEATURE FILES
# ==========================================

feature_files = sorted(
    DATASET_DIR.glob("*_features.csv")
)

# Keep only 01_features.csv to 16_features.csv
feature_files = [
    file for file in feature_files
    if file.stem.split("_")[0].isdigit()
    and 1 <= int(file.stem.split("_")[0]) <= 16
]


print("=" * 60)
print("COMBINING TRAINING FEATURE DATASETS")
print("=" * 60)

print(f"\nFiles found: {len(feature_files)}")

for file in feature_files:
    print(file.name)


# ==========================================
# COMBINE FILES
# ==========================================

all_data = []

for file in feature_files:

    print(f"\nReading: {file.name}")

    df = pd.read_csv(file)

    # Extract video number
    video_id = file.stem.split("_")[0]

    # Add video identifier
    df["video_id"] = video_id

    all_data.append(df)

    print(f"Rows: {len(df)}")


# ==========================================
# MERGE
# ==========================================

training_df = pd.concat(
    all_data,
    ignore_index=True
)


# ==========================================
# SAVE
# ==========================================

training_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("COMBINATION COMPLETED")
print("=" * 60)

print(f"\nTotal rows: {len(training_df)}")
print(f"Total columns: {len(training_df.columns)}")

print("\nRows per video:")

print(
    training_df["video_id"]
    .value_counts()
    .sort_index()
)

print("\nOutput:")
print(OUTPUT_FILE)

print("=" * 60)