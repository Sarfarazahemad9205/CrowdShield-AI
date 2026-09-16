import pandas as pd
from pathlib import Path


# ============================================================
# FIND ALL TRACKING CSV FILES
# ============================================================

PROCESSED_DIR = Path("processed")

tracking_files = sorted(
    PROCESSED_DIR.glob("*_processed_data.csv")
)

print("=" * 60)
print("TRACKING DATA ANALYSIS")
print("=" * 60)

print("Tracking files found:", len(tracking_files))

for file in tracking_files:
    print(" -", file.name)


# ============================================================
# ANALYZE EACH VIDEO
# ============================================================

for file in tracking_files:

    print("\n")
    print("=" * 60)
    print(f"ANALYZING: {file.name}")
    print("=" * 60)

    # Load CSV
    df = pd.read_csv(file)

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    print("\n===== BASIC INFORMATION =====")

    print("Total rows:", len(df))

    print("Unique person IDs:",
          df["person_id"].nunique())

    print("Total frames:",
          df["frame"].nunique())

    # --------------------------------------------------------
    # PERSON ID RANGE
    # --------------------------------------------------------

    print("\n===== PERSON ID RANGE =====")

    print("Minimum ID:",
          df["person_id"].min())

    print("Maximum ID:",
          df["person_id"].max())

    # --------------------------------------------------------
    # TRACK LENGTH
    # --------------------------------------------------------

    track_lengths = (
        df.groupby("person_id")["frame"]
        .nunique()
    )

    print("\n===== TRACK LENGTH =====")

    print("Average frames per ID:",
          round(track_lengths.mean(), 2))

    print("Maximum frames for one ID:",
          track_lengths.max())

    print("Minimum frames for one ID:",
          track_lengths.min())

    # --------------------------------------------------------
    # TRACK DISTRIBUTION
    # --------------------------------------------------------

    print("\n===== TRACK DISTRIBUTION =====")

    print("IDs appearing only once:",
          (track_lengths == 1).sum())

    print("IDs appearing <= 5 frames:",
          (track_lengths <= 5).sum())

    print("IDs appearing > 30 frames:",
          (track_lengths > 30).sum())

    print("IDs appearing > 100 frames:",
          (track_lengths > 100).sum())

    # --------------------------------------------------------
    # TOP 20 LONGEST TRACKS
    # --------------------------------------------------------

    print("\n===== TOP 20 LONGEST TRACKS =====")

    print(
        track_lengths
        .sort_values(ascending=False)
        .head(20)
    )

    # --------------------------------------------------------
    # SAMPLE FRAMES
    # --------------------------------------------------------

    print("\n===== SAMPLE IDS =====")

    min_frame = df["frame"].min()
    max_frame = df["frame"].max()

    sample_frames = [
        min_frame,
        int(max_frame * 0.25),
        int(max_frame * 0.50),
        int(max_frame * 0.75),
        max_frame
    ]

    # Remove duplicate frame numbers
    sample_frames = sorted(set(sample_frames))

    for frame in sample_frames:

        ids = (
            df[df["frame"] == frame]["person_id"]
            .unique()
        )

        print(f"\nFrame {frame}")

        print("IDs:", ids)

    # --------------------------------------------------------
    # CSV COLUMNS
    # --------------------------------------------------------

    print("\n===== CSV COLUMNS =====")

    print(df.columns.tolist())


print("\n")
print("=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)