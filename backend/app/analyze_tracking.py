import pandas as pd

# Load the generated tracking CSV
df = pd.read_csv("processed/stampede_video_processed_data.csv")

print("===== BASIC INFORMATION =====")
print("Total rows:", len(df))
print("Unique person IDs:", df["person_id"].nunique())
print("Total frames:", df["frame"].nunique())

print("\n===== PERSON ID RANGE =====")
print("Minimum ID:", df["person_id"].min())
print("Maximum ID:", df["person_id"].max())

# Number of frames each person ID appears in
track_lengths = df.groupby("person_id")["frame"].nunique()

print("\n===== TRACK LENGTH =====")
print("Average frames per ID:", round(track_lengths.mean(), 2))
print("Maximum frames for one ID:", track_lengths.max())
print("Minimum frames for one ID:", track_lengths.min())

print("\n===== TRACK DISTRIBUTION =====")
print("IDs appearing only once:",
      (track_lengths == 1).sum())

print("IDs appearing <= 5 frames:",
      (track_lengths <= 5).sum())

print("IDs appearing > 30 frames:",
      (track_lengths > 30).sum())

print("IDs appearing > 100 frames:",
      (track_lengths > 100).sum())

print("\n===== TOP 20 LONGEST TRACKS =====")

print(
    track_lengths
    .sort_values(ascending=False)
    .head(20)
)

print("\n===== SAMPLE IDS =====")

sample_frames = [
    df["frame"].min(),
    int(df["frame"].max() * 0.25),
    int(df["frame"].max() * 0.50),
    int(df["frame"].max() * 0.75),
    df["frame"].max()
]
for frame in sample_frames:
    ids = df[df["frame"] == frame]["person_id"].unique()

    print(f"\nFrame {frame}")
    print("IDs:", ids)

print("\n===== CSV COLUMNS =====")
print(df.columns.tolist())