from pathlib import Path
from video_processor import process_video


# ==============================
# PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAINING_DIR = BASE_DIR / "videos" / "training"
PROCESSED_DIR = BASE_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# SUPPORTED VIDEO FORMATS
# ==============================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}


# ==============================
# FIND TRAINING VIDEOS
# ==============================

videos = sorted(
    [
        file
        for file in TRAINING_DIR.iterdir()
        if file.suffix.lower() in VIDEO_EXTENSIONS
    ]
)


if not videos:
    print("No training videos found.")
    print(f"Put your videos inside: {TRAINING_DIR}")
    exit()


print("=" * 60)
print("STAMPedeGUARD - TRAINING VIDEO PROCESSING")
print("=" * 60)

print(f"\nTraining folder : {TRAINING_DIR}")
print(f"Videos found    : {len(videos)}")

print("\nVideos:")

for i, video in enumerate(videos, start=1):
    print(f"{i}. {video.name}")


# ==============================
# PROCESS EACH VIDEO
# ==============================

for index, video_path in enumerate(videos, start=1):

    print("\n" + "=" * 60)
    print(f"PROCESSING VIDEO {index}/{len(videos)}")
    print(f"File: {video_path.name}")
    print("=" * 60)

    output_name = f"{video_path.stem}_processed.mp4"
    output_path = PROCESSED_DIR / output_name

    try:

        process_video(
            video_path,
            output_path
        )

        print(f"\n✓ Completed: {video_path.name}")

    except Exception as e:

        print(f"\n✗ Failed: {video_path.name}")
        print(f"Error: {e}")

        # Continue with the next video
        continue


# ==============================
# FINISHED
# ==============================

print("\n" + "=" * 60)
print("ALL TRAINING VIDEOS PROCESSED")
print("=" * 60)

print(f"\nProcessed files are in:")
print(PROCESSED_DIR)