from pathlib import Path
from app.video_processor import process_video


VIDEOS_DIR = Path("videos")
PROCESSED_DIR = Path("processed")


VIDEOS = [
    "umn_crowd.avi",
    "railway_crowd.mp4",
    "Bengaluru_Crowd_stampede.mp4",
]


def main():
    PROCESSED_DIR.mkdir(exist_ok=True)

    for video_name in VIDEOS:
        input_path = VIDEOS_DIR / video_name

        if not input_path.exists():
            print(f"\n[ERROR] Video not found: {input_path}")
            continue

        output_path = PROCESSED_DIR / f"{input_path.stem}_processed.mp4"

        print("\n==========================================")
        print(f"Processing: {video_name}")
        print("==========================================")

        try:
            process_video(
                input_path,
                output_path
            )

            print(f"[SUCCESS] Finished: {video_name}")

        except Exception as e:
            print(f"[FAILED] {video_name}")
            print(f"Reason: {e}")


if __name__ == "__main__":
    main()