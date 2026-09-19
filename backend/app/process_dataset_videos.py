from pathlib import Path
from app.video_processor import process_video


# ============================================================
# HAJJV2 TRAINING VIDEOS
# ============================================================

VIDEOS_DIR = Path("videos/hajj_training")
PROCESSED_DIR = Path("processed")


VIDEOS = [
    "2.mp4",
    "3.mp4",
    "5.mp4",
    "7.mp4",
    "8.mp4",
    "9.mp4",
    "10.mp4",
    "11.mp4",
    "12.mp4",
]


# ============================================================
# MAIN
# ============================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("       HAJJV2 TRAINING VIDEO PROCESSING")
    print("=" * 60)

    for video_name in VIDEOS:

        input_path = VIDEOS_DIR / video_name

        if not input_path.exists():

            print(f"\n[ERROR] Video not found: {input_path}")
            continue

        output_path = (
            PROCESSED_DIR /
            f"hajj_{input_path.stem}_processed.mp4"
        )

        print("\n" + "=" * 60)
        print(f"Processing: {video_name}")
        print("=" * 60)

        try:

            process_video(
                input_path,
                output_path
            )

            print(f"[SUCCESS] Finished: {video_name}")

        except Exception as e:

            print(f"[FAILED] {video_name}")
            print(f"Reason: {e}")

    print("\n" + "=" * 60)
    print("       HAJJV2 PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()