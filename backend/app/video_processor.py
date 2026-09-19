import cv2
import subprocess
import csv
import math
from pathlib import Path
from ultralytics import YOLO


# Load YOLO model
model = YOLO("yolo11n.pt")


def process_video(input_path: Path, output_path: Path):

    # Temporary file created by OpenCV
    raw_output_path = output_path.with_name(
        f"{output_path.stem}_raw.mp4"
    )

    # CSV file for extracted tracking data
    csv_output_path = output_path.with_name(
        f"{output_path.stem}_data.csv"
    )

    # Open input video
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise ValueError("Could not open the input video.")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Total frame area
    frame_area = width * height

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary OpenCV output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(raw_output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the output video.")

    # Store previous position of each person
    previous_positions = {}

    # Create CSV file
    with open(
        csv_output_path,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        csv_writer = csv.writer(csv_file)

        # CSV column names
        csv_writer.writerow([
            "frame",
            "timestamp",
            "person_id",
            "x",
            "y",
            "width",
            "height",
            "movement_x",
            "movement_y",
            "displacement",
            "velocity_x",
            "velocity_y",
            "speed",
            "direction",
            "movement_state",
            "people_count",
            "density"
        ])

        # Process frames
        frame_number = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Current timestamp
            timestamp = frame_number / fps

            # Run YOLO tracking
            results = model.track(
             frame,
             persist=True,
             classes=[0],
              tracker="bytetrack.yaml",
              imgsz=1280,
            conf=0.15,
              verbose=False 
            )
            # Get tracking result
            result = results[0]

            # Copy original frame
            annotated_frame = frame.copy()

            # Number of people currently detected
            people_count = 0

            # Check if tracking IDs are available
            if result.boxes.id is not None:

                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)

                people_count = len(track_ids)

                for box, track_id in zip(boxes, track_ids):

                    x1, y1, x2, y2 = map(int, box)

                    # Bounding box dimensions
                    box_width = x2 - x1
                    box_height = y2 - y1

                    # Center point of person
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Default values
                    movement_x = 0
                    movement_y = 0
                    displacement = 0
                    velocity_x = 0
                    velocity_y = 0
                    speed = 0
                    direction = "STATIONARY"
                    movement_state = "STATIONARY"

                    # Check whether we have previous position
                    if track_id in previous_positions:

                        previous_x, previous_y, previous_time = (
                            previous_positions[track_id]
                        )

                        # Position change
                        movement_x = center_x - previous_x
                        movement_y = center_y - previous_y

                        # Time difference
                        time_difference = timestamp - previous_time

                        if time_difference > 0:

                            # Displacement
                            displacement = math.sqrt(
                                movement_x ** 2 +
                                movement_y ** 2
                            )

                            # Velocity
                            velocity_x = movement_x / time_difference
                            velocity_y = movement_y / time_difference

                            # Speed
                            speed = math.sqrt(
                                velocity_x ** 2 +
                                velocity_y ** 2
                            )

                            # Direction
                            if displacement < 2:
                                direction = "STATIONARY"

                            else:
                                angle = math.degrees(
                                    math.atan2(
                                        movement_y,
                                        movement_x
                                    )
                                )

                                if -22.5 <= angle < 22.5:
                                    direction = "RIGHT"

                                elif 22.5 <= angle < 67.5:
                                    direction = "DOWN-RIGHT"

                                elif 67.5 <= angle < 112.5:
                                    direction = "DOWN"

                                elif 112.5 <= angle < 157.5:
                                    direction = "DOWN-LEFT"

                                elif angle >= 157.5 or angle < -157.5:
                                    direction = "LEFT"

                                elif -157.5 <= angle < -112.5:
                                    direction = "UP-LEFT"

                                elif -112.5 <= angle < -67.5:
                                    direction = "UP"

                                elif -67.5 <= angle < -22.5:
                                    direction = "UP-RIGHT"

                            # Movement state
                            if speed < 10:
                                movement_state = "SLOW"

                            elif speed < 50:
                                movement_state = "MOVING"

                            else:
                                movement_state = "FAST"

                    # Save current position for next frame
                    previous_positions[track_id] = (
                        center_x,
                        center_y,
                        timestamp
                    )

                    # Crowd density
                    density = people_count / frame_area

                    # Write data to CSV
                    csv_writer.writerow([
                        frame_number,
                        round(timestamp, 3),
                        track_id,
                        round(center_x, 2),
                        round(center_y, 2),
                        box_width,
                        box_height,
                        round(movement_x, 2),
                        round(movement_y, 2),
                        round(displacement, 2),
                        round(velocity_x, 2),
                        round(velocity_y, 2),
                        round(speed, 2),
                        direction,
                        movement_state,
                        people_count,
                        round(density, 8)
                    ])

                    # Draw bounding box
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )

                    # Draw tracking ID
                    cv2.putText(
                        annotated_frame,
                        f"ID: {track_id}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

            # Display people count on video
            cv2.putText(
                annotated_frame,
                f"People: {people_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Write processed frame
            writer.write(annotated_frame)

    # Release resources
    cap.release()
    writer.release()

    # Convert OpenCV video to browser-compatible H.264 MP4
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i", str(raw_output_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]

    result = subprocess.run(
        ffmpeg_command,
        capture_output=True,
        text=True
    )

    # Check whether FFmpeg succeeded
    if result.returncode != 0:

        raw_output_path.unlink(missing_ok=True)

        raise ValueError(
            f"FFmpeg conversion failed:\n{result.stderr}"
        )

    # Delete temporary OpenCV video
    raw_output_path.unlink(missing_ok=True)

    print(f"Processed video: {output_path}")
    print(f"Tracking data: {csv_output_path}")

    return output_path