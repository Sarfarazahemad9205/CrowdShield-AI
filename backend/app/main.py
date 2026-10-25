import csv
import os
from pathlib import Path
import shutil
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.video_processor import process_video
from app.processing_options import get_processing_options
from app.model_runtime import yolo_runtime
from ml.inference.live_risk_engine import analyze_tracking_csv


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="StampedeGuard AI",
    description="AI-Based Crowd Monitoring and Stampede Risk Early-Warning System",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

configured_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *configured_origins,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BACKEND_DIR / "uploads"
PROCESSED_DIR = BACKEND_DIR / "processed"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ALLOWED VIDEO FORMATS
# ============================================================

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}


# ============================================================
# SERVE VIDEO FILES
# ============================================================

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads"
)

app.mount(
    "/processed",
    StaticFiles(directory=str(PROCESSED_DIR)),
    name="processed"
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "StampedeGuard AI backend is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# UPLOAD VIDEO
# ============================================================

@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected"
        )

    # --------------------------------------------------------
    # Check extension
    # --------------------------------------------------------

    extension = (
        Path(file.filename)
        .suffix
        .lower()
    )

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid video format. "
                "Allowed formats: MP4, AVI, MOV, MKV"
            )
        )

    # --------------------------------------------------------
    # Safe filename
    # --------------------------------------------------------

    filename = Path(
        file.filename
    ).name

    file_path = (
        UPLOAD_DIR /
        filename
    )

    # --------------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------------

    try:

        with file_path.open("wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save video: {str(e)}"
        )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "message": "Video uploaded successfully",

        "filename": filename,

        "video_url": f"/uploads/{filename}"
    }


# ============================================================
# PROCESS VIDEO + RISK ANALYSIS
# ============================================================

@app.post("/process-video/{filename}")
def process_uploaded_video(
    filename: str,
    profile: Literal["fast", "balanced", "accurate"] = Query(
        default="balanced",
        description="fast samples every other frame; accurate processes all frames at higher resolution",
    ),
    annotate: bool = Query(
        default=False,
        description="Set true to render and encode a downloadable annotated MP4",
    ),
):

    # ========================================================
    # 1. FIND UPLOADED VIDEO
    # ========================================================

    safe_filename = Path(filename).name
    if safe_filename != filename:
        raise HTTPException(status_code=400, detail="Invalid video filename")

    input_path = (
        UPLOAD_DIR /
        safe_filename
    )

    if not input_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Video not found"
        )


    # ========================================================
    # 2. CREATE PROCESSED VIDEO PATH
    # ========================================================

    output_filename = (
        f"{input_path.stem}_processed.mp4"
    )

    output_path = (
        PROCESSED_DIR /
        output_filename
    )


    # ========================================================
    # 3. PROCESS VIDEO
    #
    # YOLO11n + ByteTrack
    # ========================================================

    try:

        options = get_processing_options(profile)

        process_video(
            input_path,
            output_path,
            options=options,
            annotate=annotate,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Video processing failed: "
                f"{str(e)}"
            )
        )


    # ========================================================
    # 4. FIND TRACKING CSV
    # ========================================================

    tracking_csv = (
        output_path.parent /
        f"{output_path.stem}_data.csv"
    )

    if not tracking_csv.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "Tracking CSV was not generated "
                "after video processing."
            )
        )


    # ========================================================
    # 5. RUN LIVE RISK ANALYSIS
    #
    # Tracking CSV
    #       ↓
    # Tracking reliability
    #       ↓
    # Frame features
    #       ↓
    # 5-second windows
    #       ↓
    # Behavioral features
    #       ↓
    # Normal reference
    #       ↓
    # Movement intensity
    #       ↓
    # Risk state
    # ========================================================

    try:
        with tracking_csv.open("r", newline="", encoding="utf-8") as csv_file:
            rows = csv.reader(csv_file)
            next(rows, None)  # header
            contains_detections = next(rows, None) is not None

        if contains_detections:
            risk_result = analyze_tracking_csv(tracking_csv)
        else:
            risk_result = {
                "risk_state": "TRACKING_UNRELIABLE",
                "risk_reason": "No people could be tracked in this video. Try Slow mode, improve the camera view, or use a clearer clip.",
                "movement_intensity_index": 0.0,
                "tracking_reliability": 0.0,
                "tracking_quality": "UNRELIABLE",
                "window_id": 0,
                "total_windows": 0,
                "windows": [],
            }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Risk analysis failed: "
                f"{str(e)}"
            )
        )


    # ========================================================
    # 6. FINAL RESPONSE
    # ========================================================

    return {

        "message":
            "Video processed and risk analysis completed",

        "filename":
            output_filename if annotate else None,

        "video_url":
            f"/processed/{output_filename}" if annotate else None,

        "processing": {
            "profile": profile,
            "image_size": options.image_size,
            "frame_stride": options.frame_stride,
            "device": f"cuda:{yolo_runtime.device}" if yolo_runtime.device != "cpu" else "cpu",
            "annotated_video": annotate,
        },

        "tracking_csv":
            str(tracking_csv),

        "risk":
            risk_result
    }
