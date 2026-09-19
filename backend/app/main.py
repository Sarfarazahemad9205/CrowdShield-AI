from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil

from app.video_processor import process_video
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
# DIRECTORIES
# ============================================================

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")

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
    StaticFiles(directory="uploads"),
    name="uploads"
)

app.mount(
    "/processed",
    StaticFiles(directory="processed"),
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

        "message":
            "Video uploaded successfully",

        "filename":
            filename,

        "video_url":
            f"/uploads/{filename}"
    }


# ============================================================
# PROCESS VIDEO + RISK ANALYSIS
# ============================================================

@app.post("/process-video/{filename}")
def process_uploaded_video(
    filename: str
):

    # ========================================================
    # 1. FIND UPLOADED VIDEO
    # ========================================================

    input_path = (
        UPLOAD_DIR /
        filename
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

        process_video(
            input_path,
            output_path
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

        risk_result = (
            analyze_tracking_csv(
                tracking_csv
            )
        )

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
            output_filename,

        "video_url":
            f"/processed/{output_filename}",

        "tracking_csv":
            str(tracking_csv),

        "risk":
            risk_result
    }