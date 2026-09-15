from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil

from app.video_processor import process_video


app = FastAPI(
    title="StampedeGuard AI",
    description="AI-Based Crowd Monitoring and Stampede Risk Early-Warning System",
    version="1.0.0"
)


# Directories
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")

UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)


# Allowed video formats
ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}


# Serve uploaded and processed videos
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


@app.get("/")
def root():
    return {
        "message": "StampedeGuard AI backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected"
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid video format. Allowed formats: MP4, AVI, MOV, MKV"
        )

    filename = Path(file.filename).name

    file_path = UPLOAD_DIR / filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Video uploaded successfully",
        "filename": filename,
        "video_url": f"/uploads/{filename}"
    }


@app.post("/process-video/{filename}")
def process_uploaded_video(filename: str):

    input_path = UPLOAD_DIR / filename

    if not input_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video not found"
        )

    output_filename = f"{input_path.stem}_processed.mp4"

    output_path = PROCESSED_DIR / output_filename

    try:

        process_video(
            input_path,
            output_path
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )

    return {
        "message": "Video processed successfully",
        "filename": output_filename,
        "video_url": f"/processed/{output_filename}"
    }