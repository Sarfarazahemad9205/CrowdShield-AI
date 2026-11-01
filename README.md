# 🚨 CrowdShield-AI
### AI-Powered Crowd Monitoring & Stampede Risk Early-Warning System

CrowdShield-AI is an AI-powered crowd monitoring and stampede-risk assessment system that uses Computer Vision and Machine Learning to analyze crowd behavior from video footage.

The system integrates **YOLO11n, ByteTrack, OpenCV, and a Machine Learning-based risk analysis pipeline** to detect and track individuals, extract movement features, analyze behavioral patterns, and classify crowd conditions into different risk levels.

## 🎯 Key Features

- Person detection using YOLO11n.
- Multi-object tracking using ByteTrack.
- Video frame processing using OpenCV.
- Automatic generation of annotated videos.
- Tracking information extraction and CSV generation.
- Basic, temporal, and directional feature extraction.
- Time-window-based crowd behavior analysis.
- Anomaly detection and risk assessment.
- Crowd risk classification: NORMAL, HIGH RISK, and CRITICAL.
- Interactive React dashboard.
- FastAPI backend for frontend and processing integration.

## 🛠️ Technology Stack

### Frontend

| Technology | Purpose |
|---|---|
| React.js | Building the interactive dashboard |
| JavaScript | Frontend application logic |
| Vite | Frontend development server and build tool |
| Axios | HTTP communication with backend APIs |
| HTML5 / CSS3 | Structure and styling of the dashboard |

### Backend

| Technology | Purpose |
|---|---|
| Python | Core backend and AI/ML implementation |
| FastAPI | Building REST APIs |
| Uvicorn | Running the FastAPI application |

### Computer Vision & AI/ML

| Technology | Purpose |
|---|---|
| YOLO11n | Person detection in video frames |
| ByteTrack | Multi-object tracking and ID assignment |
| OpenCV | Video reading, frame extraction and annotation |
| NumPy | Numerical and mathematical computations |
| Pandas | Tracking data processing and CSV operations |
| Machine Learning | Crowd behavior analysis and risk classification |

### Video Processing & Development Tools

| Technology | Purpose |
|---|---|
| FFmpeg | Video processing and media handling |
| Git | Version control |
| GitHub | Source code hosting and project management |
| Python Virtual Environment | Isolated Python dependencies |

---

## 🏗️ Project Architecture

```text
CrowdShield-AI/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── video_processor.py
│   │
│   ├── videos/
│   ├── tracking/
│   ├── features/
│   ├── models/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app.jsx
│   │   └── api.js
│   │
│   ├── public/
│   └── package.json
│
├── .gitignore
└── README.md
```

*Illustrative structure: adjust the subfolder names to match your actual repository.*

## 🔄 System Workflow

```text
       Input Video
            |
            ▼
      React Dashboard
            |
            ▼
        FastAPI
         Backend
            |
            ▼
     Video Processor
            |
            ▼
       OpenCV
    Frame Extraction
            |
            ▼
        YOLO11n
    Person Detection
            |
            ▼
        ByteTrack
     Person Tracking
            |
            ▼
     Tracking Outputs
        /       \
       ▼         ▼
  Annotated    Tracking
    Video        CSV
                   |
                   ▼
            Feature Extraction
                   |
                   ▼
            Temporal Analysis
                   |
                   ▼
              Time Windows
                   |
                   ▼
             Risk Engine
                   |
                   ▼
           Risk Classification
                   |
                   ▼
            React Dashboard
```

---

# ⚙️ Installation & Execution Guide

Follow these steps to run CrowdShield-AI locally.

## 1. Prerequisites

Install the following software before starting:

| Software | Requirement |
|---|---|
| Python | 3.10+ (use the version compatible with your dependencies) |
| Node.js | LTS version |
| npm | Installed with Node.js |
| FFmpeg | Installed and added to system PATH |
| Git | Latest stable version |

Verify installations:

```bash
python --version
node --version
npm --version
git --version
ffmpeg -version
```

## 2. Clone the Repository

Open your terminal or VS Code terminal.

```bash
git clone https://github.com/sarfarazahemad9205/crowdshield-ai.git
```

Navigate into the project:

```bash
cd crowdshield-ai
```

## 3. Backend Setup

Open the backend directory:

```bash
cd backend
```

### Step 1: Create a Python Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

The backend dependencies include the Python packages required for video processing, computer vision, tracking, numerical operations, machine learning and API development.

### Step 4: Configure FFmpeg

Make sure FFmpeg is installed and available through your system PATH.

Verify:

```bash
ffmpeg -version
```

### Step 5: Start the FastAPI Backend

From the `backend` directory:

```bash
uvicorn app.main:app --reload
```

The backend should be accessible at:

```text
http://127.0.0.1:8000
```

FastAPI interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Keep this terminal running.

---

## 4. Frontend Setup

Open a **new terminal** in VS Code.

Navigate to the frontend directory:

```bash
cd crowdshield-ai/frontend
```

### Step 1: Install Frontend Dependencies

```bash
npm install
```

### Step 2: Start the React Development Server

```bash
npm run dev
```

Vite will display a local development URL, typically:

```text
http://localhost:5173
```

Open this URL in your browser to access the CrowdShield-AI dashboard.

---

## 5. Running the Complete Application

You need two terminals running simultaneously.

**Terminal 1 – Backend**

```bash
cd crowdshield-ai/backend

venv\Scripts\activate

uvicorn app.main:app --reload
```

**Terminal 2 – Frontend**

```bash
cd crowdshield-ai/frontend

npm run dev
```

Once both servers are running:

1. Open the React dashboard in your browser.
2. Select and upload a video.
3. The frontend communicates with the FastAPI backend.
4. The backend processes the video using the computer vision pipeline.
5. YOLO11n detects people and ByteTrack tracks their movement.
6. Tracking data and annotated video are generated.
7. The risk analysis engine evaluates crowd behavior.
8. The dashboard displays the analysis results.

**Note:** The frontend API base URL must point to the running backend, and any required model weights, configuration files and input/output directories must be available.

---

## 🧠 Feature Extraction & Risk Analysis

The project uses tracking information to extract movement-related features.

| Feature Category | Description |
|---|---|
| Basic Features | Fundamental movement and tracking information |
| Temporal Features | Behavioral changes over time |
| Directional Features | Direction and movement patterns |

The extracted features are analyzed using time-window-based processing to understand crowd behavior over sequences of video frames.

The risk engine produces three classifications:

| Risk Level | Description |
|---|---|
| 🟢 NORMAL | No significant abnormal behavior identified |
| 🟡 HIGH RISK | Potentially concerning behavior detected |
| 🔴 CRITICAL | Severe abnormal behavior requiring attention |

---

## ⚠️ Challenges & Limitations

- Detection accuracy can decrease in highly dense crowds because of occlusion.
- Tracking IDs can switch when individuals overlap.
- Pixel-based movement measurements depend on camera perspective.
- Processing performance depends on hardware capabilities and video resolution.
- Risk assessment requires validation against diverse real-world crowd scenarios.

## 🚀 Future Enhancements

- Live CCTV stream integration.
- Improved crowd density estimation.
- Better detection and tracking in highly congested environments.
- Multi-camera monitoring.
- Optimized real-time inference.
- Automated emergency alert notifications.

---

## 👨‍💻 Author

**Sarfaraz Ahamed**

Computer Science and Engineering Student | AI/ML & Full-Stack Developer

- GitHub: [Sarfarazahemad9205](https://github.com/sarfarazahemad9205)
- Repository: [CrowdShield-AI](https://github.com/sarfarazahemad9205/crowdshield-ai)

---

⭐ If you find this project interesting, consider giving the repository a star!
