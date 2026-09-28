import { useState } from "react";

import {
  Activity,
  BarChart3,
  Bell,
  Camera,
  CheckCircle2,
  History,
  LayoutDashboard,
  Menu,
  ShieldCheck,
  Upload,
  Video,
  XCircle,
} from "lucide-react";

import { uploadVideo, processVideo } from "./services/api";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [analysis, setAnalysis] = useState(null);

  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");

  // --------------------------------------------------
  // Select video
  // --------------------------------------------------

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];

    if (!file) return;

    // Only allow video files
    if (!file.type.startsWith("video/")) {
      setError("Please select a valid video file.");
      return;
    }

    setSelectedFile(file);
    setAnalysis(null);
    setError("");

    // Create local video preview
    const previewUrl = URL.createObjectURL(file);
    setVideoUrl(previewUrl);
  };

  // --------------------------------------------------
  // Upload + Process video
  // --------------------------------------------------

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select a video first.");
      return;
    }

    try {
      setError("");
      setAnalysis(null);

      // Upload
      setUploading(true);

      const uploadResult = await uploadVideo(selectedFile);

      setUploading(false);

      // Process
      setProcessing(true);

      const result = await processVideo(uploadResult.filename);

      setAnalysis(result);

      setProcessing(false);
    } catch (err) {
      console.error("Video processing error:", err);

      setUploading(false);
      setProcessing(false);

      setError(
        err.response?.data?.detail ||
          "Something went wrong while processing the video."
      );
    }
  };

  // --------------------------------------------------
  // Risk information
  // --------------------------------------------------

  const risk = analysis?.risk || "NORMAL";

  const getRiskStyles = () => {
    switch (risk) {
      case "CRITICAL":
        return {
          color: "text-red-400",
          bg: "bg-red-500/10",
          border: "border-red-500/20",
          icon: <XCircle size={32} className="text-red-400" />,
        };

      case "HIGH_RISK":
        return {
          color: "text-yellow-400",
          bg: "bg-yellow-500/10",
          border: "border-yellow-500/20",
          icon: <Activity size={32} className="text-yellow-400" />,
        };

      default:
        return {
          color: "text-emerald-400",
          bg: "bg-emerald-500/10",
          border: "border-emerald-500/20",
          icon: <CheckCircle2 size={32} className="text-emerald-400" />,
        };
    }
  };

  const riskStyles = getRiskStyles();

  // --------------------------------------------------
  // UI
  // --------------------------------------------------

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">

      {/* ============================================== */}
      {/* SIDEBAR */}
      {/* ============================================== */}

      <aside className="fixed left-0 top-0 hidden h-screen w-64 border-r border-slate-800 bg-slate-900/80 md:block">

        {/* Logo */}
        <div className="flex h-20 items-center gap-3 border-b border-slate-800 px-6">

          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600">
            <ShieldCheck size={23} />
          </div>

          <div>
            <h1 className="font-semibold tracking-tight">
              StampedeGuard
            </h1>

            <p className="text-xs text-slate-500">
              AI Safety System
            </p>
          </div>

        </div>

        {/* Navigation */}
        <nav className="space-y-2 p-4">

          <NavItem
            icon={<LayoutDashboard size={19} />}
            label="Dashboard"
            active
          />

          <NavItem
            icon={<Activity size={19} />}
            label="Live Monitor"
          />

          <NavItem
            icon={<BarChart3 size={19} />}
            label="Analysis"
          />

          <NavItem
            icon={<History size={19} />}
            label="History"
          />

        </nav>

        {/* System status */}
        <div className="absolute bottom-0 w-full border-t border-slate-800 p-4">

          <div className="rounded-xl bg-slate-800/60 p-4">

            <div className="mb-2 flex items-center gap-2">

              <span className="h-2 w-2 rounded-full bg-emerald-400" />

              <span className="text-sm font-medium">
                System Online
              </span>

            </div>

            <p className="text-xs leading-5 text-slate-500">
              AI monitoring system is ready for analysis.
            </p>

          </div>

        </div>

      </aside>

      {/* ============================================== */}
      {/* MAIN */}
      {/* ============================================== */}

      <main className="md:ml-64">

        {/* HEADER */}
        <header className="flex h-20 items-center justify-between border-b border-slate-800 bg-slate-950/90 px-5 md:px-8">

          <div className="flex items-center gap-3">

            <button className="rounded-lg p-2 text-slate-400 hover:bg-slate-800 md:hidden">
              <Menu size={21} />
            </button>

            <div>

              <p className="text-sm text-slate-500">
                Overview
              </p>

              <h2 className="text-lg font-semibold">
                Crowd Risk Monitoring
              </h2>

            </div>

          </div>

          <div className="flex items-center gap-3">

            <button className="rounded-xl border border-slate-800 bg-slate-900 p-2.5 text-slate-400 hover:text-white">
              <Bell size={19} />
            </button>

            <div className="hidden items-center gap-2 rounded-xl border border-slate-800 bg-slate-900 px-3 py-2 sm:flex">

              <span className="h-2 w-2 rounded-full bg-emerald-400" />

              <span className="text-sm text-slate-300">
                Online
              </span>

            </div>

          </div>

        </header>

        {/* ============================================== */}
        {/* DASHBOARD CONTENT */}
        {/* ============================================== */}

        <section className="p-5 md:p-8">

          {/* Page title */}

          <div className="mb-8">

            <h1 className="text-2xl font-bold tracking-tight md:text-3xl">
              Crowd Risk Dashboard
            </h1>

            <p className="mt-2 text-sm text-slate-400">
              Monitor crowd movement and identify abnormal behavior
              using computer vision.
            </p>

          </div>

          {/* ========================================== */}
          {/* MAIN GRID */}
          {/* ========================================== */}

          <div className="grid gap-6 xl:grid-cols-[1.6fr_1fr]">

            {/* ======================================== */}
            {/* VIDEO SECTION */}
            {/* ======================================== */}

            <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900">

              {/* Video header */}

              <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">

                <div className="flex items-center gap-3">

                  <Video
                    size={19}
                    className="text-blue-400"
                  />

                  <div>

                    <h3 className="text-sm font-semibold">
                      Video Monitor
                    </h3>

                    <p className="text-xs text-slate-500">
                      Crowd analysis input
                    </p>

                  </div>

                </div>

                <span
                  className={`rounded-full px-3 py-1 text-xs ${
                    selectedFile
                      ? "bg-blue-500/10 text-blue-400"
                      : "bg-slate-800 text-slate-400"
                  }`}
                >
                  {selectedFile ? "Video Selected" : "No Video"}
                </span>

              </div>

              {/* Video preview */}

              <div className="flex aspect-video items-center justify-center bg-slate-950">

                {videoUrl ? (

                  <video
                    src={videoUrl}
                    controls
                    className="h-full w-full object-contain"
                  />

                ) : (

                  <div className="text-center">

                    <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-900">

                      <Camera
                        size={25}
                        className="text-slate-500"
                      />

                    </div>

                    <p className="text-sm font-medium text-slate-300">
                      No video selected
                    </p>

                    <p className="mt-1 text-xs text-slate-500">
                      Upload a video to begin crowd analysis
                    </p>

                  </div>

                )}

              </div>

              {/* Upload controls */}

              <div className="border-t border-slate-800 p-5">

                <label className="flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-800 px-5 py-3 text-sm font-semibold transition hover:bg-slate-700">

                  <Upload size={18} />

                  Choose Video

                  <input
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={handleFileChange}
                  />

                </label>

                {/* Selected file */}

                {selectedFile && (

                  <div className="mt-3">

                    <p className="truncate text-xs text-slate-400">
                      Selected: {selectedFile.name}
                    </p>

                    <button
                      onClick={handleUpload}
                      disabled={uploading || processing}
                      className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
                    >

                      {uploading
                        ? "Uploading..."
                        : processing
                          ? "Analyzing Crowd..."
                          : "Analyze Video"}

                    </button>

                  </div>

                )}

                {/* Error */}

                {error && (

                  <div className="mt-4 rounded-xl border border-red-500/20 bg-red-500/10 p-3">

                    <p className="text-sm text-red-400">
                      {error}
                    </p>

                  </div>

                )}

                {/* Processing */}

                {processing && (

                  <div className="mt-4 rounded-xl border border-blue-500/20 bg-blue-500/10 p-4">

                    <div className="flex items-center gap-3">

                      <div className="h-5 w-5 animate-spin rounded-full border-2 border-blue-400 border-t-transparent" />

                      <div>

                        <p className="text-sm font-medium text-blue-300">
                          Analyzing crowd movement...
                        </p>

                        <p className="mt-1 text-xs text-blue-400/70">
                          YOLO11n + ByteTrack + Risk Engine
                        </p>

                      </div>

                    </div>

                  </div>

                )}

              </div>

            </div>

            {/* ======================================== */}
            {/* RISK SECTION */}
            {/* ======================================== */}

            <div className="space-y-6">

              {/* Risk card */}

              <div
                className={`rounded-2xl border ${riskStyles.border} bg-slate-900 p-6`}
              >

                <p className="text-sm text-slate-500">
                  Current Risk Level
                </p>

                <div className="mt-5 flex items-center gap-4">

                  <div
                    className={`flex h-16 w-16 items-center justify-center rounded-2xl ${riskStyles.bg}`}
                  >
                    {riskStyles.icon}
                  </div>

                  <div>

                    <h3
                      className={`text-3xl font-bold ${riskStyles.color}`}
                    >
                      {risk}
                    </h3>

                    <p className="mt-1 text-xs text-slate-500">
                      {analysis
                        ? "AI analysis completed"
                        : "Awaiting video analysis"}
                    </p>

                  </div>

                </div>

                {/* Risk reason */}

                <div className="mt-6 rounded-xl bg-slate-950 p-4">

                  <p className="text-xs uppercase tracking-wider text-slate-600">
                    Risk Explanation
                  </p>

                  <p className="mt-2 text-sm leading-6 text-slate-400">

                    {analysis?.reason ||
                      "Upload a video to receive the AI-generated crowd risk assessment."}

                  </p>

                </div>

              </div>

              {/* Statistics */}

              <div className="grid grid-cols-2 gap-4">

                <StatCard
                  label="Tracking Reliability"
                  value={
                    analysis?.tracking_reliability != null
                      ? `${(
                          analysis.tracking_reliability * 100
                        ).toFixed(1)}%`
                      : "—"
                  }
                  description="Tracker reliability"
                />

                <StatCard
                  label="Windows Analyzed"
                  value={
                    analysis?.total_windows ?? "—"
                  }
                  description="5-second windows"
                />

                <StatCard
                  label="Movement Intensity"
                  value={
                    analysis?.movement_intensity != null
                      ? Number(
                          analysis.movement_intensity
                        ).toFixed(3)
                      : "—"
                  }
                  description="Relative to normal"
                />

                <StatCard
                  label="Tracking Quality"
                  value={
                    analysis?.tracking_quality
                      ? analysis.tracking_quality
                      : "—"
                  }
                  description="Analysis quality"
                />

              </div>

            </div>

          </div>

          {/* ========================================== */}
          {/* ANALYSIS OVERVIEW */}
          {/* ========================================== */}

          <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-6">

            <div className="flex items-center justify-between">

              <div>

                <h3 className="font-semibold">
                  Analysis Overview
                </h3>

                <p className="mt-1 text-sm text-slate-500">
                  Behavioral analysis generated by the StampedeGuard
                  risk engine.
                </p>

              </div>

              <BarChart3
                size={22}
                className="text-slate-600"
              />

            </div>

            {analysis ? (

              <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

                <InfoItem
                  label="Risk"
                  value={analysis.risk || "—"}
                />

                <InfoItem
                  label="Peak Movement"
                  value={
                    analysis.movement_intensity != null
                      ? Number(
                          analysis.movement_intensity
                        ).toFixed(3)
                      : "—"
                  }
                />

                <InfoItem
                  label="Tracking Reliability"
                  value={
                    analysis.tracking_reliability != null
                      ? `${(
                          analysis.tracking_reliability * 100
                        ).toFixed(1)}%`
                      : "—"
                  }
                />

                <InfoItem
                  label="Total Windows"
                  value={
                    analysis.total_windows ?? "—"
                  }
                />

              </div>

            ) : (

              <div className="mt-6 rounded-xl bg-slate-950 p-6 text-center">

                <p className="text-sm text-slate-500">
                  Detailed behavioral analysis will appear here
                  after processing.
                </p>

              </div>

            )}

          </div>

        </section>

      </main>

    </div>
  );
}

// ==================================================
// Navigation item
// ==================================================

function NavItem({
  icon,
  label,
  active = false,
}) {
  return (
    <button
      className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm transition ${
        active
          ? "bg-blue-600/10 text-blue-400"
          : "text-slate-400 hover:bg-slate-800 hover:text-slate-100"
      }`}
    >
      {icon}

      <span>{label}</span>
    </button>
  );
}

// ==================================================
// Statistics card
// ==================================================

function StatCard({
  label,
  value,
  description,
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">

      <p className="text-xs text-slate-500">
        {label}
      </p>

      <p className="mt-3 text-2xl font-bold text-slate-100">
        {value}
      </p>

      <p className="mt-1 text-xs text-slate-600">
        {description}
      </p>

    </div>
  );
}

// ==================================================
// Information item
// ==================================================

function InfoItem({
  label,
  value,
}) {
  return (
    <div className="rounded-xl bg-slate-950 p-4">

      <p className="text-xs text-slate-600">
        {label}
      </p>

      <p className="mt-2 text-lg font-semibold text-slate-200">
        {value}
      </p>

    </div>
  );
}

export default App;