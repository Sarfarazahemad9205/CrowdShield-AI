import { useEffect, useState } from "react";
import { ArrowRight, Camera, LoaderCircle } from "lucide-react";
import AnalysisResults from "./AnalysisResults";
import AnalysisSettings from "./AnalysisSettings";
import { checkBackendHealth, getApiErrorMessage, processVideo, uploadVideo } from "../../services/api";

export default function VideoUploader() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [profile, setProfile] = useState("balanced");
  const [annotate, setAnnotate] = useState(false);
  const [backendReady, setBackendReady] = useState(null);

  useEffect(() => {
    checkBackendHealth()
      .then(() => setBackendReady(true))
      .catch(() => setBackendReady(false));
  }, []);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const chooseVideo = (event) => {
    const selected = event.target.files?.[0];
    if (!selected) return;
    if (!selected.type.startsWith("video/") && !/\.(mp4|mov|avi|mkv)$/i.test(selected.name)) {
      setError("Choose a supported video file: MP4, MOV, AVI, or MKV.");
      return;
    }
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    setResult(null);
    setError("");
    event.target.value = "";
  };

  const analyze = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const uploaded = await uploadVideo(file);
      const analysis = await processVideo(uploaded.filename, { profile, annotate });
      setResult(analysis);
    } catch (requestError) {
      setError(getApiErrorMessage(requestError));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="analysis-layout">
      <section className="analysis-workspace">
        <div className="analysis-title">
          <div>
            <span className="eyebrow"><span className="status-dot" /> Video analysis</span>
            <h2>Review your footage</h2>
          </div>
          <span className={`panel-live${backendReady === false ? " panel-live--offline" : ""}`}>
            {busy ? "PROCESSING" : backendReady === false ? "BACKEND OFFLINE" : backendReady ? "BACKEND READY" : "CONNECTING"}
          </span>
        </div>

        {previewUrl ? (
          <video className="analysis-video" src={previewUrl} controls playsInline />
        ) : (
          <label className="analysis-drop">
            <input type="file" accept="video/*,.avi,.mkv" onChange={chooseVideo} />
            <Camera size={30} />
            <strong>Choose a crowd video</strong>
            <span>Browse your files to preview the footage here</span>
          </label>
        )}

        <div className="analysis-controls">
          <span className="filename">{file ? file.name : "No video selected"}</span>
          <button className="button button-light" onClick={analyze} disabled={!file || busy}>
            {busy ? <><LoaderCircle className="spin" size={15} /> Analyzing…</> : <>Analyze video <ArrowRight size={15} /></>}
          </button>
        </div>

        <AnalysisSettings
          profile={profile}
          onProfileChange={setProfile}
          annotate={annotate}
          onAnnotateChange={setAnnotate}
          disabled={busy}
        />
        {backendReady === false && <p className="analysis-error" role="status">Backend is offline. Start the FastAPI server, then reload this page.</p>}
        {error && <p className="analysis-error" role="alert">{error}</p>}
      </section>
      <AnalysisResults result={result} />
    </div>
  );
}
