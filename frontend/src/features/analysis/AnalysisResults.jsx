import { Activity, AlertTriangle, CheckCircle2, CircleHelp, XCircle } from "lucide-react";
import API from "../../services/api";

const states = {
  NORMAL: { label: "Normal", Icon: CheckCircle2 },
  HIGH_RISK: { label: "High risk", Icon: AlertTriangle },
  CRITICAL: { label: "Critical", Icon: XCircle },
  TRACKING_UNRELIABLE: { label: "Tracking uncertain", Icon: CircleHelp },
};

export default function AnalysisResults({ result }) {
  // The API contract nests computed fields under `risk`.
  const risk = result?.risk ?? null;
  const state = states[risk?.risk_state] || { label: "Awaiting analysis", Icon: Activity };
  const { Icon } = state;
  const reliability = risk?.tracking_reliability;

  return (
    <aside className="analysis-results" aria-live="polite">
      <span className="eyebrow">Analysis results</span>
      {risk ? (
        <>
          <div className="risk-level">
            <Icon size={27} />
            <div><small>Current risk level</small><strong>{state.label}</strong></div>
          </div>
          <div className="result-detail">
            <span>Risk explanation</span>
            <p>{risk.risk_reason || "No additional explanation was returned."}</p>
          </div>
          <div className="result-metrics">
            {[
              ["Tracking reliability", reliability != null ? `${(reliability * 100).toFixed(1)}%` : "—"],
              ["Windows analyzed", risk.total_windows ?? "—"],
              ["Movement intensity", risk.movement_intensity_index != null ? Number(risk.movement_intensity_index).toFixed(3) : "—"],
              ["Tracking quality", risk.tracking_quality?.replaceAll("_", " ") || "—"],
            ].map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}
          </div>
          {risk.windows?.length > 0 && (
            <div className="result-detail result-windows">
              <span>Time-window results</span>
              {risk.windows.map((window) => (
                <p key={window.window_id}>
                  {Number(window.window_start).toFixed(0)}–{Number(window.window_end).toFixed(0)} sec · {window.risk_state.replaceAll("_", " ")}
                </p>
              ))}
            </div>
          )}
          {result.video_url && (
            <a
              className="button button-glass result-video-download"
              href={new URL(result.video_url, API.defaults.baseURL).toString()}
              download={result.filename || "annotated-video.mp4"}
            >
              Download annotated video
            </a>
          )}
        </>
      ) : (
        <div className="empty-results">
          <Activity size={23} />
          <strong>Your results will appear here</strong>
          <p>Choose a video and run an analysis to see the risk assessment and movement indicators.</p>
        </div>
      )}
    </aside>
  );
}
