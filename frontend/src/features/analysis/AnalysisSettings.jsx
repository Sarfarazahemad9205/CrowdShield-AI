const modes = [
  {
    id: "fast",
    label: "Fast",
    detail: "640 px · every other frame",
  },
  {
    id: "balanced",
    label: "Medium",
    detail: "640 px · every frame",
  },
  {
    id: "accurate",
    label: "Slow",
    detail: "1280 px · every frame",
  },
];

export default function AnalysisSettings({ profile, onProfileChange, annotate, onAnnotateChange, disabled }) {
  return (
    <section className="analysis-settings" aria-label="Analysis settings">
      <fieldset className="analysis-mode-group" disabled={disabled}>
        <legend>Analysis speed</legend>
        <div className="analysis-mode-options">
          {modes.map((mode) => (
            <label className={`analysis-mode${profile === mode.id ? " is-selected" : ""}`} key={mode.id}>
              <input
                type="radio"
                name="analysis-profile"
                value={mode.id}
                checked={profile === mode.id}
                onChange={() => onProfileChange(mode.id)}
              />
              <strong>{mode.label}</strong>
              <small>{mode.detail}</small>
            </label>
          ))}
        </div>
      </fieldset>
      <label className="analysis-annotate-option">
        <input
          type="checkbox"
          checked={annotate}
          disabled={disabled}
          onChange={(event) => onAnnotateChange(event.target.checked)}
        />
        <span>
          Export annotated video
          <small>Extra rendering and encoding time</small>
        </span>
      </label>
    </section>
  );
}
