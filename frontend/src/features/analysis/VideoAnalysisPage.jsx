import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import VideoUploader from "./VideoUploader";
export default function VideoAnalysisPage(){return <div className="analysis-page"><Link className="back-link" to="/"><ArrowLeft size={15}/> Back to home</Link><header className="analysis-page-heading"><span className="eyebrow">StampedeGuard workspace</span><h1>Video <em>analysis</em></h1><p>Upload and review crowd footage with a clear assessment beside the video.</p></header><VideoUploader/></div>}
