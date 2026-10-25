import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import PageHeading from "../components/PageHeading";
const steps=[["01","Add a video","Choose a crowd video from your device to begin an analysis."],["02","Read the movement","The vision system tracks people and movement over time."],["03","Review the assessment","See a risk summary and supporting analysis alongside your video."]];
export default function HowItWorksPage(){return <div className="inner-page"><PageHeading eyebrow="How it works" title="From footage to" accent="forward thinking." description="A straightforward workflow turns video into a useful crowd movement assessment."/><div className="page-steps">{steps.map(([number,title,text])=><article className="process-step" key={number}><span>{number}</span><div><h2>{title}</h2><p>{text}</p></div></article>)}</div><Link className="button button-light" to="/video-analysis">Start an analysis <ArrowRight size={15}/></Link></div>}
