import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import SiteLayout from "./components/SiteLayout";
import "./App.css";

const HomePage = lazy(() => import("./pages/HomePage"));
const AboutPage = lazy(() => import("./pages/AboutPage"));
const BenefitsPage = lazy(() => import("./pages/BenefitsPage"));
const HowItWorksPage = lazy(() => import("./pages/HowItWorksPage"));
const FaqPage = lazy(() => import("./pages/FaqPage"));
const VideoAnalysisPage = lazy(() => import("./features/analysis/VideoAnalysisPage"));

export default function App() {
  return <BrowserRouter><Suspense fallback={<div className="route-loading">Loading page…</div>}><Routes><Route element={<SiteLayout />}>
    <Route path="/" element={<HomePage />} />
    <Route path="/about" element={<AboutPage />} />
    <Route path="/benefits" element={<BenefitsPage />} />
    <Route path="/how-it-works" element={<HowItWorksPage />} />
    <Route path="/faqs" element={<FaqPage />} />
    <Route path="/video-analysis" element={<VideoAnalysisPage />} />
    <Route path="*" element={<Navigate to="/" replace />} />
  </Route></Routes></Suspense></BrowserRouter>;
}
