import axios from "axios";

const defaultBaseUrl = `${window.location.protocol}//${window.location.hostname}:8000`;
const API = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || defaultBaseUrl,
  // Full video analysis is synchronous until the job-queue API is introduced.
  timeout: 15 * 60 * 1000,
});

export const uploadVideo = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  // Let the browser/Axios set the multipart boundary.
  const { data } = await API.post("/upload-video", formData);
  return data;
};

export const processVideo = async (filename, options = {}) => {
  const { data } = await API.post(
    `/process-video/${encodeURIComponent(filename)}`,
    null,
    {
      params: {
        profile: options.profile || "balanced",
        annotate: Boolean(options.annotate),
      },
    },
  );
  return data;
};

export const checkBackendHealth = async () => {
  const { data } = await API.get("/health", { timeout: 10000 });
  return data;
};

export const getApiErrorMessage = (error) => {
  const detail = error.response?.data?.detail;
  if (detail) {
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  if (error.code === "ECONNABORTED") {
    return "Analysis is taking longer than 15 minutes. Try Fast mode or a shorter clip.";
  }
  if (!error.response) {
    return `Cannot reach the backend at ${API.defaults.baseURL}. Check that the backend is running.`;
  }
  return "The video could not be analyzed. Please try again.";
};

export default API;
