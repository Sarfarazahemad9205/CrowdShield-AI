import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 120000,
});

export const uploadVideo = async (file) => {
  const formData = new FormData();

  formData.append("file", file);

  const response = await API.post(
    "/upload-video",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );

  return response.data;
};

export const processVideo = async (filename) => {
  const response = await API.post(
    `/process-video/${filename}`
  );

  return response.data;
};

export const checkBackendHealth = async () => {
  const response = await API.get("/health");

  return response.data;
};

export default API;