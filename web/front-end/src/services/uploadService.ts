import { apiFetch } from "./api";
const API_URL = "http://127.0.0.1:5000"; // endpoint real depois
// Upload usando CSV
export async function uploadFileData(
  file: File,
  model: string | null,
  hyperParams: any
) {
  const formData = new FormData();
  formData.append("lightcurve", file);
  if (model) formData.append("model", model);
  formData.append("hyperParams", JSON.stringify(hyperParams));

  const response = await fetch(`${API_URL}/look-for-exoplanet`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) throw new Error("File upload failed");
  return response.json();
}

// Upload usando Star Name
export async function uploadStarData(
  starName: string,
  model: string | null,
  hyperParams: any
) {
  const response = await fetch(`${API_URL}/look-for-exoplanet`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ starName, model, hyperParams }),
  });

  if (!response.ok) throw new Error("Star upload failed");
  return response.json();
}
