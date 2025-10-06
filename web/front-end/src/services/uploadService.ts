export async function uploadFileData(file: File, model: string | null, hyperParams: any) {
  const formData = new FormData();
  formData.append("file", file);
  if (model) formData.append("model", model);
  formData.append("hyperParams", JSON.stringify(hyperParams));

  const response = await fetch("/api/upload-file", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) throw new Error("File upload failed");
  return response.json();
}

// Upload usando Star Name
export async function uploadStarData(starName: string, model: string | null, hyperParams: any) {
  const response = await fetch("/api/upload-star", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ starName, model, hyperParams }),
  });

  if (!response.ok) throw new Error("Star upload failed");
  return response.json();
}

