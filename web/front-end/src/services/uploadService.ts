import { apiFetch } from "./api";

export const uploadData = async (file: File, learningRate: number, epochs: number) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("learning_rate", learningRate.toString());
  formData.append("epochs", epochs.toString());

  return apiFetch("/predict", {
    method: "POST",
    body: formData,
  });
};
