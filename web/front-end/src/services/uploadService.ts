import { apiFetch } from "./api";

export const uploadData = async (file: File, model: string | null, hyperParams: any) => {
  const formData = new FormData();
  formData.append("file", file);

  if (model) {
    formData.append("model", model); // adiciona o modelo selecionado
  }

  if (hyperParams) {
    formData.append("hyperParams", JSON.stringify(hyperParams)); // envia como string JSON
  }

  return apiFetch("/predict", {
    method: "POST",
    body: formData,
  });
};

