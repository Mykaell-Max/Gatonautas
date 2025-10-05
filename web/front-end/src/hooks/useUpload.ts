import { useState } from "react";
import { uploadData } from "../services/uploadService";

export const useUpload = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

const submitUpload = async (file: File, model: string | null, hyperParams: any) => {
  try {
    setLoading(true);
    setError(null);

    const response = await uploadData(file, model, hyperParams); // PASSANDO TUDO
    setResult(response);
  } catch (err: any) {
    setError(err.message || "Upload failed");
  } finally {
    setLoading(false);
  }
};


  return { submitUpload, loading, error, result };
};
