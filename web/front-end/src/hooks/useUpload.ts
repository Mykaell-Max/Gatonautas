import { useState } from "react";
import { uploadData } from "../services/uploadService";

export const useUpload = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const submitUpload = async (file: File, learningRate: number, epochs: number) => {
    try {
      setLoading(true);
      setError(null);
      const response = await uploadData(file, learningRate, epochs);
      setResult(response);
    } catch (err: any) {
      setError(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return { submitUpload, loading, error, result };
};
