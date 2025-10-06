import { useState } from "react";
import { uploadFileData, uploadStarData } from "../services/uploadService";

export const useUpload = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  // Upload via CSV File
  const submitUploadFile = async (
    file: File,
    model: string | null,
    hyperParams: any
  ) => {
    try {
      setLoading(true);
      setError(null);

      const response = await uploadFileData(file, model, hyperParams);
      setResult(response);
    } catch (err: any) {
      setError(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  // Upload via Star Name
  const submitUploadStar = async (
    starName: string,
    model: string | null,
    hyperParams: any
  ) => {
    try {
      setLoading(true);
      setError(null);

      const response = await uploadStarData(starName, model, hyperParams);
      setResult(response);
    } catch (err: any) {
      setError(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return { submitUploadFile, submitUploadStar, loading, error, result };
};
