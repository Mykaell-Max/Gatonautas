import { useState } from "react";
import type { ChangeEvent } from "react";
import "./UploadData.css";
import { useUpload } from "../../../hooks/useUpload";

const UploadDataView: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [learningRate, setLearningRate] = useState(0.01);
  const [epochs, setEpochs] = useState(10);

  const defaultParams = { learningRate: 0.01, epochs: 10 };
  const { submitUpload, loading, error, result } = useUpload();

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleReset = () => {
    setLearningRate(defaultParams.learningRate);
    setEpochs(defaultParams.epochs);
  };

  const handleSubmit = () => {
  if (!file) return;
  submitUpload(file, learningRate, epochs);
};

  return (
    <div className="upload-container">
      <h1>Discover Exoplanets with Your Light Curves Data</h1>
      <p>
        Upload your CSV files and let our ML model analyze them. You can either
        stick with the default hyperparameters, carefully optimized by our
        developers for best results, or adjust them yourself in Advanced Mode.
      </p>

      <button
        className="advanced-toggle"
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        {showAdvanced ? "Hide Advanced Mode" : "Advanced Mode"}
      </button>

      {showAdvanced && (
        <div className="advanced-settings">
          <h3>Adjust Hyperparameters</h3>

          <label>
            Learning Rate:
            <input
              type="number"
              step="0.001"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
            />
          </label>

          <label>
            Epochs:
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
            />
          </label>

          <button className="reset-btn" onClick={handleReset}>
            Reset to Default
          </button>
        </div>
      )}

      <div
        className="upload-dropzone"
        onClick={() => document.getElementById("fileInput")?.click()}
      >
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          id="fileInput"
          style={{ display: "none" }}
        />
        {file ? (
          <p className="file-name">{file.name}</p>
        ) : (
          <p>Click to select your CSV here</p>
        )}
      </div>

      <button className="submit-btn" onClick={handleSubmit} disabled={loading}>
  {loading ? "Processing..." : "Run Prediction"}
</button>

{error && <p style={{ color: "red" }}>{error}</p>}
{result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
};

export default UploadDataView;
