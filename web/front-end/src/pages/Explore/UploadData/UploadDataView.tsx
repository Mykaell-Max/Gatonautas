import { useState } from "react";
import type { ChangeEvent } from "react";
import "./UploadData.css";
import { useUpload } from "../../../hooks/useUpload";
import { modelsConfig } from "../../../config/modelsConfig";


const UploadDataView: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

const [selectedModel, setSelectedModel] = useState<keyof typeof modelsConfig | null>(null);
const [hyperParams, setHyperParams] = useState<any>({});

const handleModelChange = (model: keyof typeof modelsConfig) => {
  setSelectedModel(model);
  setHyperParams(modelsConfig[model]); // carrega defaults do modelo
};


  const { submitUpload, loading, error, result } = useUpload();

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

const handleSubmit = () => {
  if (!file) return;
  submitUpload(file, selectedModel, hyperParams);
};


  return (
    <div className="upload-container">
      <h1>Discover Exoplanets with Your Light Curves Data</h1>
      <p>
        Upload your CSV files and let our ML model analyze them. You can either
        stick with the default hyperparameters, carefully optimized by our
        developers for best results, or adjust them yourself in Advanced Mode.
      </p>
<select onChange={(e) => handleModelChange(e.target.value as keyof typeof modelsConfig)}>
  <option value=""> Select a model </option>
  {Object.keys(modelsConfig).map((model) => (
    <option key={model} value={model}>{model}</option>
  ))}
</select>
      <div style={{ display: "flex", gap: "1rem", marginTop: "1rem" }}>
        <button
          className="advanced-toggle"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? "Hide Advanced Mode" : "Advanced Mode"}
        </button>

      </div>

{showAdvanced && selectedModel && (
  <div className="advanced-settings">
    <h3>{selectedModel} Hyperparameters</h3>
{Object.entries(hyperParams).map(([param, value]) => {
  // Se o valor for um array de strings -> renderiza como dropdown
  if (Array.isArray(value) && value.every((v) => typeof v === "string")) {
    return (
      <label key={param}>
        {param}:
        <select
          value={hyperParams[param]}
          onChange={(e) =>
            setHyperParams({
              ...hyperParams,
              [param]: e.target.value,
            })
          }
        >
          {value.map((option: string) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
    );
  }

  // Caso contrário: number ou texto
  return (
    <label key={param}>
      {param}:
      <input
        type={typeof value === "number" ? "number" : "text"}
        value={value as any}
        onChange={(e) =>
          setHyperParams({
            ...hyperParams,
            [param]:
              typeof value === "number"
                ? Number(e.target.value)
                : e.target.value,
          })
        }
      />
    </label>
  );
})}
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
