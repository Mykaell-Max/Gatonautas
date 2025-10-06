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

  // novo estado para toggle
  const [inputMode, setInputMode] = useState<"csv" | "star">("csv");
  const [starName, setStarName] = useState<string>("");

  const handleModelChange = (model: keyof typeof modelsConfig) => {
    setSelectedModel(model);
    setHyperParams(modelsConfig[model]); // carrega defaults do modelo
  };

  const { submitUploadFile, submitUploadStar, loading, error, result } = useUpload();


  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

const handleSubmit = () => {
  if (inputMode === "csv" && file) {
    submitUploadFile(file, selectedModel, hyperParams);
  } else if (inputMode === "star" && starName.trim()) {
    submitUploadStar(starName, selectedModel, hyperParams);
  }
};


  return (
    <div className="upload-container">
      <h1>Discover Exoplanets with Your Light Curves Data</h1>
      <p>
        Upload your CSV files or enter a star name. Our ML model will analyze them.
        You can either stick with the default hyperparameters, carefully optimized
        by our developers, or adjust them yourself in Advanced Mode.
      </p>

      <select onChange={(e) => handleModelChange(e.target.value as keyof typeof modelsConfig)}>
        <option value=""> Select a model </option>
        {Object.keys(modelsConfig).map((model) => (
          <option key={model} value={model}>
            {model}
          </option>
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
          {/* Toggle entre CSV e Star Name */}
      <div className="input-mode-toggle">
        <button
          className={inputMode === "csv" ? "active" : ""}
          onClick={() => setInputMode("csv")}
        >
          Upload CSV
        </button>
        <button
          className={inputMode === "star" ? "active" : ""}
          onClick={() => setInputMode("star")}
        >
          Enter Star Name
        </button>
      </div>
      {/* Se inputMode = csv, mostra upload */}
      {inputMode === "csv" && (
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
      )}

      {/* Se inputMode = star, mostra input texto */}
      {inputMode === "star" && (
        <div className="star-input">
          <label>
            Star Name:
            <input
              type="text"
              value={starName}
              onChange={(e) => setStarName(e.target.value)}
              placeholder="Enter star name..."
            />
          </label>
        </div>
      )}

      <button className="submit-btn" onClick={handleSubmit} disabled={loading}>
        {loading ? "Processing..." : "Run Prediction"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
};

export default UploadDataView;
