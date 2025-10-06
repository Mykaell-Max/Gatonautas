import { useState } from "react";
import type { ChangeEvent } from "react";
import "./UploadData.css";
import { useUpload } from "../../../hooks/useUpload";
import { modelsConfig } from "../../../config/modelsConfig";

const mockResult = {
    prediction: {
    prediction_label: "CONFIRMED",
    exoplanet_confidence: 0.95,
    non_exoplanet_confidence: 0.05
  },
    features_extracted: 30,
  features: {
    period_days: 2.2047,
    t0: 231.5954,
    duration_days: 0.142,
    duration_hours: 3.408,
    scale_mean: -0.2586,
    scale_std: 1.0,
    scale_skewness: -3.4564,
    scale_kurtosis: 10.3765,
    scale_outlier_resistance: 0.0,
    local_noise: 0.000164,
    depth_stability: 0.01053,
    acf_lag_1h: 0.7314,
    acf_lag_3h: 0.04745,
    acf_lag_6h: -0.06877,
    acf_lag_12h: -0.0609,
    acf_lag_24h: -0.06381,
    cadence_hours: 0.01634,
    depth_mean_per_transit: 0.007,
    depth_std_per_transit: 0.0000787,
    npts_transit_median: 833,
    cdpp_3h: 511.6117,
    cdpp_6h: 901.1231,
    cdpp_12h: 1305.7418,
    SES_mean: 12.401,
    SES_std: 0.1395,
    MES: 42.961,
    snr_global: 12.401,
    snr_per_transit_mean: 12.401,
    snr_per_transit_std: 0.1395,
    resid_rms_global: 0.00153,
    vshape_metric: 0.9410,
    secondary_depth: 0.000818,
    skewness_flux: -3.4564,
    kurtosis_flux: 10.3765,
    outlier_resistance: 0.0,
    planet_radius_rearth: 17.8196,
    planet_radius_rjup: 1.6251
  },
};

const UploadDataView: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [resultmock, setResult] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState<keyof typeof modelsConfig | null>(null);
  const [hyperParams, setHyperParams] = useState<any>({});

  const handleModelChange = (model: keyof typeof modelsConfig) => {
    setSelectedModel(model);
    setHyperParams(modelsConfig[model]); // carrega defaults do modelo
  };

  const { loading, error } = useUpload();


  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

const handleSubmit = () => {
  if (!file) return;
  // Mock temporário:
  console.log("Using mock result");
  setResult(mockResult);

  // Depois, descomente para usar a API real:
  // submitUploadFile(file, selectedModel, hyperParams);
};


  return (
    <div className="upload-container">
      <h1>Discover Exoplanets with Your Light Curves Data</h1>
      <p>
        Upload your CSV file and let our ML model analyze it. You can
        either stick with the default hyperparameters, carefully optimized by our
        developers, or adjust them yourself in Advanced Mode.
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
      {resultmock && <pre>{JSON.stringify(resultmock, null, 2)}</pre>}
    </div>
  );
};

export default UploadDataView;
