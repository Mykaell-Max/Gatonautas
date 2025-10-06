import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import "./UploadData.css";
import { useUpload } from "../../../hooks/useUpload";
import { modelsConfig } from "../../../config/modelsConfig";

const mockResult = {
  prediction: {
    prediction_label: "CONFIRMED",
    exoplanet_confidence: 0.95,
    non_exoplanet_confidence: 0.05,
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
    vshape_metric: 0.941,
    secondary_depth: 0.000818,
    skewness_flux: -3.4564,
    kurtosis_flux: 10.3765,
    outlier_resistance: 0.0,
    planet_radius_rearth: 17.8196,
    planet_radius_rjup: 1.6251,
  },
};

  const USE_MOCK = true;

const formatPercent = (value?: number) => {
  if (typeof value !== "number") return "—";
  return `${(value * 100).toFixed(1)}%`;
};

const formatFeatureValue = (value: unknown) => {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number") {
    if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
      return value.toExponential(2);
    }
    return value.toFixed(4);
  }
  if (value == null) {
    return "—";
  }
  return String(value);
};

const UploadDataView: React.FC = () => {
  const modelOptions = Object.keys(modelsConfig) as (keyof typeof modelsConfig)[];
  const defaultModel = modelOptions[0] ?? null;

  const [file, setFile] = useState<File | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [localResult, setLocalResult] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState<keyof typeof modelsConfig | null>(defaultModel);
  const [hyperParams, setHyperParams] = useState<any>(
    defaultModel ? modelsConfig[defaultModel] : {}
  );
  const [fileError, setFileError] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const { submitUploadFile, loading, error, result } = useUpload();

  useEffect(() => {
    if (selectedModel) {
      setHyperParams(modelsConfig[selectedModel]);
    }
  }, [selectedModel]);

  useEffect(() => {
    if (!USE_MOCK && result) {
      setLocalResult(result);
    }
  }, [result]);

  const isLoading = USE_MOCK ? isProcessing : loading;
  const activeError = USE_MOCK ? null : error;
  const activeResult = USE_MOCK ? localResult : result ?? localResult;

  const handleModelChange = (model: keyof typeof modelsConfig) => {
    setSelectedModel(model);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setFileError(false);
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      setFile(event.dataTransfer.files[0]);
      setFileError(false);
      event.dataTransfer.clearData();
    }
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleSubmit = async () => {
    if (!file) {
      setFileError(true);
      return;
    }

    setFileError(false);
    setLocalResult(null);

    if (USE_MOCK) {
      setIsProcessing(true);
      setTimeout(() => {
        setLocalResult(mockResult);
        setIsProcessing(false);
      }, 400);
      return;
    }

    try {
      setIsProcessing(true);
      await submitUploadFile(file, selectedModel, hyperParams);
    } catch (requestError) {
      console.warn("Upload request failed", requestError);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setLocalResult(null);
    setFileError(false);
    setIsProcessing(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const featureEntries = activeResult?.features
    ? Object.entries(activeResult.features)
    : [];

  const predictedLabel = activeResult?.prediction?.prediction_label ?? "Awaiting run";
  const exoplanetConfidence = activeResult?.prediction?.exoplanet_confidence;
  const nonExoplanetConfidence = activeResult?.prediction?.non_exoplanet_confidence;
  const featuresCount = activeResult?.features_extracted ?? featureEntries.length;

  return (
    <div className="upload-page">
      <section className="upload-hero">
        <span className="badge">Light Curve Lab</span>
        <h1>Upload observations &amp; retune models</h1>
        <p>
          Bring your CSV light curve files, experiment with different models and
          hyperparameters, and inspect the extracted features alongside the
          prediction results. Everything below is running with mock data so you
          can iterate on the interface before connecting the real backend.
        </p>
      </section>

      <div className="upload-content">
        <section className="configuration-panel">
          <header className="panel-header">
            <h2>Configuration</h2>
            <p>Select a model, tweak the hyperparameters and upload your file.</p>
          </header>

          <div className="config-group">
            <label htmlFor="model-select">Choose a model</label>
            <select
              id="model-select"
              value={selectedModel ?? ""}
              onChange={(e) => handleModelChange(e.target.value as keyof typeof modelsConfig)}
            >
              <option value="" disabled>
                Select a model
              </option>
              {modelOptions.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>

          <button
            className="advanced-toggle"
            type="button"
            onClick={() => setShowAdvanced((prev) => !prev)}
            disabled={!selectedModel}
          >
            {showAdvanced ? "Hide advanced hyperparameters" : "Show advanced hyperparameters"}
          </button>

          {showAdvanced && selectedModel && (
            <div className="advanced-settings">
              <div className="advanced-header">
                <h3>{selectedModel} hyperparameters</h3>
                <span>Experiment with the search space configured for the model.</span>
              </div>
              <div className="advanced-grid">
                {Object.entries(hyperParams).map(([param, value]) => {
                  if (Array.isArray(value) && value.every((v) => typeof v === "string")) {
                    return (
                      <label key={param} className="advanced-field">
                        <span>{param}</span>
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
                    <label key={param} className="advanced-field">
                      <span>{param}</span>
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
            </div>
          )}

          <div
            className={`upload-dropzone ${file ? "has-file" : ""} ${fileError ? "has-error" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            role="button"
            tabIndex={0}
            aria-label="Upload CSV file"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              id="lightcurve-upload"
              onChange={handleFileChange}
              style={{ display: "none" }}
            />
            {file ? (
              <div className="dropzone-summary">
                <p className="file-name">{file.name}</p>
                <span>{(file.size / 1024).toFixed(1)} KB</span>
              </div>
            ) : (
              <div className="dropzone-placeholder">
                <p>Drag &amp; drop your CSV here or click to browse</p>
                <span>We currently support the same schema used by the mock pipeline.</span>
              </div>
            )}
          </div>
          {fileError && <p className="input-hint">Select a CSV file before running the prediction.</p>}

          <div className="action-row">
            <button
              className="primary-btn"
              onClick={handleSubmit}
              disabled={isLoading}
            >
              {isLoading ? "Processing..." : "Run prediction"}
            </button>
            {file && (
              <button className="ghost-btn" onClick={handleReset} type="button">
                Clear selection
              </button>
            )}
          </div>

          {activeError && <p className="error-banner">{activeError}</p>}
        </section>

        <aside className="helper-panel">
          <div className="helper-card">
            <h3>What happens after upload?</h3>
            <ul>
              <li>We run the light-curve pre-processing pipeline (mocked here).</li>
              <li>Feature extraction generates transit statistics and noise metrics.</li>
              <li>The chosen model produces classification confidences for exoplanet vs non-exoplanet.</li>
            </ul>
          </div>

          <div className="helper-card">
            <h3>Tips for exploration</h3>
            <ul>
              <li>Switch models to compare how the default hyperparameters behave.</li>
              <li>Toggle advanced mode to stress different search spaces.</li>
              <li>Inspect the feature chips below to build intuition on your dataset.</li>
            </ul>
          </div>
        </aside>
      </div>

      {activeResult && (
        <section className="results-section">
          <div className="results-header">
            <div>
              <h2>Prediction overview</h2>
              <p>
                Showing the latest run for <strong>{selectedModel ?? "—"}</strong>. The
                numbers below are mock outputs so the UI can be validated before
                the API is wired.
              </p>
            </div>
            <div className="features-chip">{featuresCount} features extracted</div>
          </div>

          <div className="results-grid">
            <article className="prediction-card">
              <span className="prediction-label">Prediction</span>
              <h3 className={`prediction-value prediction-${predictedLabel.toLowerCase()}`}>
                {predictedLabel}
              </h3>
              <p className="prediction-footnote">
                Combined decision based on the selected model and the extracted feature vector.
              </p>
            </article>

            <article className="confidence-card">
              <header>
                <h4>Exoplanet confidence</h4>
                <span>{formatPercent(exoplanetConfidence)}</span>
              </header>
              <div className="confidence-bar">
                <span
                  className="confidence-bar-fill"
                  style={{ width: `${((exoplanetConfidence ?? 0) * 100).toFixed(1)}%` }}
                />
              </div>
            </article>

            <article className="confidence-card">
              <header>
                <h4>Non-exoplanet confidence</h4>
                <span>{formatPercent(nonExoplanetConfidence)}</span>
              </header>
              <div className="confidence-bar">
                <span
                  className="confidence-bar-fill alternative"
                  style={{ width: `${((nonExoplanetConfidence ?? 0) * 100).toFixed(1)}%` }}
                />
              </div>
            </article>
          </div>

          <div className="features-panel">
            <header>
              <h3>Extracted feature set</h3>
              <p>All metrics below are generated from the mocked pipeline.</p>
            </header>
            <div className="features-grid">
              {featureEntries.map(([name, value]) => (
                <div className="feature-pill" key={name}>
                  <span className="feature-name">{name}</span>
                  <span className="feature-value">{formatFeatureValue(value)}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}
    </div>
  );
};

export default UploadDataView;
