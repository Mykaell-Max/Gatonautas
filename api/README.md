# Exoplanet Detection API

Simple API for detecting exoplanets from light curve data.

## Quick Start

```bash
make install
make run
```

## Endpoints

- `POST /look-for-exoplanet` - Detect exoplanets

  - Parameters: `label` (target name) or `lightcurve` (CSV file)
  - Returns: Features + prediction with confidence scores

- `GET /hyperparameters` - Get all hyperparameters
- `GET /hyperparameters/{model}` - Get model hyperparameters (rf, gb, lgbm)
- `POST /hyperparameters/{model}` - Update model hyperparameters
- `POST /hyperparameters/{model}/reset` - Reset to defaults

- `GET /` - API info

## Example

```bash
curl -X POST -F "label=HAT-P-7" http://localhost:5000/look-for-exoplanet
```

## Response

```json
{
  "features": { "period_days": 3.45, "MES": 12.3, ... },
  "prediction": { "exoplanet_confidence": 78.5 },
  "metadata": { "data_source": "mast_download" }
}
```
