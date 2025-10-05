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

- `GET /health` - Health check
- `GET /` - API info

## Example

```bash
curl -X POST -F "label=HAT-P-7" http://localhost:5000/look-for-exoplanet
```

## Response

```json
{
  "features": { "period_days": 3.45, "MES": 12.3, ... },
  "prediction": { "exoplanet_confidence": 78.5, ... },
  "metadata": { "data_source": "mast_download", ... }
}
```
