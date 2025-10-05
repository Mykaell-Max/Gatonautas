import numpy as np
from scipy import stats

# 7 - Utils

def scaling_and_metrics(time, flux_detr):
    flux = np.asarray(flux_detr).copy()
    flux[np.isnan(flux)] = np.nanmedian(flux)
    median = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - median))
    std = np.nanstd(flux)
    if mad < 0.001:
        flux_scaled = (flux - median) / std if std > 0 else flux - median
    else:
        flux_scaled = (flux - median) / (1.4826 * mad)
    finite = np.isfinite(flux_scaled)
    scaling_metrics = {
        'mean': np.nanmean(flux_scaled),
        'std': np.nanstd(flux_scaled),
        'skewness': stats.skew(flux_scaled[finite]) if np.sum(finite) > 2 else np.nan,
        'kurtosis': stats.kurtosis(flux_scaled[finite]) if np.sum(finite) > 2 else np.nan,
        'outlier_resistance': np.sum(np.abs(flux_scaled[finite]) > 5) / np.sum(finite) * 100
    }
    return flux_scaled, scaling_metrics

def calculate_detection_rate(flux_scaled, n_sigma=3):
    median = np.nanmedian(flux_scaled)
    mad = np.nanmedian(np.abs(flux_scaled - median))
    threshold = median - n_sigma * 1.4826 * mad
    finite = np.isfinite(flux_scaled)
    detection_rate = np.sum(flux_scaled[finite] < threshold) / np.sum(finite) * 100
    return {"adaptive_threshold": float(threshold), "detection_rate": float(detection_rate)}

def _compute_secondary_depth(time, flux_detr, period, t0, dur_days):
    mask = np.isfinite(time) & np.isfinite(flux_detr)
    if np.sum(mask) < 3:
        return np.nan
    t = np.asarray(time)[mask]; f = np.asarray(flux_detr)[mask]
    phase = ((t - t0)/period) % 1.0
    phase = (phase + 0.5) % 1.0 - 0.5
    phase_0to1 = ((t - t0)/period) % 1.0
    sec_center = 0.5
    sec_half = 1.5 * (dur_days / period) if (period>0) else 0.05
    sel = (phase_0to1 > (sec_center - sec_half)) & (phase_0to1 < (sec_center + sec_half))
    if not np.any(sel):
        return np.nan
    baseline = np.nanmedian(f)
    sec_min = np.nanmin(f[sel])
    return float(baseline - sec_min)

def _interp_cdpp(cdpp_dict, duration_hours):
    if cdpp_dict is None or duration_hours is None or not np.isfinite(duration_hours):
        return np.nan
    c3 = cdpp_dict.get('cdpp_3h', np.nan)
    c6 = cdpp_dict.get('cdpp_6h', np.nan)
    c12 = cdpp_dict.get('cdpp_12h', np.nan)
    vals = np.array([c for c in (c3,c6,c12) if np.isfinite(c)])
    if vals.size == 0:
        return np.nan
    if not np.isfinite(c3): c3 = np.nanmean(vals)
    if not np.isfinite(c6): c6 = np.nanmean(vals)
    if not np.isfinite(c12): c12 = np.nanmean(vals)
    h = duration_hours
    if h <= 3:
        return float(c3)
    elif h <= 6:
        return float(c3 + (c6-c3)*(h-3)/(6-3))
    elif h <= 12:
        return float(c6 + (c12-c6)*(h-6)/(12-6))
    else:
        return float(c12)
