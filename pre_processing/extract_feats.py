import numpy as np
from scipy.stats import binned_statistic
from collections import OrderedDict
from scipy.stats import skew, kurtosis
from scipy.stats import binned_statistic

from downloadclean import download_and_save_lightcurve, download_and_clean
from detrend_and_period import detrend_with_bls_mask
from folded_binned_metrics import folded_binned_metrics
from cdpp import calculate_cdpp
from sesmes import compute_SES_MES
from utils import scaling_and_metrics, _interp_cdpp, _compute_secondary_depth
from per_trans_stat import per_transit_stats_simple

def extract_all_features(target, mission="Kepler", sigma_clip=5.0, verbose=True):

    _, lc = download_and_save_lightcurve(target=target, mission=mission, sigma_clip=sigma_clip, verbose=verbose)
    # lc = download_and_clean(target=target, mission=mission, sigma_clip=sigma_clip)
    time = lc.time.value
    flux = lc.flux.value

    feats = OrderedDict()

    mask0 = np.isfinite(time) & np.isfinite(flux)
    if np.sum(mask0) < 3:
        raise ValueError("too few valid points")
    time_arr = np.asarray(time)[mask0]
    flux_arr = np.asarray(flux)[mask0]

    flux_detr_full, trend_full, mask_transit, bls_info = detrend_with_bls_mask(time_arr, flux_arr)
    period = float(bls_info.get("best_period", np.nan))
    t0 = float(bls_info.get("t0", np.nan))
    duration_days = float(bls_info.get("best_duration", np.nan))

    feats["period_days"] = period
    feats["t0"] = t0
    feats["duration_days"] = duration_days
    feats["duration_hours"] = duration_days * 24.0

    flux_scaled, scaling_metrics = scaling_and_metrics(time_arr, flux_detr_full.copy())
    feats["scale_mean"] = scaling_metrics.get("mean", np.nan)
    feats["scale_std"] = scaling_metrics.get("std", np.nan)
    feats["scale_skewness"] = scaling_metrics.get("skewness", np.nan)
    feats["scale_kurtosis"] = scaling_metrics.get("kurtosis", np.nan)
    feats["scale_outlier_resistance"] = scaling_metrics.get("outlier_resistance", np.nan)

    binned = folded_binned_metrics(time_arr, flux_detr_full, period, t0,
                                   lags_hours=(1,3,6,12,24))
    feats["local_noise"] = binned.get("local_noise", np.nan)
    feats["depth_stability"] = binned.get("depth_stability", np.nan)
    acf_l = binned.get("acf_lags", {})
    feats["acf_lag_1h"]  = acf_l.get(1, np.nan)
    feats["acf_lag_3h"]  = acf_l.get(3, np.nan)
    feats["acf_lag_6h"]  = acf_l.get(6, np.nan)
    feats["acf_lag_12h"] = acf_l.get(12, np.nan)
    feats["acf_lag_24h"] = acf_l.get(24, np.nan)
    feats["cadence_hours"] = binned.get("cadence_hours", np.nan)

    per = per_transit_stats_simple(time_arr, flux_detr_full, period, t0, duration_days)
    depths = per.get("depths", np.array([]))
    npts_in_transit = per.get("npts_in_transit", np.array([]))
    feats["depth_mean_per_transit"] = float(np.nanmean(depths)) if depths.size else np.nan
    feats["depth_std_per_transit"]  = float(np.nanstd(depths)) if depths.size else np.nan
    feats["npts_transit_median"] = float(np.nanmedian(npts_in_transit)) if npts_in_transit.size else np.nan

    cdpp = calculate_cdpp(flux_detr_full, cadence_hours=feats["cadence_hours"])
    feats["cdpp_3h"]  = cdpp.get("cdpp_3h", np.nan)
    feats["cdpp_6h"]  = cdpp.get("cdpp_6h", np.nan)
    feats["cdpp_12h"] = cdpp.get("cdpp_12h", np.nan)

    duration_hours = feats["duration_hours"]
    cdpp_interp = _interp_cdpp(cdpp, duration_hours)
    sesmes = compute_SES_MES(depths, feats["local_noise"], npts_in_transit,
                             cdpp_dict=cdpp, duration_hours=duration_hours, method='auto')
    SES_arr = sesmes.get("SES", np.array([]))
    MES_val = sesmes.get("MES", np.nan)
    feats["SES_mean"] = float(np.nanmean(SES_arr)) if SES_arr.size else np.nan
    feats["SES_std"]  = float(np.nanstd(SES_arr)) if SES_arr.size else np.nan
    feats["MES"] = float(MES_val)

    depth_mean = feats["depth_mean_per_transit"]
    if np.isfinite(depth_mean) and np.isfinite(cdpp_interp) and cdpp_interp > 0:
        feats["snr_global"] = float((depth_mean * 1e6) / cdpp_interp)
    else:
        feats["snr_global"] = np.nan

    if SES_arr.size:
        feats["snr_per_transit_mean"] = float(np.nanmean(SES_arr))
        feats["snr_per_transit_std"]  = float(np.nanstd(SES_arr))
    else:
        feats["snr_per_transit_mean"] = np.nan
        feats["snr_per_transit_std"] = np.nan

    resid_global_full = flux_detr_full - np.nanmedian(flux_detr_full)
    feats["resid_rms_global"] = float(np.nanstd(resid_global_full)) if np.any(np.isfinite(resid_global_full)) else np.nan

    try:
        nbins = 200
        phase = ((time_arr - t0) / period) % 1.0
        phase = (phase + 0.5) % 1.0 - 0.5
        bins = np.linspace(-0.5, 0.5, nbins+1)
        med_profile, _, _ = binned_statistic(phase, flux_detr_full, statistic="median", bins=bins)
        bin_centers = 0.5*(bins[:-1] + bins[1:])
        dur_phase = (duration_days / period) if (period>0) else 0.05

        center_mask   = np.abs(bin_centers) < (0.25 * dur_phase)
        shoulder_mask = (np.abs(bin_centers) > (0.5 * dur_phase)) & (np.abs(bin_centers) < dur_phase)

        center_flux   = np.nanmedian(med_profile[center_mask])   if np.any(center_mask)   else np.nan
        shoulder_flux = np.nanmedian(med_profile[shoulder_mask]) if np.any(shoulder_mask) else np.nan

        depth_for_shape = abs(feats["depth_mean_per_transit"]) 
        num = (shoulder_flux - center_flux)

        if np.isfinite(center_flux) and np.isfinite(shoulder_flux) and np.isfinite(depth_for_shape) and depth_for_shape > 0:
            feats["vshape_metric"] = max(0.0, num / depth_for_shape)
        else:
            feats["vshape_metric"] = np.nan
    except Exception:
        feats["vshape_metric"] = np.nan

    feats["secondary_depth"] = _compute_secondary_depth(time_arr, flux_detr_full, period, t0, duration_days)

    finite_scaled = np.isfinite(flux_scaled)
    if np.sum(finite_scaled) > 2:
        feats["skewness_flux"] = float(skew(flux_scaled[finite_scaled]))
        feats["kurtosis_flux"] = float(kurtosis(flux_scaled[finite_scaled]))
    else:
        feats["skewness_flux"] = np.nan
        feats["kurtosis_flux"] = np.nan
    feats["outlier_resistance"] = float(np.sum(np.abs(flux_scaled[finite_scaled]) > 5) / np.sum(finite_scaled) * 100) if np.sum(finite_scaled)>0 else np.nan

    if "RADIUS" in lc.meta and np.isfinite(lc.meta["RADIUS"]):
        stellar_radius = lc.meta["RADIUS"]  
        Rp_over_Rs = np.sqrt(feats["depth_mean_per_transit"])
        feats["planet_radius_rearth"] = stellar_radius * 109.1 * Rp_over_Rs
        feats["planet_radius_rjup"]   = stellar_radius * 9.95 * Rp_over_Rs
    else:
        feats["planet_radius_rearth"] = np.nan
        feats["planet_radius_rjup"]   = np.nan

    if verbose:
        print("\n=== Extracted features (ordered) ===")
        for k, v in feats.items():
            print(f"{k}: {v}")

    return feats

