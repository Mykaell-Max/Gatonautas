# Imports

import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from astropy.timeseries import BoxLeastSquares 
from scipy import stats
from scipy.ndimage import uniform_filter1d
import warnings
from scipy.stats import binned_statistic
from statsmodels.tsa.stattools import acf as sm_acf
from collections import OrderedDict
from scipy.stats import skew, kurtosis
from scipy.stats import binned_statistic
from concurrent.futures import ProcessPoolExecutor, as_completed

# 1 - Dowload and clean light curve

def download_and_clean(target, mission="Kepler", sigma_clip=5.0):
    lc = lk.search_lightcurve(target, mission=mission).download()
    lc = lc.remove_nans()
    lc = lc.normalize()
    lc = lc.remove_outliers(sigma=sigma_clip)
    return lc

# 2 -  Detrend light curve with BLS and mask

def detrend_with_bls_mask(time, flux,
                          min_period=0.5, max_period=None,
                          n_periods=2000, n_durations=50,
                          oversample=10,
                          bin_width=0.5, spline_s=0.001,
                          max_iter=5, sigma=3.0,
                          refine_duration=True):
    mask_valid = np.isfinite(time) & np.isfinite(flux)
    time = np.asarray(time)[mask_valid]
    flux = np.asarray(flux)[mask_valid]

    if max_period is None:
        max_period = (time.max() - time.min()) / 3.0

    # Build a cadence-aware duration grid. Floor the minimum duration to ~3 samples.
    diffs = np.diff(time)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size:
        dt_days = np.median(diffs)
        min_dur_floor = max(0.01, 3.0 * dt_days)
    else:
        dt_days = 0.02 
        min_dur_floor = 0.06
    # Cap durations to a fraction of the minimum period so BLS constraints are satisfied
    max_dur_cap = 0.2 * min_period 
    if not np.isfinite(max_dur_cap) or max_dur_cap <= 0:
        max_dur_cap = 0.2
    if min_dur_floor >= max_dur_cap:
        min_dur_floor = 0.5 * max_dur_cap
    durations = np.linspace(min_dur_floor, max_dur_cap, n_durations)
    period_grid = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)

    bls = BoxLeastSquares(time, flux)
    periodogram = bls.power(period_grid, durations, oversample=oversample)

    if np.ndim(periodogram.power) == 1:
        idx_best = np.nanargmax(periodogram.power)
        best_period = periodogram.period[idx_best]
        best_duration = periodogram.duration[idx_best] if hasattr(periodogram, "duration") else durations[0]
        t0 = periodogram.transit_time[idx_best] if hasattr(periodogram, "transit_time") else time[0]
    else:
        power_per_period = np.nanmax(periodogram.power, axis=1)
        idx_best = int(np.nanargmax(power_per_period))
        best_period = periodogram.period[idx_best]
        dur_idx = int(np.nanargmax(periodogram.power[idx_best, :]))
        best_duration = periodogram.duration[dur_idx] if hasattr(periodogram, "duration") else durations[dur_idx]
        t0 = periodogram.transit_time[idx_best] if hasattr(periodogram, "transit_time") else time[0]

    if diffs.size:
        dt_days_best = np.median(diffs)
        best_floor = max(0.01, 3.0 * dt_days_best)
        best_cap = 0.2 * best_period
        if np.isfinite(best_duration):
            if best_duration < best_floor:
                best_duration = best_floor
            if np.isfinite(best_cap) and best_duration >= best_cap:
                best_duration = max(best_floor, 0.9 * best_cap)
    
    # Optional two-pass refinement for more accurate duration
    if refine_duration and np.isfinite(best_period) and np.isfinite(best_duration):
        # Second pass: focused search around detected period with proper duration range
        per_lo = best_period * 0.98
        per_hi = best_period * 1.02
        dur_lo = best_floor
        dur_hi = min(0.5, 0.2 * best_period)  # Allow up to 0.5 days or 20% of period
        
        if dur_hi > dur_lo:
            durations_refined = np.linspace(dur_lo, dur_hi, n_durations)
            periods_refined = np.logspace(np.log10(per_lo), np.log10(per_hi), max(256, n_periods//4))
            
            periodogram_refined = bls.power(periods_refined, durations_refined, oversample=oversample)
            
            if np.ndim(periodogram_refined.power) == 1:
                idx_best_refined = np.nanargmax(periodogram_refined.power)
                best_period_refined = periodogram_refined.period[idx_best_refined]
                best_duration_refined = periodogram_refined.duration[idx_best_refined] if hasattr(periodogram_refined, "duration") else durations_refined[0]
                t0_refined = periodogram_refined.transit_time[idx_best_refined] if hasattr(periodogram_refined, "transit_time") else t0
            else:
                power_per_period_refined = np.nanmax(periodogram_refined.power, axis=1)
                idx_best_refined = int(np.nanargmax(power_per_period_refined))
                best_period_refined = periodogram_refined.period[idx_best_refined]
                dur_idx_refined = int(np.nanargmax(periodogram_refined.power[idx_best_refined, :]))
                best_duration_refined = periodogram_refined.duration[dur_idx_refined] if hasattr(periodogram_refined, "duration") else durations_refined[dur_idx_refined]
                t0_refined = periodogram_refined.transit_time[idx_best_refined] if hasattr(periodogram_refined, "transit_time") else t0
            
            # Use refined values if they're reasonable
            if (np.isfinite(best_period_refined) and np.isfinite(best_duration_refined) and 
                best_period_refined > 0 and best_duration_refined > 0):
                best_period = best_period_refined
                best_duration = best_duration_refined
                t0 = t0_refined
                periodogram = periodogram_refined  # Update for consistency

                # Optional micro-zoom on duration at fixed period to reduce grid quantization
                try:
                    dur_half_span = max(0.02 * best_duration, best_floor * 0.5)
                    d_lo = max(best_floor, best_duration - dur_half_span)
                    d_hi = min(dur_hi, best_duration + dur_half_span)
                    if np.isfinite(d_lo) and np.isfinite(d_hi) and d_hi > d_lo:
                        durations_zoom = np.linspace(d_lo, d_hi, max(64, n_durations))
                        periodogram_zoom = bls.power(np.array([best_period]), durations_zoom, oversample=oversample)
                        # power has shape (len(durations),) when period array len == 1
                        idx_zoom = int(np.nanargmax(periodogram_zoom.power))
                        dur_zoom = periodogram_zoom.duration[idx_zoom] if hasattr(periodogram_zoom, "duration") else durations_zoom[idx_zoom]
                        if np.isfinite(dur_zoom) and dur_zoom > 0:
                            best_duration = float(dur_zoom)
                except Exception:
                    pass

                # Trapezoid refit at fixed period to refine T14 (duration)
                try:
                    t14_lo = max(best_floor, 0.5 * best_duration)
                    t14_hi = min(0.25 * best_period, 1.8 * best_duration, 0.6)
                    if np.isfinite(t14_lo) and np.isfinite(t14_hi) and t14_hi > t14_lo:
                        # Fold to phase time (days) around 0
                        phase = ((time - t0) / best_period + 0.5) % 1.0 - 0.5
                        x_days = phase * best_period
                        # Limit to neighborhood to improve robustness
                        win = max(2.0 * best_duration, 4.0 * best_floor)
                        sel = np.isfinite(x_days) & np.isfinite(flux) & (np.abs(x_days) <= win)
                        if np.sum(sel) >= 50:
                            xb = x_days[sel]
                            yb = flux[sel]
                            # Median bin to reduce noise
                            nb = min(200, max(60, int(np.sqrt(xb.size))))
                            bins = np.linspace(-win, win, nb + 1)
                            ymed, _, _ = binned_statistic(xb, yb, statistic="median", bins=bins)
                            xc = 0.5 * (bins[:-1] + bins[1:])
                            mask_med = np.isfinite(ymed) & np.isfinite(xc)
                            xc = xc[mask_med]
                            ymed = ymed[mask_med]

                            def _shape_trap(x, T14, T12):
                                half = 0.5 * T14
                                T12 = max(1e-6, min(T12, half))
                                flat_half = max(0.0, half - T12)
                                s = np.zeros_like(x)
                                # ingress
                                m = (x >= -half) & (x < -flat_half)
                                s[m] = (x[m] + half) / T12
                                # flat
                                m = (x >= -flat_half) & (x <= flat_half)
                                s[m] = 1.0
                                # egress
                                m = (x > flat_half) & (x <= half)
                                s[m] = (half - x[m]) / T12
                                # Outside remains 0
                                s = np.clip(s, 0.0, 1.0)
                                return s

                            best_loss = np.inf
                            best_t14 = best_duration
                            # Explore T14 and T12/T14 ratios (grazing→triangular up to 0.5)
                            t14_grid = np.linspace(t14_lo, t14_hi, 48)
                            ratio_grid = np.linspace(0.1, 0.45, 8)
                            one = np.ones_like(ymed)
                            for T14 in t14_grid:
                                for r in ratio_grid:
                                    T12 = r * T14
                                    s = _shape_trap(xc, T14, T12)
                                    A = np.column_stack([one, -s])
                                    try:
                                        coef, _, _, _ = np.linalg.lstsq(A, ymed, rcond=None)
                                        b0, d0 = float(coef[0]), float(coef[1])
                                        yhat = b0 - d0 * s
                                        resid = ymed - yhat
                                        loss = float(np.nanmean(resid * resid))
                                        if np.isfinite(loss) and loss < best_loss and d0 > 0:
                                            best_loss = loss
                                            best_t14 = float(T14)
                                    except Exception:
                                        continue
                            if np.isfinite(best_t14) and best_t14 > 0:
                                best_duration = best_t14
                except Exception:
                    pass
    
    mask_transit = bls.transit_mask(time, best_period, best_duration, t0)

    flux_work = flux.copy()
    mask_use = ~mask_transit

    for _ in range(max_iter):
        bins = np.arange(time.min(), time.max() + bin_width, bin_width)
        digitized = np.digitize(time, bins)
        bin_means = []
        bin_times = []
        for i in range(1, len(bins)):
            sel = (digitized == i) & mask_use
            if np.any(sel):
                bin_means.append(np.median(flux_work[sel]))
                bin_times.append(np.median(time[sel]))
        bin_means = np.array(bin_means)
        bin_times = np.array(bin_times)

        good = np.isfinite(bin_means) & np.isfinite(bin_times)
        if good.sum() < 5:
            break
        spline = UnivariateSpline(bin_times[good], bin_means[good], s=spline_s)
        trend = spline(time)
        resid = flux_work - trend

        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        thresh = -sigma * 1.4826 * mad
        mask_use = mask_use & (resid > thresh)

    if np.sum(mask_use) < 3:
        spline = UnivariateSpline(time, flux, s=spline_s)
    else:
        spline = UnivariateSpline(time[mask_use], flux[mask_use], s=spline_s)
    trend = spline(time)
    flux_detr = flux / trend

    bls_info = {
        "best_period": float(best_period),
        "best_duration": float(best_duration),
        "t0": float(t0),
        "power": periodogram,
        "mask_transit": mask_transit,
        "time": time,
        "flux": flux
    }

    flux_detr_full = np.full(mask_valid.shape, np.nan, dtype=float)
    trend_full = np.full(mask_valid.shape, np.nan, dtype=float)
    flux_detr_full[mask_valid] = flux_detr
    trend_full[mask_valid] = trend

    return flux_detr_full, trend_full, mask_transit, bls_info

# 3 - Compute folded and binned metrics 

def folded_binned_metrics(time, flux, period, t0, lags_hours=(1,3,6,12,24), nbins=200):
    mask = np.isfinite(time) & np.isfinite(flux)
    time_u = time[mask]
    flux_u = flux[mask]
    if len(time_u) < 10:
        return {"local_noise": np.nan, "depth_stability": np.nan, "acf_lags": {h: np.nan for h in lags_hours}, "cadence_hours": np.nan}

    
    diffs = np.diff(time_u)
    diffs = diffs[np.isfinite(diffs)]
    diffs_small = diffs[diffs < 1.0]  
    dt_days = np.median(diffs_small) if len(diffs_small) > 0 else np.median(diffs)
    cadence_hours = dt_days * 24.0 if dt_days > 0 else np.nan

    phase = ((time_u - t0) / period) % 1.0
    phase = (phase + 0.5) % 1.0 - 0.5

    bins = np.linspace(-0.5, 0.5, nbins + 1)
    med, _, _ = binned_statistic(phase, flux_u, statistic="median", bins=bins)
    bin_centers = 0.5*(bins[:-1] + bins[1:])
    out_idx = np.abs(bin_centers) > 0.25
    baseline = np.nanmedian(med[out_idx]) if np.any(out_idx) else np.nanmedian(med)
    depth = baseline - np.nanmin(med)

    if np.isfinite(depth) and depth > 0:
        half = baseline - depth/2
        in_half = med < half
        if np.any(in_half):
            left = np.argmax(in_half)
            right = len(in_half) - np.argmax(in_half[::-1]) - 1
            dur_bins = max(1, right - left + 1)
            dur_phase = dur_bins / nbins
        else:
            dur_phase = max(0.05, (med.size / nbins) * 1.0)
    else:
        dur_phase = 0.05

    in_transit_mask = np.abs(phase) < (1.5 * dur_phase)
    oot_mask = ~in_transit_mask

    if np.any(oot_mask):
        mad_oot = np.nanmedian(np.abs(flux_u[oot_mask] - np.nanmedian(flux_u[oot_mask])))
        local_noise = 1.4826 * mad_oot if mad_oot > 0 else np.nanstd(flux_u[oot_mask])
    else:
        local_noise = np.nanstd(flux_u)

    epochs = np.floor((time_u - t0) / period).astype(int)
    depths = []
    for e in np.unique(epochs):
        sel = epochs == e
        if np.sum(sel) < 3:
            continue
        ph_e = phase[sel]; fl_e = flux_u[sel]
        oot_e = np.abs(ph_e) > (1.5 * dur_phase)
        baseline_e = np.nanmedian(fl_e[oot_e]) if np.any(oot_e) else np.nanmedian(fl_e)
        depths.append(baseline_e - np.nanmin(fl_e))
    depths = np.array(depths)
    depth_stability = (np.nanstd(depths)/np.nanmean(depths)) if (depths.size > 1 and np.nanmean(depths)!=0) else (0.0 if depths.size==1 else np.nan)

    try:
        x = flux_u - np.nanmean(flux_u)
        acf_vals = sm_acf(x, nlags=min(int(len(x)/2), 10000), fft=True, missing='drop')
    except Exception:
        acf_vals = np.array([np.nan])

    acf_lags = {}
    for h in lags_hours:
        if np.isnan(cadence_hours) or cadence_hours <= 0:
            acf_lags[h] = np.nan
        else:
            lag = int(round(h / cadence_hours))
            acf_lags[h] = float(acf_vals[lag]) if lag < len(acf_vals) else np.nan

    return {"local_noise": float(local_noise),
            "depth_stability": float(depth_stability),
            "acf_lags": acf_lags,
            "cadence_hours": float(cadence_hours)}

# 4 - Compute CDPP 

def calculate_cdpp(flux, cadence_hours, durations=[3.0, 6.0, 12.0]):
    flux = np.asarray(flux)
    flux = np.where(np.isfinite(flux), flux, np.nanmedian(flux))
    median = np.nanmedian(flux)
    if median == 0 or not np.isfinite(median):
        median = 1.0
    flux_norm = flux / median

    cdpp_results = {}
    points_per_hour = 1.0 / cadence_hours
    npts = len(flux_norm)

    for dur in durations:
        window = int(round(dur * points_per_hour))
        if window < 2:
            window = 2
        if window >= npts:
            window = max(2, npts // 2)
        smooth = uniform_filter1d(flux_norm, size=window, mode="nearest")
        resid = flux_norm - smooth
        cdpp_val = np.nanstd(resid) * 1e6 if np.any(np.isfinite(resid)) else np.nan
        cdpp_results[f'cdpp_{int(dur)}h'] = float(cdpp_val)

    return cdpp_results

# 5 - Compute SES and MES

def compute_SES_MES(depths, local_noise, npts_in_transit, cdpp_dict=None, duration_hours=None, method='auto'):
    """
    depths: array (flux units, e.g. relative flux like 0.0067)
    local_noise: sigma per-point (same flux units)
    npts_in_transit: array of ints (n points used per transit)
    cdpp_dict: dict with keys 'cdpp_3h','cdpp_6h','cdpp_12h' (ppm)
    duration_hours: approximate transit duration in hours (float)
    method: 'auto'|'cdpp'|'point_sigma'
    Returns: {'SES': array, 'MES': float}
    """
    depths = np.asarray(depths, dtype=float)
    npts = np.asarray(npts_in_transit, dtype=float)

    if depths.size == 0 or npts.size == 0:
        return {'SES': np.array([]), 'MES': np.nan}

    def cdpp_interp(h):
        if not cdpp_dict or duration_hours is None:
            return np.nan
        c3 = cdpp_dict.get('cdpp_3h', np.nan)
        c6 = cdpp_dict.get('cdpp_6h', np.nan)
        c12 = cdpp_dict.get('cdpp_12h', np.nan)
        if not np.isfinite(c3) or not np.isfinite(c6) or not np.isfinite(c12):
            return np.nan
        if h <= 3:
            return c3
        elif h <= 6:
            return c3 + (c6 - c3) * (h - 3) / (6 - 3)
        elif h <= 12:
            return c6 + (c12 - c6) * (h - 6) / (12 - 6)
        else:
            return c12

    use_cdpp = False
    if method == 'cdpp':
        use_cdpp = True
    elif method == 'auto':
        use_cdpp = (cdpp_dict is not None and duration_hours is not None and np.isfinite(cdpp_interp(duration_hours)))

    if use_cdpp:
        cdpp_est = float(cdpp_interp(duration_hours))
        depths_ppm = depths * 1e6
        SES = np.full_like(depths_ppm, np.nan, dtype=float)
        if np.isfinite(cdpp_est) and cdpp_est > 0:
            SES = depths_ppm / cdpp_est     
    else:
        SES = np.full_like(depths, np.nan, dtype=float)
        if local_noise is not None and np.isfinite(local_noise) and local_noise > 0:
            valid = (npts > 0) & np.isfinite(depths)
            SES[valid] = depths[valid] / (local_noise / np.sqrt(npts[valid]))

    ses_valid = SES[np.isfinite(SES)]
    MES = float(np.sqrt(np.nansum(ses_valid**2))) if ses_valid.size > 0 else np.nan

    return {'SES': SES, 'MES': MES}

# 6 - Compute per-transit statistics

def per_transit_stats_simple(time, flux, period, t0, transit_duration_days, window_factor=2.0, min_points=3):
    time = np.asarray(time); flux = np.asarray(flux)
    mask = np.isfinite(time) & np.isfinite(flux)
    if np.sum(mask) < min_points:
        return {'per_transit': [], 'depths': np.array([]), 'npts_in_transit': np.array([])}
    t = time[mask]; f = flux[mask]
    if transit_duration_days is None or transit_duration_days <= 0:
        transit_duration_days = 0.1
    # Floor duration to ensure at least ~3 cadence samples per transit
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size:
        dt_days = np.median(diffs)
        min_dur = 3.0 * dt_days
        if np.isfinite(transit_duration_days) and transit_duration_days < min_dur:
            transit_duration_days = min_dur
    epochs = np.floor((t - t0) / period + 1e-12).astype(int)
    unique_epochs = np.unique(epochs)
    per_transit = []; depths = []; npts_list = []
    half_window = window_factor * transit_duration_days
    for e in unique_epochs:
        center_time = t0 + e * period
        sel = (t >= center_time - half_window) & (t <= center_time + half_window)
        if np.sum(sel) < min_points:
            continue
        t_e = t[sel]; f_e = f[sel]
        in_tr_mask = np.abs(t_e - center_time) < (transit_duration_days / 2.0)
        if np.sum(~in_tr_mask) >= max(3, int(0.3*np.sum(sel))):
            baseline = float(np.nanmedian(f_e[~in_tr_mask]))
        else:
            baseline = float(np.nanmedian(f_e))
        min_in = float(np.nanmin(f_e[in_tr_mask])) if np.any(in_tr_mask) else float(np.min(f_e))
        depth_i = baseline - min_in
        model = np.full_like(f_e, baseline)
        if np.any(in_tr_mask):
            model[in_tr_mask] = baseline - depth_i
        resid = f_e - model
        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        sigma = 1.4826 * mad if (mad > 0 and np.isfinite(mad)) else np.nanstd(resid)
        chi2 = float(np.nansum((resid / sigma)**2)) if (np.isfinite(sigma) and sigma > 0) else np.nan
        per_transit.append({'epoch': int(e), 'center_time': float(center_time), 'npts': int(np.sum(sel)),
                             'baseline': baseline, 'depth': float(depth_i), 'resid_rms': float(np.nanstd(resid)), 'chi2': chi2})
        depths.append(depth_i); npts_list.append(int(np.sum(sel)))
    depths = np.array(depths) if len(depths)>0 else np.array([])
    npts_arr = np.array(npts_list) if len(npts_list)>0 else np.array([])
    return {'per_transit': per_transit, 'depths': depths, 'npts_in_transit': npts_arr}


# 7 - Utils

def augment_folded_metrics(binned_metrics, time, flux_detr, period, transit_duration, t0):
    feats = dict(binned_metrics)
    mask = np.isfinite(time)
    t = np.asarray(time)[mask]
    data_span = float(t[-1] - t[0]) if len(t)>=2 else 0.0
    expected = data_span / period if period > 0 else np.nan
    epochs = np.floor((t - t0) / period).astype(int)
    observed_transits = len(np.unique(epochs))
    duty_cycle = transit_duration / period if period > 0 else np.nan
    feats.update({'data_span_days': data_span, 'expected_transits': expected,
                  'n_observed_transits': observed_transits, 'duty_cycle': duty_cycle})
    per = per_transit_stats_simple(time, flux_detr, period, t0, transit_duration)
    depths = per['depths']; npts = per['npts_in_transit']
    feats.update({'depth_mean_per_transit': float(np.nanmean(depths)) if depths.size else np.nan,
                  'depth_std_per_transit': float(np.nanstd(depths)) if depths.size else np.nan,
                  'npts_transit_median': float(np.median(npts)) if npts.size else np.nan})
    cdpp = calculate_cdpp(flux_detr, cadence_hours = feats.get('cadence_hours', np.nan))
    sesmes = compute_SES_MES(depths,
                             local_noise = feats.get('local_noise', np.nan),
                             npts_in_transit = npts,
                             cdpp_dict = cdpp,
                             duration_hours = transit_duration * 24.0)
    feats['MES'] = sesmes['MES']
    feats['SES_mean'] = float(np.nanmean(sesmes['SES'])) if sesmes['SES'].size else np.nan
    feats['SES_std'] = float(np.nanstd(sesmes['SES'])) if sesmes['SES'].size else np.nan
    feats['centroid_shift_x'] = np.nan
    feats['centroid_shift_y'] = np.nan
    feats.update(cdpp)
    return feats

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
    

# Extract features from light curve

def extract_all_features_from_csv(csv_path, verbose=False):
    """Extract features from a CSV file containing light curve data."""
    import pandas as pd
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    time = df['time'].values
    flux = df['flux'].values
    
    if verbose:
        print(f"Loaded light curve data from: {csv_path}")
        print(f"Data points: {len(time)}")
    
    # Use the same feature extraction logic as extract_all_features_v2
    return _extract_features_from_arrays(time, flux, verbose=verbose)


def _extract_features_from_arrays(time, flux, verbose=False, refine_duration=True):
    """Internal function to extract features from time and flux arrays."""
    feats = OrderedDict()

    mask0 = np.isfinite(time) & np.isfinite(flux)
    if np.sum(mask0) < 3:
        raise ValueError("too few valid points")
    time_arr = np.asarray(time)[mask0]
    flux_arr = np.asarray(flux)[mask0]

    flux_detr_full, trend_full, mask_transit, bls_info = detrend_with_bls_mask(time_arr, flux_arr, refine_duration=refine_duration)
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

    # Note: For CSV data, we don't have stellar radius information
    feats["planet_radius_rearth"] = np.nan
    feats["planet_radius_rjup"]   = np.nan

    if verbose:
        print("\n=== Extracted features (ordered) ===")
        for k, v in feats.items():
            print(f"{k}: {v}")

    return feats


def extract_all_features_v2(target, mission="Kepler", sigma_clip=5.0, verbose=False, refine_duration=True):
    """Extract features from a target by downloading and processing light curve data."""
    lc = download_and_clean(target=target, mission=mission, sigma_clip=sigma_clip)
    time = lc.time.value
    flux = lc.flux.value
    
    # Use the internal function for feature extraction
    feats = _extract_features_from_arrays(time, flux, verbose=verbose, refine_duration=refine_duration)
    
    # Add stellar radius information if available
    if "RADIUS" in lc.meta and np.isfinite(lc.meta["RADIUS"]):
        stellar_radius = lc.meta["RADIUS"]  
        Rp_over_Rs = np.sqrt(feats["depth_mean_per_transit"])
        feats["planet_radius_rearth"] = stellar_radius * 109.1 * Rp_over_Rs
        feats["planet_radius_rjup"]   = stellar_radius * 9.95 * Rp_over_Rs
    
    return feats

# For batch processing

def batch_extract_features_from_targets(targets, mission="Kepler", sigma_clip=5.0, n_workers=None, verbose=True, refine_duration=True):
    targets = list(targets or [])
    total = len(targets)
    if n_workers is None:
        import os as _os
        n_workers = _os.cpu_count() or 1
    n_workers = max(1, int(n_workers))

    def _worker(targ):
        try:
            feats = extract_all_features_v2(targ, mission=mission, sigma_clip=sigma_clip, verbose=False, refine_duration=refine_duration)
            out = OrderedDict(feats)
            out["target"] = targ
            return out
        except Exception as e:
            out = OrderedDict()
            out["target"] = targ
            out["error"] = str(e)
            return out

    if n_workers == 1:
        results = []
        for i, targ in enumerate(targets):
            results.append(_worker(targ))
            if verbose and total >= 10 and ((i + 1) % max(1, total // 10) == 0 or (i + 1) == total):
                print(f"Processed {i+1}/{total}")
        return results

    results_by_index = [None] * total
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_worker, targ): idx for idx, targ in enumerate(targets)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = OrderedDict()
                res["target"] = targets[idx]
                res["error"] = str(e)
            results_by_index[idx] = res
            completed += 1
            if verbose and total >= 10 and (completed % max(1, total // 10) == 0 or completed == total):
                print(f"Completed {completed}/{total}")
    return [r for r in results_by_index if r is not None]
