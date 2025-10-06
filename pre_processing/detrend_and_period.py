import numpy as np
from scipy.interpolate import UnivariateSpline
from astropy.timeseries import BoxLeastSquares 
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic
from transitleastsquares import transitleastsquares


def detrend_with_bls_mask(time, flux,
                          min_period=0.5, max_period=None,
                          n_periods=2000, n_durations=50,
                          oversample=10,
                          bin_width=0.5, spline_s=0.001,
                          max_iter=5, sigma=3.0,
                          refine_duration=True,
                          best_floor=0.01):
    mask_valid = np.isfinite(time) & np.isfinite(flux)
    time = np.asarray(time)[mask_valid]
    flux = np.asarray(flux)[mask_valid]
    print("Detrend version 2")

    if max_period is None:
        max_period = (time.max() - time.min()) / 3.0

    durations = np.linspace(0.01, 0.2, n_durations)
    period_grid = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)

    bls = BoxLeastSquares(time, flux)
    periodogram = bls.power(period_grid, durations, oversample=oversample)

    # Detect initial best period/duration
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

    # --- Refinamento opcional ---
    if refine_duration and np.isfinite(best_period) and np.isfinite(best_duration):
        print("Starting period/duration refinement...")
        per_lo = best_period * 0.98
        per_hi = best_period * 1.02
        dur_lo = best_floor
        dur_hi = min(0.5, 0.2 * best_period)
        if dur_hi > dur_lo:
            durations_refined = np.linspace(dur_lo, dur_hi, n_durations)
            periods_refined = np.logspace(np.log10(per_lo), np.log10(per_hi), max(256, n_periods // 4))
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

            if (np.isfinite(best_period_refined) and np.isfinite(best_duration_refined) and
                best_period_refined > 0 and best_duration_refined > 0):
                best_period = best_period_refined
                best_duration = best_duration_refined
                t0 = t0_refined
                periodogram = periodogram_refined

    # --- Criação da máscara de trânsito e detrending ---
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


# ========================================================================================================================

# def detrend_with_bls_mask(
#     time, flux,
#     min_period=0.5, max_period=None,
#     n_periods=2000, n_durations=50,
#     oversample=10,
#     bin_width=0.5, spline_s=0.001,
#     max_iter=5, sigma=3.0,
#     n_top_candidates=50
# ):
#     """
#     BLS detrend com busca global robusta e retorno de múltiplos períodos candidatos.
#     Mantém o comportamento original, mas retorna lista de períodos ordenados por power.
#     """

#     mask_valid = np.isfinite(time) & np.isfinite(flux)
#     time = np.asarray(time)[mask_valid]
#     flux = np.asarray(flux)[mask_valid]

#     if max_period is None:
#         max_period = (time.max() - time.min()) / 3.0

#     durations = np.linspace(0.01, 0.2, n_durations)
#     period_grid = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)

#     bls = BoxLeastSquares(time, flux)
#     periodogram = bls.power(period_grid, durations, oversample=oversample)

#     # --- Seleção dos candidatos ---
#     if np.ndim(periodogram.power) == 1:
#         power = periodogram.power
#         t0s = periodogram.transit_time
#         durs = periodogram.duration
#     else:
#         power = np.nanmax(periodogram.power, axis=1)
#         dur_idx = np.nanargmax(periodogram.power, axis=1)
#         durs = periodogram.duration[dur_idx]
#         t0s = periodogram.transit_time[dur_idx]

#     sort_idx = np.argsort(power)[::-1]
#     top_idx = sort_idx[:n_top_candidates]
#     candidate_periods = periodogram.period[top_idx]
#     candidate_powers = power[top_idx]
#     candidate_durations = durs[top_idx]
#     candidate_t0s = t0s[top_idx]

#     # --- Melhor candidato (maior power) ---
#     idx_best = top_idx[0]
#     best_period = periodogram.period[idx_best]
#     best_duration = durs[idx_best]
#     t0 = t0s[idx_best]

#     print(f"[INFO] Melhor período: {best_period:.6f} d, duração {best_duration:.4f} d, t0={t0:.4f}")
#     print(f"[INFO] Top {n_top_candidates} candidatos ordenados por power extraídos.")

#     # --- Detrending usando o melhor ---
#     mask_transit = bls.transit_mask(time, best_period, best_duration, t0)
#     flux_work = flux.copy()
#     mask_use = ~mask_transit

#     for _ in range(max_iter):
#         bins = np.arange(time.min(), time.max() + bin_width, bin_width)
#         digitized = np.digitize(time, bins)
#         bin_means, bin_times = [], []
#         for i in range(1, len(bins)):
#             sel = (digitized == i) & mask_use
#             if np.any(sel):
#                 bin_means.append(np.median(flux_work[sel]))
#                 bin_times.append(np.median(time[sel]))
#         bin_means = np.array(bin_means)
#         bin_times = np.array(bin_times)
#         good = np.isfinite(bin_means) & np.isfinite(bin_times)
#         if good.sum() < 5:
#             break
#         spline = UnivariateSpline(bin_times[good], bin_means[good], s=spline_s)
#         trend = spline(time)
#         resid = flux_work - trend
#         mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
#         thresh = -sigma * 1.4826 * mad
#         mask_use = mask_use & (resid > thresh)

#     if np.sum(mask_use) < 3:
#         spline = UnivariateSpline(time, flux, s=spline_s)
#     else:
#         spline = UnivariateSpline(time[mask_use], flux[mask_use], s=spline_s)

#     trend = spline(time)
#     flux_detr = flux / trend

#     # --- Empacotamento dos resultados ---
#     bls_info = {
#         "best_period": float(best_period),
#         "best_duration": float(best_duration),
#         "t0": float(t0),
#         "periodogram": periodogram,
#         "mask_transit": mask_transit,
#         "time": time,
#         "flux": flux,
#         "candidate_periods": candidate_periods,
#         "candidate_powers": candidate_powers,
#         "candidate_durations": candidate_durations,
#         "candidate_t0s": candidate_t0s
#     }

#     flux_detr_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     trend_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     flux_detr_full[mask_valid] = flux_detr
#     trend_full[mask_valid] = trend

#     return flux_detr_full, trend_full, mask_transit, bls_info

# ========================================================================================================================

# def detrend_with_bls_mask(time, flux,
#                           min_period=0.5, max_period=None,
#                           n_periods=2000, n_durations=50,
#                           oversample=10,
#                           bin_width=0.5, spline_s=0.001,
#                           max_iter=5, sigma=3.0):
#     mask_valid = np.isfinite(time) & np.isfinite(flux)
#     time = np.asarray(time)[mask_valid]
#     flux = np.asarray(flux)[mask_valid]
#     print("Detrend version 1")
#     if max_period is None:
#         max_period = (time.max() - time.min()) / 3.0

#     durations = np.linspace(0.01, 0.2, n_durations)
#     period_grid = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)

#     bls = BoxLeastSquares(time, flux)
#     periodogram = bls.power(period_grid, durations, oversample=oversample)

#     if np.ndim(periodogram.power) == 1:
#         idx_best = np.nanargmax(periodogram.power)
#         best_period = periodogram.period[idx_best]
#         best_duration = periodogram.duration[idx_best] if hasattr(periodogram, "duration") else durations[0]
#         t0 = periodogram.transit_time[idx_best] if hasattr(periodogram, "transit_time") else time[0]
#     else:
#         power_per_period = np.nanmax(periodogram.power, axis=1)
#         idx_best = int(np.nanargmax(power_per_period))
#         best_period = periodogram.period[idx_best]
#         dur_idx = int(np.nanargmax(periodogram.power[idx_best, :]))
#         best_duration = periodogram.duration[dur_idx] if hasattr(periodogram, "duration") else durations[dur_idx]
#         t0 = periodogram.transit_time[idx_best] if hasattr(periodogram, "transit_time") else time[0]

#     mask_transit = bls.transit_mask(time, best_period, best_duration, t0)

#     flux_work = flux.copy()
#     mask_use = ~mask_transit

#     for _ in range(max_iter):
#         bins = np.arange(time.min(), time.max() + bin_width, bin_width)
#         digitized = np.digitize(time, bins)
#         bin_means = []
#         bin_times = []
#         for i in range(1, len(bins)):
#             sel = (digitized == i) & mask_use
#             if np.any(sel):
#                 bin_means.append(np.median(flux_work[sel]))
#                 bin_times.append(np.median(time[sel]))
#         bin_means = np.array(bin_means)
#         bin_times = np.array(bin_times)

#         good = np.isfinite(bin_means) & np.isfinite(bin_times)
#         if good.sum() < 5:
#             break
#         spline = UnivariateSpline(bin_times[good], bin_means[good], s=spline_s)
#         trend = spline(time)
#         resid = flux_work - trend

#         mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
#         thresh = -sigma * 1.4826 * mad
#         mask_use = mask_use & (resid > thresh)

#     if np.sum(mask_use) < 3:
#         spline = UnivariateSpline(time, flux, s=spline_s)
#     else:
#         spline = UnivariateSpline(time[mask_use], flux[mask_use], s=spline_s)
#     trend = spline(time)
#     flux_detr = flux / trend

#     bls_info = {
#         "best_period": float(best_period),
#         "best_duration": float(best_duration),
#         "t0": float(t0),
#         "power": periodogram,
#         "mask_transit": mask_transit,
#         "time": time,
#         "flux": flux
#     }

#     flux_detr_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     trend_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     flux_detr_full[mask_valid] = flux_detr
#     trend_full[mask_valid] = trend

#     return flux_detr_full, trend_full, mask_transit, bls_info

# ========================================================================================================================

# def detrend_with_bls_mask(time, flux,
#                           min_period=0.5, max_period=None,
#                           n_periods=2000, n_durations=50,
#                           oversample=10,
#                           bin_width=0.5, spline_s=0.001,
#                           max_iter=5, sigma=3.0):
#     # Filtra os valores válidos
#     mask_valid = np.isfinite(time) & np.isfinite(flux)
#     time = np.asarray(time)[mask_valid]
#     flux = np.asarray(flux)[mask_valid]

#     # Ordena o tempo e reorganiza o fluxo correspondente
#     sorted_indices = np.argsort(time)
#     time = time[sorted_indices]
#     flux = flux[sorted_indices]

#     # Calcula o período máximo se não fornecido
#     if max_period is None:
#         max_period = (time.max() - time.min()) / 3.0

#     # Define a grade de períodos e durações
#     durations = np.linspace(0.01, 0.2, n_durations)
#     period_grid = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)

#     # Realiza a análise de Box Least Squares
#     bls = BoxLeastSquares(time, flux)
#     periodogram = bls.power(period_grid, durations, oversample=oversample)

#     # Identifica o melhor período, duração e tempo de início
#     if np.ndim(periodogram.power) == 1:
#         idx_best = np.nanargmax(periodogram.power)
#         best_period = periodogram.period[idx_best]
#         best_duration = periodogram.duration[idx_best] if hasattr(periodogram, "duration") else durations[0]
#         t0 = periodogram.transit_time[idx_best] if hasattr(periodogram, "transit_time") else time[0]
#     else:
#         power_per_period = np.nanmax(periodogram.power, axis=1)
#         idx_best = int(np.nanargmax(power_per_period))
#         best_period = periodogram.period[idx_best]
#         dur_idx = int(np.nanargmax(periodogram.power[idx_best, :]))
#         best_duration = periodogram.duration[dur_idx] if hasattr(periodogram, "duration") else durations[dur_idx]
#         t0 = periodogram.transit_time[idx_best] if hasattr(periodogram, "transit_time") else time[0]

#     # Cria a máscara de trânsito
#     mask_transit = bls.transit_mask(time, best_period, best_duration, t0)

#     # Inicializa o trabalho com o fluxo
#     flux_work = flux.copy()
#     mask_use = ~mask_transit

#     # Itera para refinar a máscara de outliers
#     for _ in range(max_iter):
#         bins = np.arange(time.min(), time.max() + bin_width, bin_width)
#         digitized = np.digitize(time, bins)
#         bin_means = []
#         bin_times = []
#         for i in range(1, len(bins)):
#             sel = (digitized == i) & mask_use
#             if np.any(sel):
#                 bin_means.append(np.median(flux_work[sel]))
#                 bin_times.append(np.median(time[sel]))
#         bin_means = np.array(bin_means)
#         bin_times = np.array(bin_times)

#         good = np.isfinite(bin_means) & np.isfinite(bin_times)
#         if good.sum() < 5:
#             break
#         spline = UnivariateSpline(bin_times[good], bin_means[good], s=spline_s, k=1)
#         trend = spline(time)
#         resid = flux_work - trend

#         mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
#         thresh = -sigma * 1.4826 * mad
#         mask_use = mask_use & (resid > thresh)

#     # Ajusta o spline final
#     if np.sum(mask_use) < 3:
#         spline = UnivariateSpline(time, flux, s=spline_s)
#     else:
#         spline = UnivariateSpline(time[mask_use], flux[mask_use], s=spline_s)
#     trend = spline(time)
#     flux_detr = flux / trend

#     # Armazena as informações do BLS
#     bls_info = {
#         "best_period": float(best_period),
#         "best_duration": float(best_duration),
#         "t0": float(t0),
#         "power": periodogram,
#         "mask_transit": mask_transit,
#         "time": time,
#         "flux": flux
#     }

#     # Preenche os resultados finais com NaN para os valores inválidos
#     flux_detr_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     trend_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     flux_detr_full[mask_valid] = flux_detr
#     trend_full[mask_valid] = trend

#     return flux_detr_full, trend_full, mask_transit, bls_info

# ========================================================================================================================

# def detrend_with_bls_mask(time, flux,
#                           min_period=0.5, max_period=None,
#                           n_periods=2000, n_durations=50,
#                           oversample=10,
#                           window_frac=0.02, polyorder=2,
#                           max_iter=5, sigma=3.0,
#                           verbose=False):
#     # Filtra os valores válidos
#     mask_valid = np.isfinite(time) & np.isfinite(flux)
#     time = np.asarray(time)[mask_valid]
#     flux = np.asarray(flux)[mask_valid]

#     # Ordena
#     sorted_indices = np.argsort(time)
#     time = time[sorted_indices]
#     flux = flux[sorted_indices]

#     timespan = time.max() - time.min()
#     if timespan <= 0:
#         raise ValueError("time span inválido")

#     # Ajuste seguro de max_period: metade do baseline por padrão (mais permissivo que /3)
#     if max_period is None:
#         max_period = max(min_period * 2.0, timespan / 2.0)

#     # grade de durações (em unidades de dias) e períodos (dias)
#     durations = np.linspace(0.01, 0.2, n_durations)
#     period_grid = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)

#     # Executa BLS (USAMOS OS DADOS NÃO-DESLINED)
#     bls = BoxLeastSquares(time, flux)
#     periodogram = bls.power(period_grid, durations, oversample=oversample)
#     # show_top_bls_peaks(periodogram, n_peaks=50)


#     # Analisa a forma da power array e encontra o melhor período/duração
#     if np.ndim(periodogram.power) == 1:
#         idx_best = int(np.nanargmax(periodogram.power))
#         best_period = float(periodogram.period[idx_best])
#         best_duration = float(periodogram.duration[idx_best]) if hasattr(periodogram, "duration") else float(durations[0])
#         # transit_time pode não existir como esperado; estimamos t0 abaixo
#         t0_est = None
#     else:
#         power_per_period = np.nanmax(periodogram.power, axis=1)
#         idx_best = int(np.nanargmax(power_per_period))
#         best_period = float(periodogram.period[idx_best])
#         dur_idx = int(np.nanargmax(periodogram.power[idx_best, :]))
#         # extrai duração; se atributo duration existir, use-o; senão use duracoes da grade
#         try:
#             best_duration = float(periodogram.duration[dur_idx])
#         except Exception:
#             best_duration = float(durations[dur_idx])
#         t0_est = None

#     # Opcional: mostrar os 5 picos mais altos (diagnóstico)
#     if verbose:
#         # para 2D power, colapsa por periodo
#         if np.ndim(periodogram.power) == 2:
#             power_c = np.nanmax(periodogram.power, axis=1)
#         else:
#             power_c = periodogram.power
#         top_idx = np.argsort(power_c)[-5:][::-1]
#         print("time span (days):", timespan)
#         print("period grid min/max: %.3f / %.3f" % (period_grid.min(), period_grid.max()))
#         print("top BLS periods (days) and power:")
#         for ii in top_idx:
#             print(f"  {periodogram.period[ii]:.6f} -> power {power_c[ii]:.6g}")
#         print("chosen best_period:", best_period, "best_duration:", best_duration)

#     # Estima t0 por folding simples (mais robusto que confiar em transit_time)
#     try:
#         phase = ((time - time[0]) / best_period) % 1.0
#         # centraliza em -0.5..0.5
#         phase = (phase + 0.5) % 1.0 - 0.5
#         nbins = max(100, int(len(time) / 1000))  # bins adaptativos (mín 100)
#         bins = np.linspace(-0.5, 0.5, nbins + 1)
#         bin_median, _, _ = binned_statistic(phase, flux, statistic="median", bins=bins)
#         min_bin = np.nanargmin(bin_median)
#         bin_centers = 0.5 * (bins[:-1] + bins[1:])
#         phase_min = bin_centers[min_bin]
#         # transforma de volta para tempo: queremos o t0 mais próximo do início
#         t0 = float(time[0] + ((phase_min + 0.5) % 1.0) * best_period)
#     except Exception:
#         t0 = float(time[0])

#     # Cria a máscara de trânsito com a melhor solução
#     mask_transit = bls.transit_mask(time, best_period, best_duration, t0)

#     # Preparação para detrending com SavGol
#     flux_work = flux.copy()
#     mask_use = ~mask_transit

#     # janela adaptativa: fração do n_pts (garante impar e <= n_pts)
#     win_len = int(len(time) * window_frac)
#     if win_len % 2 == 0:
#         win_len += 1
#     win_len = max(win_len, polyorder + 1)
#     if win_len > len(time):
#         win_len = len(time) if (len(time) % 2 == 1) else (len(time) - 1)

#     if verbose:
#         print("savgol window length:", win_len, "polyorder:", polyorder)

#     # Itera para refinar máscara de outliers (aplica SavGol apenas aos pontos selecionados)
#     for _ in range(max_iter):
#         # se mask_use tiver poucos pontos, computa trend sobre todo o conjunto (fallback)
#         if np.sum(mask_use) <= win_len:
#             trend_full = savgol_filter(flux, window_length=win_len, polyorder=min(polyorder, max(1, win_len-1)), mode="interp")
#             resid = flux_work - trend_full
#         else:
#             # Para evitar chamar savgol com um array muito pequeno, construímos um trend interpolado:
#             t_sel = time[mask_use]
#             f_sel = flux_work[mask_use]
#             # se f_sel muito pequeno em tamanho, use fallback
#             if len(f_sel) <= win_len:
#                 trend_sel = savgol_filter(flux, window_length=win_len, polyorder=min(polyorder, max(1, win_len-1)), mode="interp")
#                 resid = flux_work - trend_sel
#             else:
#                 trend_sel = savgol_filter(f_sel, window_length=win_len if win_len <= len(f_sel) else (len(f_sel)-1 if (len(f_sel)-1)%2==1 else len(f_sel)), polyorder=min(polyorder, max(1, win_len-1)), mode="interp")
#                 # interpola trend_sel nos tempos originais
#                 trend_full = np.interp(time, t_sel, trend_sel, left=np.nan, right=np.nan)
#                 # para bins sem valor, preenche com trend do filtro aplicado a todo o fluxo
#                 mask_nan = ~np.isfinite(trend_full)
#                 if np.any(mask_nan):
#                     fallback_trend = savgol_filter(flux, window_length=win_len, polyorder=min(polyorder, max(1, win_len-1)), mode="interp")
#                     trend_full[mask_nan] = fallback_trend[mask_nan]
#                 resid = flux_work - trend_full

#         mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
#         thresh = -sigma * 1.4826 * mad
#         # atualiza mask_use: remove pontos que estão abaixo do threshold (prováveis trânsitos/outliers negativos)
#         mask_use = mask_use & (resid > thresh)

#     # Filtro final aplicado em todo o fluxo
#     final_trend = savgol_filter(flux, window_length=win_len, polyorder=min(polyorder, max(1, win_len-1)), mode="interp")
#     flux_detr = flux / final_trend

#     bls_info = {
#         "best_period": float(best_period),
#         "best_duration": float(best_duration),
#         "t0": float(t0),
#         "power": periodogram,
#         "mask_transit": mask_transit,
#         "time": time,
#         "flux": flux
#     }

#     # Reconstrói vetores com NaNs na posição original
#     flux_detr_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     trend_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     flux_detr_full[mask_valid] = flux_detr
#     trend_full[mask_valid] = final_trend

#     return flux_detr_full, trend_full, mask_transit, bls_info

# ========================================================================================================================

# def detrend_with_tls(time, flux, min_period=0.5, max_period=None, n_periods=2000,
#                       n_durations=50, oversample=10, window_frac=0.02, polyorder=2,
#                       max_iter=5, sigma=3.0, verbose=False):
#     # Filtra os valores válidos
#     mask_valid = np.isfinite(time) & np.isfinite(flux)
#     time = np.asarray(time)[mask_valid]
#     flux = np.asarray(flux)[mask_valid]

#     # Ordena os dados
#     sorted_indices = np.argsort(time)
#     time = time[sorted_indices]
#     flux = flux[sorted_indices]

#     # Determina o intervalo de tempo
#     timespan = time.max() - time.min()
#     if timespan <= 0:
#         raise ValueError("Intervalo de tempo inválido")

#     # Ajuste seguro de max_period
#     if max_period is None:
#         max_period = max(min_period * 2.0, timespan / 2.0)

#     # Executa TLS
#     flux_flat = pre_detrend_flux(time, flux, window_length_days=5, polyorder=3)
#     model = transitleastsquares(time, flux_flat)
#     results = model.power(period_max=max_period)
#     # Exibe os resultados principais
#     print(f"Período: {results.period:.5f} dias")
#     print(f"Profundidade do trânsito: {results.depth:.5f}")
#     print(f"Melhor duração: {results.duration:.5f} dias")
#     print(f"Eficiência de detecção do sinal (SDE): {results.SDE:.2f}")

#     if verbose:
#         # Exibe os 5 principais picos
#         top_idx = np.argsort(results.power)[-5:][::-1]
#         print("Top 5 picos:")
#         for idx in top_idx:
#             print(f"Período: {results.periods[idx]:.5f} dias, SDE: {results.power[idx]:.2f}")

#     # Estima o tempo de início do trânsito (t0)
#     phase = ((time - time[0]) / results.period) % 1.0
#     phase = (phase + 0.5) % 1.0 - 0.5
#     nbins = max(100, int(len(time) / 1000))
#     bins = np.linspace(-0.5, 0.5, nbins + 1)
#     bin_median, _, _ = binned_statistic(phase, flux, statistic="median", bins=bins)
#     min_bin = np.nanargmin(bin_median)
#     bin_centers = 0.5 * (bins[:-1] + bins[1:])
#     phase_min = bin_centers[min_bin]
#     t0 = float(time[0] + ((phase_min + 0.5) % 1.0) * results.period)

#     # Cria a máscara de trânsito
#     # mask_transit = model.transit_mask(
#     #     time, results.period, results.duration, t0
#     # )

#     # Detrend com Savitzky-Golay
#     flux_detrended = flux / savgol_filter(flux, window_length=11, polyorder=3)

#     return flux_detrended, {
#         "period": results.period,
#         "duration": results.duration,
#         "depth": results.depth,
#         "t0": t0,
#         "SDE": results.SDE
#     }

# def pre_detrend_flux(time, flux, window_length_days=5.0, polyorder=3):
#     """
#     Pré-detrending do fluxo usando Savitzky-Golay.
    
#     time: array de tempos em dias
#     flux: array de fluxos normalizados
#     window_length_days: janela do filtro em dias
#     polyorder: ordem do polinômio para ajuste
    
#     Retorna:
#         flux_detrended: flux detrended
#     """
#     time = np.asarray(time)
#     flux = np.asarray(flux)
    
#     # calcula o tamanho da janela em número de pontos
#     diffs = np.diff(time)
#     median_dt = np.median(diffs[np.isfinite(diffs)])
#     window_length_pts = int(round(window_length_days / median_dt))
    
#     # o window_length do Savitzky-Golay precisa ser ímpar
#     if window_length_pts % 2 == 0:
#         window_length_pts += 1
    
#     # garante tamanho mínimo de 5 pontos
#     window_length_pts = max(window_length_pts, 5)
    
#     flux_trend = savgol_filter(flux, window_length=window_length_pts, polyorder=polyorder)
#     flux_detrended = flux / flux_trend  # ou flux - flux_trend, dependendo da escala
    
#     return flux_detrended