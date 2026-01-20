import numpy as np
import csv
from scipy.optimize import curve_fit, least_squares

def fit_sine_wave(timestamps, angles, method='multi'):
    """
    Master function to dispatch to the specific fitting strategy.
    Args:
        method (str): 'single' for best-continuous-segment, 'multi' for joint-optimization.
    """
    if method == 'multi':
        print("multi method called")
        return fit_sine_wave_multi_segment(timestamps, angles)
    else:
        print("single method called")
        return fit_sine_wave_single_segment(timestamps, angles)


def fit_sine_wave_multi_segment(timestamps, angles):
    """
    STRATEGY 2: JOINT OPTIMIZATION ("Global Frequency, Local Phase")
    
    Uses ALL data clusters by solving a custom least_squares problem.
    Assumes Amplitude, Frequency, and Offset are constant, but Phase shifts 
    arbitrarily between time gaps.
    """
    # 1. Clean Data
    t_all = np.array(timestamps)
    y_all = np.array(angles)
    valid = ~np.isnan(y_all)
    t_clean = t_all[valid]
    y_clean = y_all[valid]

    # --- DEBUG CSV ---
    try:
        with open("debug_curve_multi.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "KneeAngle"])
            for ti, yi in zip(t_clean, y_clean):
                writer.writerow([ti, yi])
    except: pass
    # -----------------

    if len(t_clean) < 20: 
        return None

    # 2. CLUSTER DETECTION
    # Group points separated by gaps > 1.0s
    time_diffs = np.diff(t_clean)
    gap_indices = np.where(time_diffs > 1.0)[0]
    split_indices = np.concatenate(([0], gap_indices + 1, [len(t_clean)]))
    
    segments = []
    total_valid_points = 0
    
    for i in range(len(split_indices) - 1):
        start, end = split_indices[i], split_indices[i+1]
        # Keep segments with > 10 frames
        if end - start > 10:
            segments.append({
                't': t_clean[start:end],
                'y': y_clean[start:end]
            })
            total_valid_points += (end - start)
            
    if not segments:
        print("[DEBUG] No valid segments found for multi-fit.")
        return None

    print(f"[DEBUG] Multi-Fit: Joint optimization on {len(segments)} segments ({total_valid_points} pts)")

    # 3. DEFINE JOINT MODEL & RESIDUALS
    # Params Vector: [Offset, Amp, Freq, Phase_1, Phase_2, ... Phase_N]
    
    def residuals_func(params):
        offset, amp, freq = params[0], params[1], params[2]
        
        all_residuals = []
        
        # params[3] is Phase_1, params[4] is Phase_2, etc.
        for i, seg in enumerate(segments):
            phase_i = params[3 + i]
            
            # Model: y = Offset + Amp * sin(2*pi*Freq*t + Phase_i)
            y_pred = offset + amp * np.sin(2 * np.pi * freq * seg['t'] + phase_i)
            
            resid = seg['y'] - y_pred
            all_residuals.append(resid)
            
        return np.concatenate(all_residuals)

    # 4. INITIAL GUESSES & BOUNDS
    y_concat = np.concatenate([s['y'] for s in segments])
    
    guess_offset = np.mean(y_concat)
    guess_amp = (np.max(y_concat) - np.min(y_concat)) / 2
    guess_freq = 1.5 # 90 RPM
    
    # Init Params: [Offset, Amp, Freq, 0, 0, ... 0]
    x0 = [guess_offset, guess_amp, guess_freq] + [0] * len(segments)
    
    # Global Bounds: Offset(80-150), Amp(0-60), Freq(0.5-2.5)
    # Phase Bounds: -inf to +inf
    lower_bounds = [80,   0, 0.15] + [-np.inf] * len(segments)
    upper_bounds = [150, 60, 2.5] + [ np.inf] * len(segments)

    try:
        # 5. RUN OPTIMIZATION
        res = least_squares(
            residuals_func, 
            x0, 
            bounds=(lower_bounds, upper_bounds),
            max_nfev=2000,
            method='trf'
        )
        
        # 6. EXTRACT RESULTS
        offset_opt, amp_opt, freq_opt = res.x[0], res.x[1], res.x[2]
        
        # Calculate R-squared
        final_residuals = res.fun
        ss_res = np.sum(final_residuals**2)
        ss_tot = np.sum((y_concat - np.mean(y_concat))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        cadence = freq_opt * 60
        
        # Coasting check
        if amp_opt < 5.0:
            print(f"[RESULT] Coasting detected (Amp={amp_opt:.1f}°).")
            return None

        robust_max = offset_opt + abs(amp_opt)
        robust_min = offset_opt - abs(amp_opt)
        
        print(f"[MULTI-FIT] Max: {robust_max:.1f}, Cadence: {cadence:.0f}, R2: {r_squared:.3f}")
        
        return {
            "max_extension": robust_max,
            "min_flexion": robust_min,
            "cadence_rpm": cadence,
            "r_squared": r_squared
        }

    except Exception as e:
        print(f"[ERROR] Multi-fit failed: {e}")
        return None


def fit_sine_wave_single_segment(timestamps, angles):
    """
    STRATEGY 1: SINGLE BEST SEGMENT
    
    Identifies the "Island" with the highest variance (most active pedaling)
    and fits a standard sine wave to it. Best for very noisy videos.
    """
    # 1. Clean Data
    t_all = np.array(timestamps)
    y_all = np.array(angles)
    valid = ~np.isnan(y_all)
    t = t_all[valid]
    y = y_all[valid]
    
    if len(y) < 20: return None

    # 2. Smart Segment Selection
    time_diffs = np.diff(t)
    gap_indices = np.where(time_diffs > 1.0)[0]
    split_indices = np.concatenate(([0], gap_indices + 1, [len(t)]))
    
    best_t = []
    best_y = []
    best_score = -1
    
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i+1]
        seg_t = t[start:end]
        seg_y = y[start:end]
        
        if len(seg_t) < 15: continue
            
        # Score = Duration * Variance
        score = len(seg_t) * np.std(seg_y)
        
        if score > best_score:
            best_score = score
            best_t = seg_t
            best_y = seg_y
            
    if len(best_t) < 20: return None
        
    t, y = best_t, best_y
    print(f"[DEBUG] Single-Fit Segment: {len(t)} pts, Duration: {t[-1]-t[0]:.1f}s")

    # 3. Model & Grid Search
    def sine_model(t, offset, amp, freq, phase):
        return offset + amp * np.sin(2 * np.pi * freq * t + phase)

    freq_guesses = [40/60, 70/60, 100/60, 130/60] 
    best_r2 = -np.inf
    best_params = None
    
    lower_bounds = [80,   0, 0.5, -np.inf]
    upper_bounds = [150, 60, 2.5,  np.inf]
    
    guess_offset = np.mean(y)
    guess_amp = (np.max(y) - np.min(y)) / 2
    
    for guess_freq in freq_guesses:
        try:
            p0 = [guess_offset, guess_amp, guess_freq, 0]
            params, _ = curve_fit(
                sine_model, t, y, p0=p0, 
                bounds=(lower_bounds, upper_bounds),
                method='trf', maxfev=2000 
            )
            residuals = y - sine_model(t, *params)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-6 else 0
            
            if r_squared > best_r2:
                best_r2, best_params = r_squared, params
        except: continue
            
    if best_params is None: return None
        
    offset, amp, freq, phase = best_params
    if amp < 5.0: return None

    return {
        "max_extension": offset + abs(amp),
        "min_flexion": offset - abs(amp),
        "cadence_rpm": freq * 60,
        "r_squared": best_r2
    }

from scipy.signal import savgol_filter, find_peaks

# Method for estimating maximum and minimum knee flexion and maximum and minimum hip angle. 
# Peak height limit and valley height limit params can be viewed as integrating prior knowledge to the angle prediction
# for example we can set the height limit to 120 degrees for knee max extension as we expect the knee max extension to be above 120 degrees
def estimate_pedal_stroke_stats(timestamps, angles, 
                            smooth_window_sec=0.33, 
                            min_interpeak_sec=0.35,
                            polyorder=2,
                            # Default limits (None means "no limit")
                            peak_height_limit=None,   
                            valley_height_limit=None,
                            prominence=None):
    """
    Generic function to analyze any rhythmic cycling signal (Knee or Hip).
    Finds Peaks (Max) and Valleys (Min).
    """
    y = np.array(angles)
    t = np.array(timestamps)
    
    # 1. Validation & Interpolation
    if np.sum(~np.isnan(y)) < 10: return None
    
    # Calculate FPS
    dt = np.median(np.diff(t))
    if dt <= 0 or np.isnan(dt): return None
    fps = 1.0 / dt
    
    # Fill NaNs
    nans = np.isnan(y)
    if np.any(nans):
        y[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), y[~nans])

    # 2. Smoothing
    window_length = int(smooth_window_sec * fps) | 1
    window_length = max(5, min(window_length, len(y)))
    if window_length >= len(y): 
        window_length = len(y) - 1 if len(y)%2==0 else len(y)
        
    try:
        y_smooth = savgol_filter(y, window_length, polyorder)
    except: return None

    distance = int(min_interpeak_sec * fps)

    # 3. Find Peaks (Max Values / Extension)
    # height: only count peaks ABOVE this value
    peaks, _ = find_peaks(y_smooth, height=peak_height_limit, distance=distance, prominence=prominence)
    
    # 4. Find Valleys (Min Values / Flexion) -> Invert signal
    # height: only count valleys BELOW this value (so -y is ABOVE -limit)
    inv_height = -valley_height_limit if valley_height_limit is not None else None
    valleys, _ = find_peaks(-y_smooth, height=inv_height, distance=distance, prominence=prominence)

    # 5. Calculate Stats
    stats = {
        "count_peaks": len(peaks),
        "count_valleys": len(valleys),
        "max_angle": np.nan,
        "min_angle": np.nan,
        "stability_std": 0.0,
        "cadence_rpm": 0
    }
    
    # Max (Extension)
    if len(peaks) > 0:
        stats["max_angle"] = np.median(y_smooth[peaks])
        stats["stability_std"] = np.std(y_smooth[peaks])
    elif len(y_smooth) > 0:
        stats["max_angle"] = np.nanpercentile(y_smooth, 95) # Fallback

    # Min (Flexion / Closed Hip)
    if len(valleys) > 0:
        stats["min_angle"] = np.median(y_smooth[valleys])
    elif len(y_smooth) > 0:
        stats["min_angle"] = np.nanpercentile(y_smooth, 5) # Fallback

    # Cadence (from peaks)
    if len(peaks) > 1:
        avg_diff = np.mean(np.diff(t[peaks]))
        if avg_diff > 0:
            stats["cadence_rpm"] = 60.0 / avg_diff
            
    return stats