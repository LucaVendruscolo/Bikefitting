import numpy as np
from scipy.optimize import curve_fit

def fit_sine_wave(timestamps, angles):
    """
    Fits a sine wave to the angle data to find the true Maximum Extension.
    
    Args:
        timestamps (np.array): Time in seconds for each frame
        angles (np.array): Noisy angle data in degrees
        
    Returns:
        dict: {
            "max_extension": float, # The robust maximum angle (Saddle Height metric)
            "min_flexion": float,   # The robust minimum angle (top of stroke)
            "cadence_rpm": float,   # Estimated cadence
            "r_squared": float      # Goodness of fit (0 to 1)
        }
    """
    # 1. Clean Data: Remove None/NaN values
    valid = ~np.isnan(angles)
    t = np.array(timestamps)[valid]
    y = np.array(angles)[valid]

    # --- DEBUG: SAVE DATA TO CSV ---
    import csv
    with open("debug_curve_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "KneeAngle"])
        for ti, yi in zip(t, y):
            writer.writerow([ti, yi])
    print(f"[DEBUG] Saved {len(t)} points to debug_curve_data.csv")
    # -------------------------------
    
    if len(y) < 30: # Need enough frames for a valid fit
        return None
    
    # Outcome even worse, longest segment may not represent cycling motion well
    # # FIND LONGEST CONTINUOUS SEGMENT WITHOUT BIG GAPS
    # # Calculate time gaps between frames
    # time_diffs = np.diff(t)
    
    # # Identify where the gap is larger than 1.0 second (indicating a turn or stop)
    # gap_indices = np.where(time_diffs > 1.0)[0]
    
    # # Create split points: [0, gap1+1, gap2+1, ..., end]
    # split_indices = np.concatenate(([0], gap_indices + 1, [len(t)]))
    
    # best_t = []
    # best_y = []
    # max_points = 0
    
    # # Iterate through all segments to find the biggest one
    # for i in range(len(split_indices) - 1):
    #     start = split_indices[i]
    #     end = split_indices[i+1]
        
    #     segment_len = end - start
    #     if segment_len > max_points:
    #         max_points = segment_len
    #         best_t = t[start:end]
    #         best_y = y[start:end]
    
    # # Update t and y to be ONLY the best segment
    # t = best_t
    # y = best_y

    # 2. Define the model function
    def sine_model(t, offset, amp, freq, phase):
        return offset + amp * np.sin(2 * np.pi * freq * t + phase)
    
    # 4. GRID SEARCH
    # We try multiple initial guesses to ensure we don't get stuck.
    
    # Guesses: 40, 70, 100, 130 RPM
    freq_guesses = [40/60, 70/60, 100/60, 130/60]

    best_r2 = -np.inf
    best_params = None
    
    # --- PHYSICAL PRIORS (BOUNDS) ---
    # We constrain the solver to only look for realistic cycling values.
    
    # Offset (Mean Knee Angle): 
    #   Lower: 80° (Very bent)
    #   Upper: 150° (Very high saddle)
    # Amplitude (1/2 Range of Motion):
    #   Lower: 10° (Prevents flat line fit)
    #   Upper: 60° (Physically extreme)
    # Frequency (Cadence / 60):
    #   Lower: 0.5 Hz (30 RPM)
    #   Upper: 2.5 Hz (150 RPM)
    
    # Bounds Format: ([Low_Offset, Low_Amp, Low_Freq, Low_Phase], [High_...])
    lower_bounds = [80,  0, 0.5, -np.inf]
    upper_bounds = [150, 60, 2.5,  np.inf]

    # lower_bounds = [-np.inf, 0, 0.5, -np.inf]
    # upper_bounds = [np.inf, np.inf, 2.5, np.inf]

    # 3. Make Initial Guesses (Crucial for convergence)
    guess_amp = (np.max(y) - np.min(y)) / 2
    guess_offset = np.mean(y)
    
    for guess_freq in freq_guesses:
        try:
            p0 = [guess_offset, guess_amp, guess_freq, 0]
            
            params, _ = curve_fit(
                sine_model, t, y, p0=p0, 
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                maxfev=2000 # Faster per try
            )
            
            # Check R-squared for this specific guess
            residuals = y - sine_model(t, *params)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            
            if ss_tot < 1e-6: # Prevent divide by zero if data is perfectly flat
                r_squared = 0.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            # Keep the winner
            if r_squared > best_r2:
                best_r2 = r_squared
                best_params = params
                
        except:
            continue

    # 5. Final Verification
    if best_params is None:
        print("[ERROR] All fit attempts failed.")
        return None
        
    offset, amp, freq, phase = best_params
    cadence = freq * 60
    
    # COASTING CHECK:
    # If the best fit has tiny amplitude, the rider is not pedaling.
    if amp < 5.0:
        print(f"[RESULT] Coasting detected (Amp={amp:.1f}°). Ignoring.")
        return None

    if best_r2 < 0.4:
        print(f"[WARN] Poor fit (R2={best_r2:.2f}). Data might be noisy.")
        # We return it anyway, but the low R2 will flag it in the UI
        
    # Max extension is the peak of the fitted wave
    robust_max = offset + abs(amp)
    robust_min = offset - abs(amp)

    print(f"\n[FIT RESULTS] Max: {robust_max:.2f}, Min: {robust_min:.2f}, Cadence: {cadence:.2f} RPM, R²: {best_r2:.3f}")

    return {
        "max_extension": robust_max,
        "min_flexion": robust_min,
        "cadence_rpm": cadence,
        "r_squared": r_squared
    }