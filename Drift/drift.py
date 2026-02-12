import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# --- CONFIGURATION ---
INPUT_RATIO = "pysindy_ratio_2025.csv"
INPUT_TIMELINE = "clusters_2025_k2.parquet"

# Markov Matrices (2024 Training)
MAT_WIN_WD = "matrix_winter_wd_2024.csv"
MAT_WIN_WE = "matrix_winter_we_2024.csv"
MAT_SUM_WD = "matrix_summer_wd_2024.csv"
MAT_SUM_WE = "matrix_summer_we_2024.csv"

# PySINDy Equations (Hardcoded from 2024 Training)
# Format: [c, a, b] for c + ax + bx^2
# Weekday: (x)' = 0.008 - 0.042x + 0.014x^2 (Stable)
COEFFS_WIN_WD = [-0.016, 0.138, -0.223] 
COEFFS_WIN_WE = [-0.076, 0.409, -0.414] 
COEFFS_SUM_WD = [0.005, 0.030, -0.238] 
COEFFS_SUM_WE = [0.011, -0.188, 0.647]

OUTPUT_PLOT = "final_comparison_adaptive.png"
SIMULATION_AGENTS = 5000 
CALIBRATION_DAYS = 90  # Learn drift from first 3 months (March-May)

# --- HELPER: Seasonal Blending (Scottish Context) ---
def get_seasonal_weight(date):
    m = date.month
    d = date.day
    # Winter: Nov, Dec, Jan, Feb, Mar (Cold)
    if m in [11, 12, 1, 2, 3]: return 1.0
    # Summer: May, Jun, Jul, Aug (Warm)
    if m in [5, 6, 7, 8]: return 0.0
    # Transitions
    if m == 4: return 1.0 - (d / 30.0) # April: Winter -> Summer
    if m == 9: return (d / 30.0)       # Sept: Summer -> Winter
    if m == 10: return 1.0             # Oct is Winter in Scotland
    return 0.0

# --- HELPER: PySINDy Math ---
def pysindy_step(x, t, coeffs):
    c, a, b = coeffs
    return c + a*x + b*(x**2)

# --- HELPER: Simulation Engine ---
def run_simulation(df_real):
    print(f"--- Running Adaptive Simulation (Calibration: {CALIBRATION_DAYS} days) ---")
    
    # 1. Setup
    dates = df_real['date']
    real_vals = df_real['high_user_ratio_smooth'].values
    n_days = len(dates)
    
    # Load Matrices
    P_win_wd = pd.read_csv(MAT_WIN_WD, index_col=0).values
    P_win_we = pd.read_csv(MAT_WIN_WE, index_col=0).values
    P_sum_wd = pd.read_csv(MAT_SUM_WD, index_col=0).values
    P_sum_we = pd.read_csv(MAT_SUM_WE, index_col=0).values
    
    # Initialize State
    # Markov Agents
    start_ratio = real_vals[0]
    start_dist = np.array([1-start_ratio, start_ratio])
    agents = np.zeros(SIMULATION_AGENTS, dtype=int)
    agents[:int(SIMULATION_AGENTS * start_ratio)] = 1
    np.random.shuffle(agents)
    
    # PySINDy State
    x_phys = start_ratio
    
    # History
    hist_markov = [np.mean(agents)]
    hist_pysindy = [x_phys]
    
    # Drift Parameters
    drift_markov = 0.0
    drift_pysindy = 0.0
    
    # 2. Main Loop
    for t in range(1, n_days):
        date = dates.iloc[t-1] # Today's date determines physics for tomorrow
        is_weekend = date.dayofweek >= 5
        w_weight = get_seasonal_weight(date)
        
        # --- CALIBRATION STEP (Day 90) ---
        if t == CALIBRATION_DAYS:
            print(f"[Day {t}] Calibrating Drift...")
            # Calculate average error over the calibration period
            # Markov Drift
            error_m = real_vals[t] - hist_markov[-1]
            drift_markov = error_m / CALIBRATION_DAYS
            
            # PySINDy Drift
            error_p = real_vals[t] - hist_pysindy[-1]
            drift_pysindy = error_p / CALIBRATION_DAYS
            
            print(f"-> Markov Drift Rate: {drift_markov:.6f} / day")
            print(f"-> PySINDy Drift Rate: {drift_pysindy:.6f} / day")

        # --- MARKOV UPDATE ---
        # 1. Blend Matrices
        if is_weekend:
            P_base = P_win_we * w_weight + P_sum_we * (1 - w_weight)
        else:
            P_base = P_win_wd * w_weight + P_sum_wd * (1 - w_weight)
            
        # 2. Extract Probabilities
        # P[0,1] is Prey->Predator
        p_up = P_base[0, 1] / P_base[0].sum()
        p_down = P_base[1, 0] / P_base[1].sum()
        
        # 3. Apply Drift (If calibrated)
        if t > CALIBRATION_DAYS:
            # Add drift to the "Upward" pressure
            p_up += (drift_markov * 5.0) # Scaling factor for probability sensitivity
            if p_up < 0: p_up = 0
            if p_up > 1: p_up = 1
            
        # 4. Roll Dice
        rolls = np.random.random(SIMULATION_AGENTS)
        # Prey becomes Predator
        mask_up = (agents == 0) & (rolls < p_up)
        agents[mask_up] = 1
        # Predator becomes Prey
        mask_down = (agents == 1) & (rolls < p_down)
        agents[mask_down] = 0
        
        hist_markov.append(np.mean(agents))
        
        # --- PYSINDY UPDATE ---
        # 1. Select Coefficients
        if is_weekend:
            c1, c2 = COEFFS_WIN_WE, COEFFS_SUM_WE
        else:
            c1, c2 = COEFFS_WIN_WD, COEFFS_SUM_WD
            
        # 2. Blend Coefficients
        # c = [c0, a, b]
        coeffs = [
            c1[0]*w_weight + c2[0]*(1-w_weight),
            c1[1]*w_weight + c2[1]*(1-w_weight),
            c1[2]*w_weight + c2[2]*(1-w_weight)
        ]
        
        # 3. Integrate (ODEINT)
        t_span = [0, 1]
        res = odeint(pysindy_step, x_phys, t_span, args=(coeffs,))
        x_new = res[-1][0]
        
        # 4. Apply Drift (Additive to state)
        if t > CALIBRATION_DAYS:
            x_new += (drift_pysindy * (t - CALIBRATION_DAYS))
            
        x_new = max(0, min(1, x_new))
        x_phys = x_new
        hist_pysindy.append(x_phys)
        
    return np.array(hist_markov), np.array(hist_pysindy)

# --- EXECUTION ---
def run_final_revised():
    if not os.path.exists(INPUT_RATIO):
        print("Error: Input data not found.")
        return
        
    df = pd.read_csv(INPUT_RATIO)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    real_vals = df['high_user_ratio_smooth'].values
    
    # Run
    sim_m, sim_p = run_simulation(df)
    
    # Metrics
    rmse_m = np.sqrt(np.mean((real_vals - sim_m)**2))
    rmse_p = np.sqrt(np.mean((real_vals - sim_p)**2))
    
    print(f"\n[RESULTS]")
    print(f"Markov RMSE:  {rmse_m:.4f}")
    print(f"PySINDy RMSE: {rmse_p:.4f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], real_vals, color='black', alpha=0.4, linewidth=3, label='Real 2025')
    plt.plot(df['date'], sim_m, color='#ff7f0e', label=f'Markov (Adaptive) RMSE={rmse_m:.2f}')
    plt.plot(df['date'], sim_p, color='#2ca02c', linestyle='--', label=f'PySINDy (Adaptive) RMSE={rmse_p:.2f}')
    
    plt.axvline(df['date'].iloc[CALIBRATION_DAYS], color='red', linestyle=':', label='Calibration End')
    
    plt.title("Final Comparison: Adaptive Models (Calibrated Drift)")
    plt.ylabel("High User Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT)
    print(f"Saved {OUTPUT_PLOT}")

if __name__ == "__main__":
    run_final_revised()