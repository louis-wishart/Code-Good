import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# --- CONFIGURATION ---
INPUT_TIMELINE = "clusters_2025_k2.parquet"
INPUT_RATIO = "pysindy_ratio_2025.csv"

# Input Matrices (Markov)
MAT_WIN_WD = "training_winter_wd.csv"
MAT_WIN_WE = "training_winter_we.csv"
MAT_SUM_WD = "training_summer_wd.csv"
MAT_SUM_WE = "training_summer_we.csv"

# Input Equations (PySINDy) - Hardcoded from your training output
# Format: [c, a, b] for c + ax + bx^2
COEFFS_WIN_WD = [-0.01623, 0.13851, -0.22358]
COEFFS_WIN_WE = [-0.07634, 0.40994, -0.41411]
COEFFS_SUM_WD = [0.00573, 0.03052, -0.23892]
COEFFS_SUM_WE = [0.01163, -0.18893, 0.64796]

OUTPUT_PLOT = "final_seasonal_comparison.png"
SIMULATION_AGENTS = 5000 
K_CLUSTERS = 2

# Season Definitions
WINTER_MONTHS = [11, 12, 1, 2, 3]

# --- HELPER: Blending Logic ---
def get_seasonal_weight(date):
    """
    Returns 'winter_weight' (0.0 to 1.0).
    1.0 = Pure Winter, 0.0 = Pure Summer.
    Blends during April (Winter->Summer) and October (Summer->Winter).
    """
    m = date.month
    d = date.day
    
    if m in [11, 12, 1, 2, 3]: return 1.0 # Pure Winter
    if m in [5, 6, 7, 8, 9]: return 0.0   # Pure Summer
    
    # Transition Months (Linear Interpolation)
    if m == 4: # April: Winter -> Summer
        return 1.0 - (d / 30.0)
    if m == 10: # Oct: Summer -> Winter
        return (d / 31.0)
    return 0.0

# --- HELPER: Markov Simulation ---
def simulate_markov(start_date, n_days, mats, start_dist):
    history = np.zeros((n_days, SIMULATION_AGENTS), dtype=int)
    # Start distribution
    start_dist = start_dist / start_dist.sum()
    history[0] = np.random.choice(range(K_CLUSTERS), size=SIMULATION_AGENTS, p=start_dist)
    
    dates = pd.date_range(start=start_date, periods=n_days)
    
    # Pre-load matrices
    P_win_wd = pd.read_csv(mats['win_wd'], index_col=0).values
    P_win_we = pd.read_csv(mats['win_we'], index_col=0).values
    P_sum_wd = pd.read_csv(mats['sum_wd'], index_col=0).values
    P_sum_we = pd.read_csv(mats['sum_we'], index_col=0).values
    
    print(f"Running Seasonal Markov Simulation...")
    
    for t in range(1, n_days):
        current_date = dates[t-1]
        is_weekend = current_date.dayofweek >= 5
        w_weight = get_seasonal_weight(current_date)
        
        # 1. Select Base Matrices
        if is_weekend:
            P_target = P_win_we * w_weight + P_sum_we * (1 - w_weight)
        else:
            P_target = P_win_wd * w_weight + P_sum_wd * (1 - w_weight)
            
        # 2. Update Agents
        current_states = history[t-1]
        for state in range(K_CLUSTERS):
            mask = (current_states == state)
            count = np.sum(mask)
            if count > 0:
                probs = P_target[state] / P_target[state].sum()
                history[t, mask] = np.random.choice(range(K_CLUSTERS), size=count, p=probs)
    
    return np.mean(history == 1, axis=1)

# --- HELPER: PySINDy Simulation ---
def pysindy_step(x, coeffs):
    c, a, b = coeffs
    # Euler step for simplicity in blended context
    dxdt = c + a*x + b*(x**2)
    return x + dxdt # dt=1 day

def simulate_pysindy(start_val, dates):
    print("Running Seasonal PySINDy Simulation...")
    predictions = [start_val]
    x_current = start_val
    
    for i in range(1, len(dates)):
        current_date = dates[i-1]
        is_weekend = current_date.dayofweek >= 5
        w_weight = get_seasonal_weight(current_date)
        
        # 1. Calculate derivatives for both seasons
        if is_weekend:
            dx_win = model_func(x_current, COEFFS_WIN_WE)
            dx_sum = model_func(x_current, COEFFS_SUM_WE)
        else:
            dx_win = model_func(x_current, COEFFS_WIN_WD)
            dx_sum = model_func(x_current, COEFFS_SUM_WD)
            
        # 2. Blend Derivatives
        dx_final = dx_win * w_weight + dx_sum * (1 - w_weight)
        
        # 3. Update
        x_new = max(0, min(1, x_current + dx_final))
        predictions.append(x_new)
        x_current = x_new
        
    return predictions

def model_func(x, coeffs):
    c, a, b = coeffs
    return c + a*x + b*(x**2)

# --- MAIN EXECUTION ---
def run_final_seasonal():
    print("--- PHASE 4: SEASONAL MODEL SHOWDOWN ---")
    
    if not os.path.exists(INPUT_RATIO):
        print(f"Error: {INPUT_RATIO} missing.")
        return

    # Load Data
    df_real = pd.read_csv(INPUT_RATIO)
    df_real['date'] = pd.to_datetime(df_real['date'])
    df_real = df_real.sort_values('date')
    
    real_dates = df_real['date']
    real_values = df_real['high_user_ratio_smooth'].values 
    
    # Run Markov
    mats = {
        'win_wd': MAT_WIN_WD, 'win_we': MAT_WIN_WE,
        'sum_wd': MAT_SUM_WD, 'sum_we': MAT_SUM_WE
    }
    # Check matrices exist
    for m in mats.values():
        if not os.path.exists(m):
            print(f"Error: {m} missing. Run training first.")
            return

    start_ratio = real_values[0]
    start_dist = np.array([1-start_ratio, start_ratio])
    
    markov_values = simulate_markov(real_dates.iloc[0], len(real_dates), mats, start_dist)
    
    # Run PySINDy
    pysindy_values = simulate_pysindy(real_values[0], real_dates)
    
    # Validation
    rmse_markov = np.sqrt(np.mean((real_values - markov_values)**2))
    rmse_pysindy = np.sqrt(np.mean((real_values - pysindy_values)**2))
    
    print(f"\n[RESULTS]")
    print(f"Markov Error (RMSE):  {rmse_markov:.4f}")
    print(f"PySINDy Error (RMSE): {rmse_pysindy:.4f}")
    
    # Drift Check (End of Year)
    print(f"\n[DRIFT CHECK - JAN 2026]")
    print(f"Real:    {real_values[-1]:.3f}")
    print(f"Markov:  {markov_values[-1]:.3f}")
    print(f"PySINDy: {pysindy_values[-1]:.3f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(real_dates, real_values, 'k-', alpha=0.3, linewidth=3, label='Real 2025')
    plt.plot(real_dates, markov_values, color='#ff7f0e', linewidth=2, label=f'Markov (Seasonal) RMSE={rmse_markov:.2f}')
    plt.plot(real_dates, pysindy_values, color='#2ca02c', linewidth=2, linestyle='--', label=f'PySINDy (Seasonal) RMSE={rmse_pysindy:.2f}')
    
    plt.title("Final Seasonal Comparison: Did we capture the Winter Peak?")
    plt.ylabel("Predator Population Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT)
    print(f"Saved {OUTPUT_PLOT}")

if __name__ == "__main__":
    run_final_seasonal()