import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os


FILE_INPUT_RATIO = "pysindy_ratio_2025.csv"
FILE_CENTROIDS   = "centroids_k2.npy"

# Markov 
FILE_WIN_WD = "winter_weekday.csv"
FILE_WIN_WE = "winter_weekend.csv"
FILE_SUM_WD = "summer_weekday.csv"
FILE_SUM_WE = "summer_weekend.csv"

# PySINDy Context Equations (Discovered during training)
# form: dx/dt = c + ax + bx^2
COEFFS_WIN_WD = [0.00886, 0.00496, -0.05857]
COEFFS_WIN_WE = [-0.09451, 0.47853, -0.50894]
COEFFS_SUM_WD = [0.00213, 0.04005, -0.22515]
COEFFS_SUM_WE = [-0.00573, 0.08363, -0.18024]

# Drift factors 
DRIFT_ANNUAL = 0.1496
DRIFT_DAILY  = DRIFT_ANNUAL / 365.0

N_FEEDERS = 200
N_AGENTS  = 5000  
SAFETY_MARGIN = 1.10


# Ratio > kW , Find Peak
centroids = np.load(FILE_CENTROIDS)

profile_low_kw  = (centroids[0] * 2) / 1000
profile_high_kw = (centroids[1] * 2) / 1000

peak_low  = np.max(profile_low_kw)
peak_high = np.max(profile_high_kw)

print(f"Prey Peak (Base):     {peak_low:.2f} kW")
print(f"Predator Peak (Base): {peak_high:.2f} kW")


# Substation Capacity
design_max = 5000
pf = 0.95
limit = 0.85
SUBSTATION_LIMIT = (design_max * pf) * limit
print(f"Substation Limit:     {SUBSTATION_LIMIT:.1f} kW\n")


# Calculate System Demand 
def get_total_system_kw(ratio_high_users, day_index=0):
    
    growth_multiplier = 1.0 + (DRIFT_DAILY * day_index)
    
    current_peak_low = peak_low * growth_multiplier
    current_peak_high = peak_high * growth_multiplier
    
    weighted_peak = (1 - ratio_high_users) * current_peak_low + (ratio_high_users * current_peak_high)
    return weighted_peak * N_FEEDERS

# Seasonal Weighting 
def get_seasonal_factor(date):
    m = date.month
    d = date.day
    if m in [10, 11, 12, 1, 2, 3]: return 1.0 # Winter
    if m in [5, 6, 7, 8]:          return 0.0 # Summer
    if m == 4: return 1.0 - (d / 30.0)        # April Blend
    if m == 9: return (d / 30.0)              # Sept Blend
    return 0.0


# Stability Analysis (Worst Case - Winter Weekend)
c, a, b = COEFFS_WIN_WE
roots = np.roots([b, a, c])

valid_roots = [r for r in roots if np.isreal(r) and r > 0]
if valid_roots:
    stable_ratio = np.max(valid_roots)
    print("STABILITY ANALYSIS (Winter Weekend):")
    print(f"Stability Point: {stable_ratio:.2%} Predators")
    
    for label, day_t in [("Current (Day 0)", 0), ("End Year 1 (Day 365)", 365), ("End Year 2 (Day 730)", 730)]:
        stable_kw = get_total_system_kw(stable_ratio, day_t)
        warning = " (UNSTABLE - BREACH)" if stable_kw > SUBSTATION_LIMIT else " (SAFE)"
        print(f"{label:<22}: {stable_kw:,.0f} kW{warning}")
else:
    print("RUNAWAY GROWTH (No stable limit)")
    stable_ratio = 1.0 


### Simulation 

# Load starting data
df_input = pd.read_csv(FILE_INPUT_RATIO)
start_ratio = df_input['high_user_ratio_smooth'].iloc[-1]

# Data Range (2 Years)
dates = pd.date_range(start='2026-01-01', periods=730)
days_count = len(dates)

## Markov Sim Matrices
M_win_wd = pd.read_csv(FILE_WIN_WD, index_col=0).values
M_win_we = pd.read_csv(FILE_WIN_WE, index_col=0).values
M_sum_wd = pd.read_csv(FILE_SUM_WD, index_col=0).values
M_sum_we = pd.read_csv(FILE_SUM_WE, index_col=0).values

# Sim initialization
current_agents = np.zeros(N_AGENTS)
num_high = int(start_ratio * N_AGENTS)
current_agents[:num_high] = 1 
np.random.shuffle(current_agents)

ratio_markov = []
ratio_pysindy = []
current_pysindy_x = start_ratio

# Daily Loop (Both Models Running Synchronously)
for i in range(days_count):
    date = dates[i]
    is_weekend = (date.dayofweek >= 5)
    w_factor = get_seasonal_factor(date) 
    
    # PySindy Step
    if is_weekend:
        dx_win = COEFFS_WIN_WE[0] + COEFFS_WIN_WE[1]*current_pysindy_x + COEFFS_WIN_WE[2]*(current_pysindy_x**2)
        dx_sum = COEFFS_SUM_WE[0] + COEFFS_SUM_WE[1]*current_pysindy_x + COEFFS_SUM_WE[2]*(current_pysindy_x**2)
    else:
        dx_win = COEFFS_WIN_WD[0] + COEFFS_WIN_WD[1]*current_pysindy_x + COEFFS_WIN_WD[2]*(current_pysindy_x**2)
        dx_sum = COEFFS_SUM_WD[0] + COEFFS_SUM_WD[1]*current_pysindy_x + COEFFS_SUM_WD[2]*(current_pysindy_x**2)
    
    # Blend the rate of change
    dx_blended = dx_win * w_factor + dx_sum * (1 - w_factor)
    
    # Step forward & clamp between 0% and 100%
    current_pysindy_x = max(0.0, min(1.0, current_pysindy_x + dx_blended))
    ratio_pysindy.append(current_pysindy_x)

    # Markov Step
    if is_weekend:
        M_today = M_win_we * w_factor + M_sum_we * (1 - w_factor)
    else:
        M_today = M_win_wd * w_factor + M_sum_wd * (1 - w_factor)

    p_0_to_1 = M_today[0, 1] / (M_today[0, 0] + M_today[0, 1])
    p_1_to_0 = M_today[1, 0] / (M_today[1, 0] + M_today[1, 1])
    
    rolls = np.random.random(N_AGENTS)
    switching_up = (current_agents == 0) & (rolls < p_0_to_1)
    switching_down = (current_agents == 1) & (rolls < p_1_to_0)
    
    current_agents[switching_up] = 1
    current_agents[switching_down] = 0
    ratio_markov.append(np.mean(current_agents))


# Ratio > kW, add drift
kw_pysindy = [get_total_system_kw(r, i) for i, r in enumerate(ratio_pysindy)]
kw_markov  = [get_total_system_kw(r, i) for i, r in enumerate(ratio_markov)]

## Plot
plt.figure(figsize=(12, 6))

# Capacity Line
plt.axhline(y=SUBSTATION_LIMIT, color='k', linestyle='--', linewidth=2, 
            label=f'Rated Capacity ({SUBSTATION_LIMIT:.0f} kW)')

# Models
plt.plot(dates, kw_pysindy, 'g-', linewidth=2, label='PySindy Model')
plt.plot(dates, kw_markov,  'orange', alpha=0.6, label='Markov Model')

# Print Peak Values
print("\nPEAK DEMAND:")
print(f"Year 1 (2026) PySINDy Peak: {np.max(kw_pysindy[:365]):,.0f} kW")
print(f"Year 1 (2026) Markov Peak:  {np.max(kw_markov[:365]):,.0f} kW")
print(f"Year 2 (2027) PySINDy Peak: {np.max(kw_pysindy[365:]):,.0f} kW")
print(f"Year 2 (2027) Markov Peak:  {np.max(kw_markov[365:]):,.0f} kW")

# Find Breach Points
arr_pysindy = np.array(kw_pysindy)
arr_markov = np.array(kw_markov)

breach_indices = np.where(arr_markov > SUBSTATION_LIMIT)[0]

if len(breach_indices) > 0:
    first_fail_idx = breach_indices[0]
    fail_date = dates[first_fail_idx]
    print(f"\nForecast: System failure predicted on {fail_date.date()}")
    
    plt.scatter([fail_date], [SUBSTATION_LIMIT], color='red', s=100, zorder=5)
    plt.annotate(f'WINTER BREACH: {fail_date.date()}', 
                 xy=(fail_date, SUBSTATION_LIMIT), 
                 xytext=(fail_date, SUBSTATION_LIMIT - 300),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontweight='bold', color='darkred')
else:
    print("\nForecast: Safe through 2027")




plt.title("Grid Load Forecast (2026-2028)\n(Including 14.96% Drift)")
plt.ylabel("System Peak Demand (kW)")
plt.xlabel("Date")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.savefig("forecast_drift.png")