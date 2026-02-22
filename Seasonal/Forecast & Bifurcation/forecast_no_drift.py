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

# Winter Weekend Coeff
# form: c + ax + bx^2
ODE_COEFFS = [-0.09451, 0.47853, -0.50894] 

N_FEEDERS = 200
N_AGENTS  = 5000  



# Ratio > kW , Find Peak
centroids = np.load(FILE_CENTROIDS)

profile_low_kw  = (centroids[0] * 2) / 1000
profile_high_kw = (centroids[1] * 2) / 1000

peak_low  = np.max(profile_low_kw)
peak_high = np.max(profile_high_kw)

print(f"Prey Peak:  {peak_low:.2f} kW")
print(f"Predator Peak: {peak_high:.2f} kW")


# Substation Capacity
design_max = 5000
pf = 0.95
limit = 0.85
SUBSTATION_LIMIT = (design_max * pf) * limit
print(f"Substation Limit: {SUBSTATION_LIMIT:.1f} kW")


# Calculate System Demand
def get_total_system_kw(ratio_high_users):
    
    weighted_peak = (1 - ratio_high_users) * peak_low + (ratio_high_users * peak_high)
    return weighted_peak * N_FEEDERS

# Seasonal Weightng 
def get_seasonal_factor(date):
    
    m = date.month
    d = date.day
    
    if m in [10, 11, 12, 1, 2, 3]: return 1.0 # Winter
    if m in [5, 6, 7, 8]:          return 0.0 # Summer
    if m == 4: return 1.0 - (d / 30.0)        # April Blend
    if m == 9: return (d / 30.0)              # Sept Blend
    return 0.0

# ODE Function
def model_physics_ode(y, t, c, a, b):
    dydt = (c + a*y + b*(y**2))
    return dydt


# Stability Analysis (Bifurcation)
c, a, b = ODE_COEFFS
roots = np.roots([b, a, c ])

stable_ratio = np.max(roots)
stable_kw = get_total_system_kw(stable_ratio)


print(f"\nTheoretical Stable Ratio: {stable_ratio:.2%}")
print(f"Stability Demand:       {stable_kw:.0f} kW")

if stable_kw > SUBSTATION_LIMIT:
    print(f"=> UNSTABLE (Stability exceeds capacity by {stable_kw - SUBSTATION_LIMIT:.0f} kW)")
else:
    print("=> STABLE (Stability within capacity)")


### Simulation 

# Load starting data
df_input = pd.read_csv(FILE_INPUT_RATIO)
start_ratio = df_input['high_user_ratio_smooth'].iloc[-1]

# Data Range (2 Years)
dates = pd.date_range(start='2026-01-01', periods=730)
days_count = len(dates)
t_span = np.linspace(0, days_count, days_count)

## PySindy Sim
result_ode = odeint(model_physics_ode, start_ratio, t_span, args=tuple(ODE_COEFFS))
ratio_pysindy = np.clip(result_ode.flatten(), 0, 1)

# Stability
stability_threshold = stable_ratio * 0.99
stable_indices = np.where(ratio_pysindy >= stability_threshold)[0]
if len(stable_indices) > 0:
    date_stable = dates[stable_indices[0]]
    print(f"Forecast: Grid reaches stability on {date_stable.date()}")
else:
    print("Forecast: Grid does not reach stability within forecast window")


## Markov Sim
M_win_wd = pd.read_csv(FILE_WIN_WD, index_col=0).values
M_win_we = pd.read_csv(FILE_WIN_WE, index_col=0).values
M_sum_wd = pd.read_csv(FILE_SUM_WD, index_col=0).values
M_sum_we = pd.read_csv(FILE_SUM_WE, index_col=0).values

# Create Sim Feeders
current_agents = np.zeros(N_AGENTS)
# Allign sim to start ratio 
num_high = int(start_ratio * N_AGENTS)
current_agents[:num_high] = 1 
np.random.shuffle(current_agents)
ratio_markov = []

# Daily Loop 
for i in range(days_count):
    date = dates[i]
    is_weekend = (date.dayofweek >= 5)
    w_factor = get_seasonal_factor(date) # 1=Winter, 0=Summer
    
    if is_weekend:
        M_today = M_win_we * w_factor + M_sum_we * (1 - w_factor)
    else:
        M_today = M_win_wd * w_factor + M_sum_wd * (1 - w_factor)

    p_0_to_1 = M_today[0, 1] / (M_today[0, 0] + M_today[0, 1])
    p_1_to_0 = M_today[1, 0] / (M_today[1, 0] + M_today[1, 1])
    
    # Ensure 0-1 probability 
    p_0_to_1 = max(0.0, min(1.0, p_0_to_1))
    
    rolls = np.random.random(N_AGENTS)
    
    # Identify switches
    switching_up = (current_agents == 0) & (rolls < p_0_to_1)
    switching_down = (current_agents == 1) & (rolls < p_1_to_0)
    # Apply switches
    current_agents[switching_up] = 1
    current_agents[switching_down] = 0
    # Record average
    ratio_markov.append(np.mean(current_agents))


# Ratio > kW
kw_pysindy = [get_total_system_kw(r) for r in ratio_pysindy]
kw_markov  = [get_total_system_kw(r) for r in ratio_markov]

## Plot
plt.figure(figsize=(12, 6))

# Capacity Line
plt.axhline(y=SUBSTATION_LIMIT, color='k', linestyle='--', linewidth=2, 
            label=f'Rated Capacity ({SUBSTATION_LIMIT:.0f} kW)')

# Models
plt.plot(dates, kw_pysindy, 'g-', linewidth=2, label='PySINDy (Equation)')
plt.plot(dates, kw_markov,  'orange', alpha=0.6, label='Markov (Simulation)')

# Find Breach Points
arr_pysindy = np.array(kw_pysindy)
arr_markov = np.array(kw_markov)
breach_indices = np.where((arr_pysindy > SUBSTATION_LIMIT) & (arr_markov > SUBSTATION_LIMIT))[0]

if len(breach_indices) > 0:
    first_fail_idx = breach_indices[0]
    fail_date = dates[first_fail_idx]
    print(f"System failure predicted on {fail_date.date()}")
    
    plt.scatter([fail_date], [SUBSTATION_LIMIT], color='red', s=100, zorder=5)
    plt.annotate(f'FAILURE: {fail_date.date()}', 
                 xy=(fail_date, SUBSTATION_LIMIT), 
                 xytext=(fail_date, SUBSTATION_LIMIT - 300),
                 arrowprops=dict(facecolor='red', shrink=0.05))


plt.title("Grid Load Forecast (2026-2027)")
plt.ylabel("System Peak Demand (kW)")
plt.xlabel("Date")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("forecast_no_drift.png")