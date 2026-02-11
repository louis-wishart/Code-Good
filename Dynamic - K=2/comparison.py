import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import re


INPUT_RATIO = "pysindy_ratio_2025.csv"
MARKOV_WD = "training_weekday.csv"
MARKOV_WE = "training_weekend.csv"
OUTPUT_PLOT = "comparison.png"

AGENTS = 1000    
CLUSTERS = 2


# Load Markov 
P_wd = pd.read_csv(MARKOV_WD, index_col=0).values
P_we = pd.read_csv(MARKOV_WE, index_col=0).values

# PySindy Coefficients
def extract_coeffs(text):
    clean = text.replace(" ", "")
    matches = re.findall(r"[-+]?\d*\.\d+", clean)
    return [float(n) for n in matches] if len(matches) >= 3 else [0.0, 0.0, 0.0]

COEFFS_WD = extract_coeffs(input("WEEKDAY Equation: "))
COEFFS_WE = extract_coeffs(input("WEEKEND Equation: "))

# 2025 Data 
df = pd.read_csv(INPUT_RATIO)
df['date'] = pd.to_datetime(df['date'])
dates = df.sort_values('date')['date']
real_vals = df['high_user_ratio_smooth'].values

# Markov Simulation
start_dist = [1 - real_vals[0], real_vals[0]]
current_states = np.random.choice(range(CLUSTERS), size=AGENTS, p=start_dist)
markov_results = [real_vals[0]]

for i in range(1, len(dates)):
    is_weekend = dates.iloc[i-1].dayofweek >= 5
    P_curr = P_we if is_weekend else P_wd
    
    for state in range(CLUSTERS):
        mask = (current_states == state)
        if np.sum(mask) > 0:
            probs = P_curr[state] / P_curr[state].sum()
            current_states[mask] = np.random.choice(CLUSTERS, size=np.sum(mask), p=probs)
            
    markov_results.append(np.mean(current_states == 1))


# PySindy Simulation
print("Running PySINDy Physics...")
def dxdt(x, t, c, a, b): return c + a*x + b*(x**2)

pysindy_results = [real_vals[0]]
curr_val = real_vals[0]

for i in range(1, len(dates)):
    is_weekend = dates.iloc[i-1].dayofweek >= 5
    coeffs = COEFFS_WE if is_weekend else COEFFS_WD
    
    # Solve ODE for 1 day
    pred = odeint(dxdt, curr_val, [0, 1], args=tuple(coeffs))
    curr_val = max(0, min(1, pred[-1][0])) 
    pysindy_results.append(curr_val)


# Results
markov_arr = np.array(markov_results)
pysindy_arr = np.array(pysindy_results)

rmse_m = np.sqrt(np.mean((real_vals - markov_arr)**2))
rmse_p = np.sqrt(np.mean((real_vals - pysindy_arr)**2))

print(f"\nRESULTS:\nMarkov RMSE:  {rmse_m:.4f}\nPySINDy RMSE: {rmse_p:.4f}")
print("-> WINNER: Markov Chain" if rmse_m < rmse_p else "-> WINNER: PySINDy")

# Plot 
plt.figure(figsize=(10, 6))
plt.plot(dates, real_vals, 'k', alpha=0.3, lw=3, label='Reality (2025)')
plt.plot(dates, markov_arr, 'orange', alpha=0.9, label=f'Markov Sim (RMSE {rmse_m:.3f})')
plt.plot(dates, pysindy_arr, 'g--', lw=2, label=f'PySindy Sim (RMSE {rmse_p:.3f})')
plt.title("Final Showdown: Simulation vs Reality")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_PLOT)
print(f"Saved plot: {OUTPUT_PLOT}")