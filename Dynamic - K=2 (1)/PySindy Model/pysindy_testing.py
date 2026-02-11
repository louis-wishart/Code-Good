import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import re 


INPUT_FILE = "pysindy_ratio_2025.csv"
OUTPUT_PLOT = "testing_plot.png"

# Coefficients
def extract_coeffs(text_input):

    clean_text = text_input.replace(" ", "")
    matches = re.findall(r"[-+]?\d*\.\d+", clean_text)
    coeffs = [float(n) for n in matches]
    
    return coeffs

# User Input 
wd_str = input("WEEKDAY Equation: ")
COEFFS_WD = extract_coeffs(wd_str)
print(f"-> Extracted: {COEFFS_WD}")

we_str = input("\nWEEKEND Equation: ")
COEFFS_WE = extract_coeffs(we_str)
print(f"-> Extracted: {COEFFS_WE}")

# Load 2025 Data 
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Sim Start
real_start_val = df['high_user_ratio_smooth'].iloc[0]
print(f"Simulation Start: {df['date'].iloc[0].date()} (Value: {real_start_val:.4f})")


# Simulation 
def dxdt_func(x, t, coeffs):
    c, a, b = coeffs
    return c + a*x + b*(x**2)

sim_results = [real_start_val]
current_val = real_start_val

for i in range(1, len(df)):
    
    date_yesterday = df['date'].iloc[i-1]
    is_weekend = date_yesterday.dayofweek >= 5
    
    if is_weekend:
        active_coeffs = COEFFS_WE
    else:
        active_coeffs = COEFFS_WD
        
    t_span = [0, 1]
    prediction = odeint(dxdt_func, current_val, t_span, args=(active_coeffs,))
    
    new_val = prediction[-1][0]
    
    if new_val < 0: new_val = 0
    if new_val > 1: new_val = 1
        
    sim_results.append(new_val)
    current_val = new_val

df['simulated'] = sim_results


# Results
real_vals = df['high_user_ratio_smooth'].values
sim_vals = df['simulated'].values
rmse = np.sqrt(np.mean((real_vals - sim_vals)**2))

print(f"\nSimulation RMSE Error: {rmse:.4f}")

real_growth = real_vals[-1] - real_vals[0]
sim_growth = sim_vals[-1] - sim_vals[0]

drift = abs(sim_growth - real_growth)
print(f"Total Drift: {drift:.4f}") 

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['date'], real_vals, color='black', alpha=0.3, linewidth=3, label='Actual 2025')
plt.plot(df['date'], sim_vals, color='green', linestyle='--', linewidth=2, label=f'PySINDy Sim (RMSE {rmse:.2f})')
plt.title("PySindy 2025 Simulation vs Reality")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Predator Ratio (x)")
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_PLOT)
print(f"Plot: {OUTPUT_PLOT}")