import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
INPUT_FILE = "pysindy_ratio_2024.csv"
OUTPUT_FILE = "pysindy_equations_seasonal.txt"
OUTPUT_PLOT = "pysindy_seasonal_fit.png"

POLY_DEGREE = 2    
THRESHOLD = 0.0001 

# Season Definitions
WINTER_MONTHS = [11, 12, 1, 2, 3] # Nov-Mar
SUMMER_MONTHS = [4, 5, 6, 7, 8, 9, 10] # Apr-Oct

# --- INITIALIZATION ---
if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} missing.")
    exit()

df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Tag Contexts
df['month'] = df['date'].dt.month
df['is_weekend'] = df['date'].dt.dayofweek >= 5
df['is_winter'] = df['month'].isin(WINTER_MONTHS)

# Storage for our 4 Contexts
data_buckets = {
    "Winter_Weekday": {"x": [], "dx": []},
    "Winter_Weekend": {"x": [], "dx": []},
    "Summer_Weekday": {"x": [], "dx": []},
    "Summer_Weekend": {"x": [], "dx": []}
}

# --- FUNCTION 1: MANUAL DERIVATIVE CALCULATION ---
# We loop through the timeline to calculate slope (Next - Current)
# This prevents calculating derivatives across gaps (e.g. DNO outages or season changes)

print("Calculating derivatives manually...")

for i in range(len(df) - 1):
    row_today = df.iloc[i]
    row_tomorrow = df.iloc[i+1]
    
    # 1. Check Continuity (Must be 1 day apart)
    days_diff = (row_tomorrow['date'] - row_today['date']).days
    if days_diff != 1:
        continue 
        
    # 2. Check Context Consistency
    # We only compute physics if Today and Tomorrow are in the SAME context.
    if row_today['is_weekend'] != row_tomorrow['is_weekend']:
        continue
    if row_today['is_winter'] != row_tomorrow['is_winter']:
        continue
        
    # 3. Calculate Math
    current_val = row_today['high_user_ratio_smooth']
    next_val = row_tomorrow['high_user_ratio_smooth']
    slope = next_val - current_val
    
    # 4. Sort into correct bucket
    label = ""
    if row_today['is_winter']: label += "Winter_"
    else: label += "Summer_"
    
    if row_today['is_weekend']: label += "Weekend"
    else: label += "Weekday"
    
    data_buckets[label]["x"].append(current_val)
    data_buckets[label]["dx"].append(slope)

# --- FUNCTION 2: TRAINING & PLOTTING ---
print("\n--- Training Seasonal Models ---")

# Setup Plot grid (2x2)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

with open(OUTPUT_FILE, "w") as f:
    f.write("PySINDy Seasonal Equations (2024)\n")
    
    for i, (label, data) in enumerate(data_buckets.items()):
        X = np.array(data["x"]).reshape(-1, 1)
        X_dot = np.array(data["dx"]).reshape(-1, 1)
        
        print(f"Processing {label}: {len(X)} transitions")
        
        if len(X) < 10:
            print(f"  [WARNING] Not enough data for {label}")
            continue
            
        # Define and Fit Model
        lib = ps.PolynomialLibrary(degree=POLY_DEGREE)
        opt = ps.STLSQ(threshold=THRESHOLD)
        model = ps.SINDy(feature_library=lib, optimizer=opt)
        
        # Fit using our manually computed derivatives
        model.fit(X, x_dot=X_dot, t=1)
        
        # Extract Coefficients
        c = model.coefficients()[0]
        eq_str = f"dx/dt = {c[0]:.5f} + {c[1]:.5f}x + {c[2]:.5f}x^2"
        
        # Save and Print
        f.write(f"\n{label}: {eq_str}\n")
        print(f"  Result: {eq_str}")
        
        # Plot Fit
        ax = axes[i]
        ax.scatter(X, X_dot, alpha=0.4, label='Real Data', color='blue')
        # Sort X for cleaner line plotting
        X_sorted = np.sort(X, axis=0)
        ax.plot(X_sorted, model.predict(X_sorted), color='red', linewidth=2, label='PySINDy Fit')
        
        ax.set_title(label)
        ax.set_xlabel("High User Ratio (x)")
        ax.set_ylabel("Rate of Change (dx/dt)")
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
print(f"\n[SUCCESS] Plots saved to {OUTPUT_PLOT}")
print(f"Equations saved to {OUTPUT_FILE}")