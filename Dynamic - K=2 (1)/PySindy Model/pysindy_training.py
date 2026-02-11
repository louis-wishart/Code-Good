import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

INPUT_FILE = "pysindy_ratio_2024.csv"
OUTPUT_FILE = "training_equations.txt"
OUTPUT_PLOT = "training_plot.png"

POLY_DEGREE = 2    
THRESHOLD = 0.0001 

# Initialise 
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df['is_weekend'] = df['date'].dt.dayofweek >= 5

x_wd = []
dx_wd = []

x_we = []
dx_we = []

# Function 1 
for i in range(len(df) - 1):
    
    row_today = df.iloc[i]
    row_tomorrow = df.iloc[i+1]
    
    days_diff = (row_tomorrow['date'] - row_today['date']).days
    if days_diff != 1:
        continue 
        
    if row_today['is_weekend'] != row_tomorrow['is_weekend']:
        continue 
        
    current_val = row_today['high_user_ratio_smooth']
    next_val = row_tomorrow['high_user_ratio_smooth']
    slope = next_val - current_val
    

    if row_today['is_weekend'] == False:
        x_wd.append(current_val)
        dx_wd.append(slope)
    else:
        x_we.append(current_val)
        dx_we.append(slope)

X_wd = np.array(x_wd).reshape(-1, 1)
X_dot_wd = np.array(dx_wd).reshape(-1, 1)

X_we = np.array(x_we).reshape(-1, 1)
X_dot_we = np.array(dx_we).reshape(-1, 1)

print(f"Weekday Transitions: {len(X_wd)}")
print(f"Weekend Transitions: {len(X_we)}")

# Define and Train 
lib = ps.PolynomialLibrary(degree=POLY_DEGREE)

model_wd = ps.SINDy(feature_library=lib, optimizer=ps.STLSQ(threshold=THRESHOLD))
model_wd.fit(X_wd, x_dot=X_dot_wd, t=1) 

model_we = ps.SINDy(feature_library=lib, optimizer=ps.STLSQ(threshold=THRESHOLD))
model_we.fit(X_we, x_dot=X_dot_we, t=1)

# Save Equations
coeffs_wd = model_wd.coefficients()[0]
coeffs_we = model_we.coefficients()[0]

print("\nGoverning Equations:")
with open(OUTPUT_FILE, "w") as f:
    f.write("PySindy Training Equations (2024)\n")
    def write_eq(label, c):
        eq_str = f"dx/dt = {c[0]:.5f} + {c[1]:.5f}x + {c[2]:.5f}x^2"
        f.write(f"{label}: {eq_str}\n")
        print(f"{label}: {eq_str}")
        
    write_eq("\nWEEKDAY", coeffs_wd)
    write_eq("WEEKEND", coeffs_we)

print(f"\nEquations: {OUTPUT_FILE}")


# Plot 
plt.figure(figsize=(12, 5))

# Weekday Plot
plt.subplot(1, 2, 1)
plt.scatter(X_wd, X_dot_wd, alpha=0.4, label='Real Data', color='blue')
plt.scatter(X_wd, model_wd.predict(X_wd), s=10, label='PySindy Model', color='red')
plt.title("Weekday")
plt.xlabel("Predator (x)")
plt.ylabel("Rate of Change (dx/dt)")
plt.legend()
plt.grid(True, alpha=0.3)

# Weekend Plot
plt.subplot(1, 2, 2)
plt.scatter(X_we, X_dot_we, alpha=0.4, label='Real Data', color='green')
plt.scatter(X_we, model_we.predict(X_we), s=10, label='PySindy Model', color='red')
plt.title("Weekend")
plt.xlabel("Predator Ratio (x)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
print(f"Plot: {OUTPUT_PLOT}")