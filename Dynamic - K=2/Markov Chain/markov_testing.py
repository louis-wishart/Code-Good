import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_TEST_DATA = "clusters_2025_k2.parquet"    
MATRIX_WEEKDAY = "training_weekday.csv"      
MATRIX_WEEKEND = "training_weekend.csv"

N = 1000     
CLUSTERS = 2      

## Function 1 
def run_simulation(n_days, start_dist, P_weekday, P_weekend, start_date):
    # Iniialise
    history = np.zeros((n_days, N), dtype=int)
    history[0] = np.random.choice(range(CLUSTERS), size=N, p=start_dist)
    
    # Weekdays
    dates = pd.date_range(start=start_date, periods=n_days)
    is_weekend = dates.dayofweek >= 5
    
    # Optimise
    P_wd = P_weekday.values
    P_we = P_weekend.values
    
    print(f"Simulation; {N} virtual feeders for {n_days} days:")
    
    # Main Loop
    for t in range(1, n_days):
        # Set Day Type
        if is_weekend[t-1]:
            P_current = P_we
        else:
            P_current = P_wd
            
        previous_states = history[t-1]
        
        # Batch Process
        for state in range(CLUSTERS):
            
            mask = (previous_states == state)
            count = np.sum(mask)
            
            if count > 0:
                
                probs = P_current[state]
                new_states = np.random.choice(range(CLUSTERS), size=count, p=probs)
                
                history[t, mask] = new_states
                
    return history.flatten() # Return one long list of all simulated states



## Markov Testing

df_test = pd.read_parquet(INPUT_TEST_DATA)

# Date Check
if 'date' not in df_test.columns:
    df_test['date'] = pd.to_datetime(df_test['day_id'].str.split('_').str[0])

df_wd = pd.read_csv(MATRIX_WEEKDAY, index_col=0)
df_we = pd.read_csv(MATRIX_WEEKEND, index_col=0)

# Sim Parameters
start_date = df_test['date'].min()
end_date = df_test['date'].max()
total_days = (end_date - start_date).days + 1

# Match Start Point
day1 = df_test[df_test['date'] == start_date]
start_counts = day1['cluster'].value_counts(normalize=True).sort_index()

# Probability Check
start_dist = np.zeros(CLUSTERS)
for i in start_counts.index:
    start_dist[i] = start_counts[i]

# Run Sim
sim_results = run_simulation(total_days, start_dist, df_wd, df_we, start_date)


# 2025 Stats
real_counts = df_test['cluster'].value_counts(normalize=True).sort_index()
real_pop = np.zeros(CLUSTERS)
for i in real_counts.index:
    real_pop[i] = real_counts[i]

# Sim Stats
sim_counts = pd.Series(sim_results).value_counts(normalize=True).sort_index()
sim_pop = np.zeros(CLUSTERS)
for i in sim_counts.index:
    sim_pop[i] = sim_counts[i]

# Results
real_0 = round(real_pop[0] * 100, 1)
sim_0  = round(sim_pop[0] * 100, 1)
diff_0 = round(real_0 - sim_0, 1)

real_1 = round(real_pop[1] * 100, 1)
sim_1  = round(sim_pop[1] * 100, 1)
diff_1 = round(real_1 - sim_1, 1)

print(f"\nPrey (Low):       Real {real_0}% vs Sim {sim_0}% (Diff: {diff_0}%)")
print(f"Predator (High):  Real {real_1}% vs Sim {sim_1}% (Diff: {diff_1}%)")

# Error (RMSE)
error = np.sqrt(np.mean((real_pop - sim_pop)**2))
print(f"\nRMSE Error: {error:.4f}")

# Plot
plt.figure(figsize=(8, 6))
indices = np.arange(CLUSTERS)
width = 0.35

plt.bar(indices - width/2, real_pop, width, label='Actual (2025)', color='#4682B4', alpha=0.8)
plt.bar(indices + width/2, sim_pop, width, label='Expected (Sim)', color='#B22222', alpha=0.8)

plt.xlabel("User State")
plt.xticks(indices, ['Prey (Low)', 'Predator (High)'])
plt.ylabel("Population Proportion")
plt.title(f"Validation: 2024 Model vs 2025 Reality\nRMSE: {error:.4f}")
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.5)

plt.savefig("validation_plot.png", dpi=300)
