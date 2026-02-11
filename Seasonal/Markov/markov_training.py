import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Files
INPUT_FILE = "clusters_2024_k2.parquet"

# Output Files for 4 Contexts
OUT_WINTER_WD = "training_winter_wd.csv"
OUT_WINTER_WE = "training_winter_we.csv"
OUT_SUMMER_WD = "training_summer_wd.csv"
OUT_SUMMER_WE = "training_summer_we.csv"

# Season Definitions
WINTER_MONTHS = [11, 12, 1, 2, 3] # Nov - Mar
SUMMER_MONTHS = [4, 5, 6, 7, 8, 9, 10] # Apr - Oct

# Function 1: Matrix Processing
def process_matrix(df_subset, label, filename):
    print(f"\nProcessing {label} (N={len(df_subset)} transitions)...")
    
    # Count Transitions
    counts = pd.crosstab(df_subset['cluster'], df_subset['next_cluster'])
    # Ensure 2x2 shape even if some transitions are missing
    counts = counts.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    
    # Normalise (Row Sum = 1)
    row_sums = counts.sum(axis=1)
    row_sums[row_sums == 0] = 1 # Safety to avoid div/0
    probs = counts.div(row_sums, axis=0)
    
    # Save Matrix
    probs.to_csv(filename)
    
    # Coefficients 
    alpha = probs.loc[0, 0] # Stability of Low
    beta = probs.loc[0, 1]  # Transition Low->High
    
    print(f"  Alpha (Stability): {alpha:.4f}")
    print(f"  Beta (Transition): {beta:.4f}")
    
    # Plot Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(probs, cmap='Blues', vmin=0, vmax=1)
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{probs.iloc[i, j]:.3f}", 
                     ha='center', va='center', color='black', fontsize=12)
    
    plt.xticks([0, 1], ["Low (0)", "High (1)"])
    plt.yticks([0, 1], ["Low (0)", "High (1)"])
    plt.title(f"Transition Matrix: {label}")
    plt.colorbar(label="Probability")
    
    plt.tight_layout()
    plt.savefig(f"{label.lower()}.png", dpi=300)
    plt.close()
    
    return probs

# --- MAIN EXECUTION ---
print("--- SEASONAL MARKOV TRAINING (2024) ---")

if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} not found.")
    exit()

df = pd.read_parquet(INPUT_FILE)

# Reconstruct Date if needed
if 'date' not in df.columns:
    df[['date_str', 'feeder']] = df['day_id'].str.split('_', n=1, expand=True)
    df['date'] = pd.to_datetime(df['date_str'])

print(f"Loaded: {len(df)} days")

# Sort & Link Transitions
df = df.sort_values(['feeder', 'date'])
df['next_cluster'] = df.groupby('feeder')['cluster'].shift(-1)
df['next_date'] = df.groupby('feeder')['date'].shift(-1)

# Gap Check (Strictly 1 day diff)
df['gap'] = (df['next_date'] - df['date']).dt.days
valid = df[(df['gap'] == 1) & (df['next_cluster'].notna())].copy()

# --- CONTEXT TAGGING ---
print("Splitting into Seasonal Contexts...")

valid['month'] = valid['date'].dt.month
valid['is_weekend'] = valid['date'].dt.dayofweek >= 5
valid['is_winter'] = valid['month'].isin(WINTER_MONTHS)

# Create 4 Subsets
df_win_wd = valid[(valid['is_winter']) & (~valid['is_weekend'])]
df_win_we = valid[(valid['is_winter']) & (valid['is_weekend'])]
df_sum_wd = valid[(~valid['is_winter']) & (~valid['is_weekend'])]
df_sum_we = valid[(~valid['is_winter']) & (valid['is_weekend'])]

print(f"  Winter Weekday: {len(df_win_wd)}")
print(f"  Winter Weekend: {len(df_win_we)}")
print(f"  Summer Weekday: {len(df_sum_wd)}")
print(f"  Summer Weekend: {len(df_sum_we)}")

# --- PROCESS MATRICES ---
# We loop through and save all 4
contexts = [
    (df_win_wd, "Winter_Weekday", OUT_WINTER_WD),
    (df_win_we, "Winter_Weekend", OUT_WINTER_WE),
    (df_sum_wd, "Summer_Weekday", OUT_SUMMER_WD),
    (df_sum_we, "Summer_Weekend", OUT_SUMMER_WE)
]

for data, name, fname in contexts:
    if len(data) > 0:
        process_matrix(data, name, fname)
    else:
        print(f"WARNING: No data for {name}")

print("\n[COMPLETE] 4 Seasonal Matrices Generated.")