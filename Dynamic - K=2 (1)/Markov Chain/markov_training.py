import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Files
INPUT_FILE = "clusters_2024_k2.parquet"
OUTPUT_WEEKDAY = "training_weekday.csv"
OUTPUT_WEEKEND = "training_weekend.csv"

# Function 1 
def process_matrix(df_subset, label):

    # Count Transitions
    counts = pd.crosstab(df_subset['cluster'], df_subset['next_cluster'])
    counts = counts.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    
    # Normalise 
    row_sums = counts.sum(axis=1)
    probs = counts.div(row_sums, axis=0)
    
    filename = f"training_{label.lower()}.csv"
    probs.to_csv(filename)
    
    # Coefficients 
    alpha = probs.loc[0, 0] # Stability of Low
    beta = probs.loc[0, 1]  # Transition Low->High
    print(f"\n{label} Results:")
    print(f"  Alpha (Stability): {alpha:.4f}")
    print(f"  Beta (Transition): {beta:.4f}")
    
    # Plot
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
    plt.savefig(f"training_{label.lower()}.png", dpi=300)
    plt.close()
    
    return probs



# Markov Training 

df = pd.read_parquet(INPUT_FILE)
print(f"Loaded: {len(df)} days")

df[['date_str', 'feeder']] = df['day_id'].str.split('_', n=1, expand=True)
df['date'] = pd.to_datetime(df['date_str'])

df = df.sort_values(['feeder', 'date'])

df['next_cluster'] = df.groupby('feeder')['cluster'].shift(-1)
df['next_date'] = df.groupby('feeder')['date'].shift(-1)

# Gap Check
df['gap'] = (df['next_date'] - df['date']).dt.days

# Gap must = 1
valid = df[(df['gap'] == 1) & (df['next_cluster'].notna())].copy()

# Weekend & Weekday
valid['is_weekend'] = valid['date'].dt.dayofweek >= 5

df_weekday = valid[valid['is_weekend'] == False]
df_weekend = valid[valid['is_weekend'] == True]

print(f"Valid Transitions: {len(valid)}")
print(f"  Weekday: {len(df_weekday)}")
print(f"  Weekend: {len(df_weekend)}")

# Process Matrices
if len(df_weekday) > 0:
    process_matrix(df_weekday, "Weekday")

if len(df_weekend) > 0:
    process_matrix(df_weekend, "Weekend")

