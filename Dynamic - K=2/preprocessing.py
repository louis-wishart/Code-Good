import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os

# Files
FILE_CLUSTERS_2024 = "clusters_2024_k2.parquet"
FILE_RAW_2025 = "200_2025.parquet"
FILE_CENTROIDS = "centroids_k2.npy"

OUTPUT_2025 = "clusters_2025_k2.parquet"
OUTPUT_RATIO_2024 = "pysindy_ratio_2024.csv"
OUTPUT_RATIO_2025 = "pysindy_ratio_2025.csv"

# Ratio Calc
def calc_ratio(df, filename):
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['day_id'].str.split('_').str[0])
    
    daily = df.groupby('date')['cluster'].agg(['count', 'sum'])
    
    daily['high_user_ratio'] = daily['sum'] / daily['count']
    # Must smooth to keep model stable
    daily['high_user_ratio_smooth'] = daily['high_user_ratio'].rolling(window=3, min_periods=1).mean()
    
    daily.to_csv(filename)
   

## 2024 Data 
df_2024 = pd.read_parquet(FILE_CLUSTERS_2024)
calc_ratio(df_2024, OUTPUT_RATIO_2024)
print(f"2024 Ratio: {OUTPUT_RATIO_2024}")


## 2025 Data

df_2025 = pd.read_parquet(FILE_RAW_2025)
df_2025['timestamp'] = pd.to_datetime(df_2025['data_collection_log_timestamp'])

# Data cleaning - extreme values & clocks
df_2025 = df_2025[df_2025['total_consumption_active_import'] < 20000] 
bad_dates = [pd.Timestamp("2025-03-30").date(), pd.Timestamp("2025-10-26").date()]
df_2025 = df_2025[~df_2025['timestamp'].dt.date.isin(bad_dates)]

# Convert to Matrix
df_2025['step'] = df_2025['timestamp'].dt.hour * 2 + (df_2025['timestamp'].dt.minute // 30)
df_2025['day_id'] = df_2025['timestamp'].dt.date.astype(str) + "_" + df_2025['lv_feeder_unique_id']

# Discard incomplete days
df_matrix = df_2025.pivot(index='day_id', columns='step', values='total_consumption_active_import').dropna()
data_2025 = df_matrix.values

print(f"\nClean 2025: {len(data_2025)} daily profiles")


centroids = np.load(FILE_CENTROIDS)
CLUSTERS = len(centroids)

# Calculate Distances (Canberra)
dists = np.zeros((len(data_2025), CLUSTERS))
for r in range(len(data_2025)):
    for c in range(CLUSTERS):
        dists[r, c] = distance.canberra(data_2025[r], centroids[c])
        
labels = np.argmin(dists, axis=1)

# 2025 Transitions
df_res = pd.DataFrame({'day_id': df_matrix.index, 'cluster': labels})
df_res.to_parquet(OUTPUT_2025)
print(f"2025 Transitions: {OUTPUT_2025}")

# Get ratios for 2025
calc_ratio(df_res, OUTPUT_RATIO_2025)
print(f"2025 Ratio: {OUTPUT_RATIO_2025}")

