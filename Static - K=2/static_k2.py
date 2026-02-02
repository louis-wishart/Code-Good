import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import sys
import os


FILENAME = "200_2024.parquet"
OUTPUT_FILE = "clusters_2024.parquet"
OUTPUT_PLOT = "cluster_plot.png"

CLUSTERS = 2
RESTARTS = 10 

# Function 1 
def get_data():
    df = pd.read_parquet(FILENAME)
    df['timestamp'] = pd.to_datetime(df['data_collection_log_timestamp'])
    df = df[(df['timestamp'] >= '2024-03-01') & (df['timestamp'] < '2025-03-01')].copy()
    
    df['step'] = df['timestamp'].dt.hour * 2 + (df['timestamp'].dt.minute // 30)
    df['day_id'] = df['timestamp'].dt.date.astype(str) + "_" + df['lv_feeder_unique_id']
    
    return df.pivot(index='day_id', columns='step', values='total_consumption_active_import').dropna()


df_matrix = get_data()
data = df_matrix.values


# Downsample 
if len(data) > 3000:
    idx = np.random.choice(len(data), 3000, replace=False)
    train_data = data[idx]
else:
    train_data = data

best_error = float('inf')
best_centroids = None


# Function 2 
for run in range(RESTARTS):
    print(f"  Run {run+1}/10")
    
    # Random start
    centroids = train_data[np.random.choice(len(train_data), CLUSTERS, replace=False)]
    
    for i in range(30):
        # Canberra
        dists = np.zeros((len(train_data), CLUSTERS))
        for r in range(len(train_data)):
            for c in range(CLUSTERS):
                dists[r, c] = distance.canberra(train_data[r], centroids[c])
        
        # Assign clusters
        labels = np.argmin(dists, axis=1)
        
        # Average
        new_centroids = np.zeros_like(centroids)
        for k in range(CLUSTERS):
            if np.sum(labels == k) > 0:
                new_centroids[k] = np.mean(train_data[labels == k], axis=0)
            else:
                new_centroids[k] = train_data[np.random.choice(len(train_data))]
        
        # Convergence
        if np.sum(np.abs(centroids - new_centroids)) < 0.001:
            break
        centroids = new_centroids

    # Error Calc
    final_dists = np.zeros((len(train_data), CLUSTERS))
    for r in range(len(train_data)):
        for c in range(CLUSTERS):
            final_dists[r, c] = distance.canberra(train_data[r], centroids[c])
            
    error = np.sum(np.min(final_dists, axis=1))
    
    if error < best_error:
        best_error = error
        best_centroids = centroids

print(f"Best Error: {best_error:.2f}")


# Cluster 0 = low
if np.sum(best_centroids[0]) > np.sum(best_centroids[1]):
    best_centroids = best_centroids[::-1]

np.save("centroids.npy", best_centroids)


# Apple to full dataset 
full_dists = np.zeros((len(data), CLUSTERS))
for r in range(len(data)):
    for c in range(CLUSTERS):
        full_dists[r, c] = distance.canberra(data[r], best_centroids[c])
final_labels = np.argmin(full_dists, axis=1)

# Results
pd.DataFrame({'day_id': df_matrix.index, 'cluster': final_labels}).to_parquet(OUTPUT_FILE)
print(f"Results: {OUTPUT_FILE}")


# Plot 