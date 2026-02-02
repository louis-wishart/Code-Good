import pandas as pd
import geopandas as gpd
import random
import os

# Setup
S3_PATH = "s3://weave.energy/smart-meter"
DNO = "SSEN"
OUT_FILE = "200_2024.parquet"

# Thresholds
LOW = 10
HIGH_HARD = 50
HIGH_SOFT = 40

print(f"Scanning {DNO}:")

try:
    # Take sample from 2 weeks in March
    t1, t2 = pd.Timestamp("2024-03-01", tz="UTC"), pd.Timestamp("2024-03-14", tz="UTC")
    
    # Access S3
    probe = pd.read_parquet(
        S3_PATH,
        storage_options={"anon": True},
        filters=[("dno_alias", "==", DNO), ("data_collection_log_timestamp", ">=", t1), ("data_collection_log_timestamp", "<", t2)],
        columns=["lv_feeder_unique_id", "aggregated_device_count_active"]
    )
except Exception as e:
    print(f"S3 Error")
    exit()

# Get IDs in the broad range
ids = probe[(probe["aggregated_device_count_active"] > LOW) & 
            (probe["aggregated_device_count_active"] < HIGH_HARD)]["lv_feeder_unique_id"].unique()

if len(ids) == 0:
    print("No feeders found")
    exit()

# Sample 200
sample = random.sample(list(ids), min(len(ids), 200))
print(f"{len(sample)} feeders found, taking year")

# Extract yearly data
df = gpd.read_parquet(
    S3_PATH,
    storage_options={"anon": True},
    filters=[
        ("lv_feeder_unique_id", "in", sample),
        ("data_collection_log_timestamp", ">=", pd.Timestamp("2024-03-01", tz="UTC")),
        ("data_collection_log_timestamp", "<", pd.Timestamp("2025-03-01", tz="UTC"))
    ]
)


# Finds loss from 50 to 40
broad_count = len(df[df['aggregated_device_count_active'] < HIGH_SOFT])
strict_df = df[df['aggregated_device_count_active'] < HIGH_HARD].copy()
strict_count = len(strict_df)

print(f"\n{broad_count - strict_count} rows lost switching to N<{HIGH_SOFT}")

# Data cleaning - extreme values & clocks
strict_df = strict_df[strict_df["total_consumption_active_import"] < 20000]

dst_dates = ['2024-03-27', '2024-10-27']
strict_df = strict_df[~strict_df["data_collection_log_timestamp"].dt.strftime('%Y-%m-%d').isin(dst_dates)]

# Take 5 useful columns
cols = ["lv_feeder_unique_id", "data_collection_log_timestamp", 
        "total_consumption_active_import", "aggregated_device_count_active", "geometry"]

strict_df[cols].to_parquet(OUT_FILE)
print(f"Saved to {OUT_FILE}")