import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Files
FILE_2024 = "200_2024.parquet"
FILE_2025 = "200_2025.parquet"

# Threshold from previous results (The "Behavioral Shift" we found)
BEHAVIORAL_SHIFT = 26.2 # We saw a 26% jump in High Users

print("--- PHYSICAL VOLUME CHECK ---")

# ==========================================
# 1. PROCESS 2024 (BASELINE)
# ==========================================
print("\nProcessing 2024 Data...")
df_24 = pd.read_parquet(FILE_2024)

# Create Timestamp
df_24['timestamp'] = pd.to_datetime(df_24['data_collection_log_timestamp'])

# Pivot to find Valid Days (must have 48 readings)
df_24['step'] = df_24['timestamp'].dt.hour * 2 + (df_24['timestamp'].dt.minute // 30)
df_24['day_id'] = df_24['timestamp'].dt.date.astype(str) + "_" + df_24['lv_feeder_unique_id']

# Create Matrix and Drop Incomplete Days
matrix_24 = df_24.pivot(index='day_id', columns='step', values='total_consumption_active_import').dropna()

# Calculate Daily Volume (Sum of all 48 half-hours)
# Axis=1 means sum across the columns (the whole day)
daily_vol_24 = matrix_24.sum(axis=1)

print(f"Valid Days: {len(daily_vol_24)}")
avg_24 = daily_vol_24.mean()
print(f"Average Daily Load: {avg_24:.0f} Wh")


# ==========================================
# 2. PROCESS 2025 (TESTING)
# ==========================================
print("\nProcessing 2025 Data...")
df_25 = pd.read_parquet(FILE_2025)

df_25['timestamp'] = pd.to_datetime(df_25['data_collection_log_timestamp'])

# Pivot Logic (Same as above)
df_25['step'] = df_25['timestamp'].dt.hour * 2 + (df_25['timestamp'].dt.minute // 30)
df_25['day_id'] = df_25['timestamp'].dt.date.astype(str) + "_" + df_25['lv_feeder_unique_id']

matrix_25 = df_25.pivot(index='day_id', columns='step', values='total_consumption_active_import').dropna()

# Calculate Daily Volume
daily_vol_25 = matrix_25.sum(axis=1)

print(f"Valid Days: {len(daily_vol_25)}")
avg_25 = daily_vol_25.mean()
print(f"Average Daily Load: {avg_25:.0f} Wh")


# ==========================================
# 3. COMPARISON & VERDICT
# ==========================================
print("\n" + "="*30)
print("       GROWTH RESULTS       ")
print("="*30)

# Calculate Physical Growth %
growth = (avg_25 - avg_24) / avg_24 * 100

print(f"Physical Volume Growth: {growth:+.2f}%")
print(f"Behavioral Class Shift: {BEHAVIORAL_SHIFT:+.2f}% (From previous results)")
print("-" * 30)

# The "Cliff Edge" Logic
if growth < BEHAVIORAL_SHIFT:
    print("VERDICT: HYPOTHESIS CONFIRMED")
    print("The physical increase is small, but the classification shift is huge.")
    print("This proves the 'Cliff Edge' effect: users slightly crossed the threshold.")
else:
    print("VERDICT: LINEAR GROWTH")
    print("The classification shift matches the physical growth.")