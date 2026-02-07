import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_2024 = "200_2024.parquet"
FILE_2025 = "200_2025.parquet"

input = input("Enter Predator Shift % from Markov results (1dp): ")
PREDATOR_SHIFT = float(input)

## 2024 

print("\n2024 Data: ")
df_24 = pd.read_parquet(FILE_2024)

# Create Timestamp
df_24['timestamp'] = pd.to_datetime(df_24['data_collection_log_timestamp'])

# Valid Days
df_24['step'] = df_24['timestamp'].dt.hour * 2 + (df_24['timestamp'].dt.minute // 30)
df_24['day_id'] = df_24['timestamp'].dt.date.astype(str) + "_" + df_24['lv_feeder_unique_id']

# Matrix and Drop Incomplete Days
matrix_24 = df_24.pivot(index='day_id', columns='step', values='total_consumption_active_import').dropna()

# Daily Volume 
daily_vol_24 = matrix_24.sum(axis=1)

print(f"            Valid Days: {len(daily_vol_24)}")
avg_24 = daily_vol_24.mean()
print(f"            Average Daily Load: {avg_24:.0f} Wh")


## 2025 

print("\n2025 Data: ")
df_25 = pd.read_parquet(FILE_2025)

# Create Timestamp
df_25['timestamp'] = pd.to_datetime(df_25['data_collection_log_timestamp'])

# Valid Days
df_25['step'] = df_25['timestamp'].dt.hour * 2 + (df_25['timestamp'].dt.minute // 30)
df_25['day_id'] = df_25['timestamp'].dt.date.astype(str) + "_" + df_25['lv_feeder_unique_id']

# Matrix and Drop Incomplete Days
matrix_25 = df_25.pivot(index='day_id', columns='step', values='total_consumption_active_import').dropna()

# Daily Volume
daily_vol_25 = matrix_25.sum(axis=1)

print(f"            Valid Days: {len(daily_vol_25)}")
avg_25 = daily_vol_25.mean()
print(f"            Average Daily Load: {avg_25:.0f} Wh")



## Compare 

# Calculate Growth 
growth = (avg_25 - avg_24) / avg_24 * 100

print(f"\nDemand Growth: {growth:+.2f}%")
print(f"Predator Shift: {PREDATOR_SHIFT:+.2f}% ")

usage_dir = "increased" if growth >= 0 else "decreased"
predator_dir = "rise" if PREDATOR_SHIFT >= 0 else "fall"
print(f"\nElectrical Demand {usage_dir} {abs(growth):.2f}% compared to "
      f"{abs(PREDATOR_SHIFT):.2f}% predator {predator_dir} from 2024 to 2025")

