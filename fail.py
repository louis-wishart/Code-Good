import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
CENTROIDS_FILE = "centroids_k2.npy"
N_FEEDERS = 200
SUBSTATION_CAPACITY_KW = 2812 # From your report output
FAILURE_RATIO = 0.660 # The ratio where the grid breaks (from your report)

def plot_failure_anatomy():
    if not os.path.exists(CENTROIDS_FILE):
        print("Error: Centroids file not found.")
        return

    # Load Physics
    centroids = np.load(CENTROIDS_FILE)
    prey_profile_kw = (centroids[0] * 2) / 1000
    pred_profile_kw = (centroids[1] * 2) / 1000
    
    # Calculate System Load at Failure Point
    total_prey_kw = (1 - FAILURE_RATIO) * prey_profile_kw * N_FEEDERS
    total_pred_kw = (FAILURE_RATIO) * pred_profile_kw * N_FEEDERS
    total_system_kw = total_prey_kw + total_pred_kw
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # X-Axis (Time)
    time_steps = np.linspace(0, 24, 48)
    
    plt.plot(time_steps, total_system_kw, color='black', linewidth=3, label='Total System Load')
    plt.fill_between(time_steps, 0, total_prey_kw, color='green', alpha=0.3, label='Prey Contribution')
    plt.fill_between(time_steps, total_prey_kw, total_system_kw, color='red', alpha=0.3, label='Predator Contribution')
    
    # Capacity Line
    plt.axhline(y=SUBSTATION_CAPACITY_KW, color='red', linestyle='--', linewidth=2, label='Substation Fuse Limit')
    
    plt.title(f"Anatomy of a Blackout: Load Profile at {FAILURE_RATIO*100:.1f}% Saturation")
    plt.ylabel("Power Demand (kW)")
    plt.xlabel("Time of Day (Hours)")
    plt.xlim(0, 24)
    plt.xticks(np.arange(0, 25, 4))
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig("failure_profile_anatomy.png")
    print("Saved 'failure_profile_anatomy.png'")

if __name__ == "__main__":
    plot_failure_anatomy()