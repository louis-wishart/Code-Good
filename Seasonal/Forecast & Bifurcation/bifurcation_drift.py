import numpy as np
import matplotlib.pyplot as plt

# PySindy Weekend
COEFFS_SUM_WE = [-0.00573, 0.08363, -0.18024] # w = 0 (Summer)
COEFFS_WIN_WE = [-0.09451, 0.47853, -0.50894] # w = 1 (Winter)

# Capacity
N_FEEDERS = 200
PEAK_PREY_KW = 9.6
PEAK_PRED_KW = 20.1
SUBSTATION_LIMIT_MW = 4.037  

# 14.96% Annual Drift (Compounded)
DRIFT_2024 = 0.0000 
DRIFT_2026 = 0.3216 # ~32.2% total hardware growth over 2 years
DRIFT_2027 = 0.5193 # ~51.9% total hardware growth over 3 years

def ratio_to_megawatts(ratio, drift_penalty):
    if np.isnan(ratio): return np.nan
    scaled_prey = PEAK_PREY_KW * (1 + drift_penalty)
    scaled_pred = PEAK_PRED_KW * (1 + drift_penalty)
    total_kw = N_FEEDERS * ((1 - ratio) * scaled_prey + ratio * scaled_pred)
    return total_kw / 1000.0  # Convert to MW

# Bifurcation (w from 0 to 1) ---
w_values = np.linspace(0, 1.0, 500)
roots_2024_mw = []
roots_2026_mw = []
roots_2027_mw = []

for w in w_values:
    c = w * COEFFS_WIN_WE[0] + (1 - w) * COEFFS_SUM_WE[0]
    a = w * COEFFS_WIN_WE[1] + (1 - w) * COEFFS_SUM_WE[1]
    b = w * COEFFS_WIN_WE[2] + (1 - w) * COEFFS_SUM_WE[2]
    
    roots = np.roots([b, a, c])
    real_roots = [r.real for r in roots if np.isclose(r.imag, 0)]
    
    if len(real_roots) == 2:
        stable_ratio = max(real_roots)
        roots_2024_mw.append(ratio_to_megawatts(stable_ratio, DRIFT_2024))
        roots_2026_mw.append(ratio_to_megawatts(stable_ratio, DRIFT_2026))
        roots_2027_mw.append(ratio_to_megawatts(stable_ratio, DRIFT_2027))
    else:
        roots_2024_mw.append(np.nan)
        roots_2026_mw.append(np.nan)
        roots_2027_mw.append(np.nan)

# Plot

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11
})

fig, ax = plt.subplots(figsize=(12, 7.5))


ax.axhspan(SUBSTATION_LIMIT_MW, 6.0, color='#ffcccc', alpha=0.3, label='Overload Zone')

ax.axhline(y=SUBSTATION_LIMIT_MW, color='#333333', linestyle='--', linewidth=2.5, 
            label=f'Demand Limit ({SUBSTATION_LIMIT_MW:.2f} MW)')

ax.plot(w_values, roots_2024_mw, color='#2A9D8F', linewidth=3.5, label='2024 Baseline (No Drift)')
ax.plot(w_values, roots_2026_mw, color='#F4A261', linewidth=3.5, label='2026 Projection (+32% Drift)')
ax.plot(w_values, roots_2027_mw, color='#D62828', linewidth=3.5, label='2027 Projection (+52% Drift)')

tipping_points = {'2026': None, '2027': None}

for mw_curve, color, year, offset_y in [(roots_2026_mw, '#F4A261', '2026', -0.3), 
                                        (roots_2027_mw, '#D62828', '2027', 0.2)]:
    # Find exact intersection using interpolation
    if max(mw_curve) >= SUBSTATION_LIMIT_MW:
    
        exact_w = np.interp(SUBSTATION_LIMIT_MW, mw_curve, w_values)
        tipping_points[year] = exact_w    
        
        ax.scatter(exact_w, SUBSTATION_LIMIT_MW, color=color, s=150, zorder=6, edgecolor='white', linewidth=2)
        
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5, alpha=0.9)
        ax.annotate(f'{year} Tipping Point\n$w = {exact_w:.2f}$', 
                     xy=(exact_w, SUBSTATION_LIMIT_MW), 
                     xytext=(exact_w - 0.15, SUBSTATION_LIMIT_MW + offset_y),
                     arrowprops=dict(facecolor=color, shrink=0.05, width=2, headwidth=8, edgecolor='none'),
                     color='#333333', fontweight='bold', ha='center', bbox=bbox_props)

ax.set_title('Bifurcation Analysis\n(Including 14.96% Annual Drift)', fontweight='bold', pad=20)
ax.set_xlabel('Winter Severity Parameter ($w$)\n[ 0.0 = Pure Summer  \u2192  1.0 = Pure Winter ]', fontweight='bold', labelpad=15)
ax.set_ylabel('Peak Demand (MW)', fontweight='bold', labelpad=15)
ax.set_ylim(2.0, 5.5)
ax.set_xlim(0, 1.0)
ax.grid(True, linestyle=':', alpha=0.6, color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower right', framealpha=0.9, edgecolor='#cccccc', borderpad=1)


plt.tight_layout()
plt.savefig("bifurcation_drift.png", dpi=300)



# Print Summary


print(f"Substation Firm Capacity Limit: {SUBSTATION_LIMIT_MW:.2f} MW\n")

print("[1] 2024 BASELINE (Current State)")
print(f"    Peak Winter Demand (w=1.0): {roots_2024_mw[-1]:.2f} MW")
print(f"    Status: SAFE. Headroom = {SUBSTATION_LIMIT_MW - roots_2024_mw[-1]:.2f} MW ")

print("\n[2] 2026 FORECAST ")
print(f"    Peak Winter Demand (w=1.0): {roots_2026_mw[-1]:.2f} MW")
if tipping_points['2026'] is not None:
    print(f"    Limit breached at w = {tipping_points['2026']:.2f}")
    print(f"    Short by {roots_2026_mw[-1] - SUBSTATION_LIMIT_MW:.2f} MW at w=1")

print("\n[3] 2027 FORECAST")
print(f"    Peak Winter Demand (w=1.0): {roots_2027_mw[-1]:.2f} MW")
if tipping_points['2027'] is not None:
    print(f"    Limit breached at Winter Severity w = {tipping_points['2027']:.2f}")
    print(f"    Short by {roots_2027_mw[-1] - SUBSTATION_LIMIT_MW:.2f} MW at w=1")
