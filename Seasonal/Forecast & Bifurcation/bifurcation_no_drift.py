import numpy as np
import matplotlib.pyplot as plt


# PySindy Weekend
COEFFS_SUMMER = [-0.00573, 0.08363, -0.18024] # w = 0 (Low Stress / Warm)
COEFFS_WINTER = [-0.09451, 0.47853, -0.50894] # w = 1 (High Stress / Freezing)

# Capacity
N_FEEDERS = 200
PEAK_PREY_KW = 9.6
PEAK_PRED_KW = 20.1
SUBSTATION_LIMIT_KW = 4037 

def get_critical_ratio():
    
    avg_limit = SUBSTATION_LIMIT_KW / N_FEEDERS
    ratio = (avg_limit - PEAK_PREY_KW) / (PEAK_PRED_KW - PEAK_PREY_KW)
    return min(1.0, ratio) 

CRITICAL_THRESHOLD = get_critical_ratio()



# Solve for 0
def solve_equilibrium(coeffs):
    
    c, a, b = coeffs
    discriminant = a**2 - 4*b*c
    
    if discriminant < 0:
        return np.nan # No stable equilibrium
        
    x1 = (-a + np.sqrt(discriminant)) / (2*b)
    x2 = (-a - np.sqrt(discriminant)) / (2*b)
 
    valid_roots = [r for r in [x1, x2] if 0 <= r <= 1.0]
    if not valid_roots:
        return np.nan
        

    for r in valid_roots:
        slope = a + 2*b*r # f'(x)
        if slope < 0:     # Negative slope means it is a Stable Attractor
            return r
            
    return max(valid_roots)



# Bifurcation Sweep: w from 0 to 1
def run_pure_bifurcation():
   
    w_values = np.linspace(0, 1.0, 500)
    equilibria = []

    for w in w_values:
        # Linearly blend the environmental coefficients
        c_blend = COEFFS_SUMMER[0]*(1-w) + COEFFS_WINTER[0]*w
        a_blend = COEFFS_SUMMER[1]*(1-w) + COEFFS_WINTER[1]*w
        b_blend = COEFFS_SUMMER[2]*(1-w) + COEFFS_WINTER[2]*w
        
        eq = solve_equilibrium([c_blend, a_blend, b_blend])
        equilibria.append(eq)

    
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    fig, ax = plt.subplots(figsize=(11, 7))

    # 1. Plot the Behavioral Attractor Curve
    ax.plot(w_values, equilibria, color='#1f77b4', linewidth=4, label='Stable Behavioral Attractor ($x^*$)')

    # 2. Shade the physical context zones
    ax.axvspan(0.0, 0.2, color='#2ca02c', alpha=0.1, label='Summer Context Dominant')
    ax.axvspan(0.8, 1.0, color='#1f77b4', alpha=0.1, label='Winter Context Dominant')

    # 3. Plot the Legacy Capacity Limit (Safely at 100%)
    ax.axhline(y=CRITICAL_THRESHOLD, color='#333333', linestyle='--', linewidth=2.5, 
               label=f'Hardware Capacity (Safe to {CRITICAL_THRESHOLD*100:.0f}%)')

    # 4. Critical Annotations
    start_val = equilibria[0]
    end_val = equilibria[-1]

    # Summer Anchor
    ax.scatter(0, start_val, color='#2ca02c', s=120, zorder=5, edgecolor='white')
    ax.annotate(f"Summer Safe State\n({start_val:.1%} Predators)", 
                xy=(0, start_val), xytext=(0.03, start_val + 0.12),
                arrowprops=dict(facecolor='#2ca02c', shrink=0.05, width=2, headwidth=8, edgecolor='none'),
                fontweight='bold', color='#155d15')

    # Winter Anchor
    ax.scatter(1, end_val, color='#1f77b4', s=120, zorder=5, edgecolor='white')
    ax.annotate(f"Winter Apex State\n({end_val:.1%} Predators)", 
                xy=(1, end_val), xytext=(0.75, end_val - 0.15),
                arrowprops=dict(facecolor='#1f77b4', shrink=0.05, width=2, headwidth=8, edgecolor='none'),
                fontweight='bold', color='#104266')

    ax.set_title('Mathematical Bifurcation: The Baseline Grid (No Drift)\nProving Environmental Forcing on Predator Saturation', fontweight='bold', pad=20)
    ax.set_xlabel('Winter Severity Parameter ($w$)\n[ 0.0 = Pure Summer  \u2192  1.0 = Pure Winter ]', fontweight='bold', labelpad=15)
    ax.set_ylabel('Stable Predator Ratio ($x^*$)', fontweight='bold', labelpad=15)

    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1.0)
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='center right', framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    plt.savefig("bifurcation_no_drift.png", dpi=300)
    
   
    print(f"Natural Saturation (w=0): {start_val:.1%}")
    print(f"Natural Saturation (w=1): {end_val:.1%}")
    print(f"Hardware Tolerance: {CRITICAL_THRESHOLD:.1%}")

run_pure_bifurcation()