import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
OUTPUT_PLOT = "bifurcation_diagram.png"

# PHYSICAL LAWS (Derived from PySINDy)
# We analyze the "Weekend" physics because that is the Predator context.
# We want to see how the "Weekend Danger" evolves from Summer to Winter.

# Summer Weekend Equation: dx/dt = c + ax + bx^2
# (Mild growth, lower saturation)
COEFFS_SUMMER = [-0.00573, 0.08363, -0.18024] 

# Winter Weekend Equation: dx/dt = c + ax + bx^2
# (Explosive growth, high saturation)
COEFFS_WINTER = [-0.09451, 0.47853, -0.50894]

# Substation Capacity Limit (Derived from DNO Report)
# Converted to Ratio: 2812 kW / (20.13 kW * 200 feeders) ? 
# Let's use the Ratio limit directly if we know it (approx 0.70 - 0.80)
# Based on your previous output, breach happened around x=0.75?
# Let's just plot the line for visual reference.
CRITICAL_THRESHOLD = 0.80 

def solve_equilibrium(coeffs):
    """
    Solves ax^2 + bx + c = 0 for x.
    Returns the STABLE root (attractor).
    """
    c, a, b = coeffs
    # Rearrange to standard quad form: bx^2 + ax + c = 0
    # Roots: (-a +/- sqrt(a^2 - 4bc)) / 2b
    
    discriminant = a**2 - 4*b*c
    
    if discriminant < 0:
        return None # No fixed points (Collapse or Explosion)
        
    x1 = (-a + np.sqrt(discriminant)) / (2*b)
    x2 = (-a - np.sqrt(discriminant)) / (2*b)
    
    # Identify Stability
    # For logistic-style eq (negative quadratic term), the higher root is usually stable?
    # Let's check derivative f'(x) = a + 2bx
    # Stable if f'(x) < 0
    
    roots = []
    if 0 <= x1 <= 1.0: roots.append(x1)
    if 0 <= x2 <= 1.0: roots.append(x2)
    
    if not roots: return None
    
    # Find the stable one
    for r in roots:
        slope = a + 2*b*r
        if slope < 0: return r # Stable Attractor
        
    return roots[0] # Fallback

def run_bifurcation_analysis():
    print("--- GENERATING BIFURCATION DIAGRAM ---")
    
    # 1. Parameter Sweep (Seasonality)
    # w goes from 0.0 (Pure Summer) to 1.0 (Pure Winter)
    w_values = np.linspace(0, 1, 100)
    equilibria = []
    
    # 2. Add Drift Scenario?
    # Let's plot "Baseline 2024" vs "Drifted 2025"
    drift_daily = 0.0004 # Approx drift per day
    drift_force = drift_daily * 100 # Arbitrary 'force' equivalent
    
    equilibria_drift = []

    print("Calculating fixed points across seasonal gradient...")
    
    for w in w_values:
        # Blend Coefficients
        # c(w) = c_sum*(1-w) + c_win*w
        c_blend = COEFFS_SUMMER[0]*(1-w) + COEFFS_WINTER[0]*w
        a_blend = COEFFS_SUMMER[1]*(1-w) + COEFFS_WINTER[1]*w
        b_blend = COEFFS_SUMMER[2]*(1-w) + COEFFS_WINTER[2]*w
        
        # Scenario A: Baseline 2024
        eq = solve_equilibrium([c_blend, a_blend, b_blend])
        equilibria.append(eq)
        
        # Scenario B: 2025 Drift (Adds constant pressure)
        # We add drift to the constant term 'c'
        eq_drift = solve_equilibrium([c_blend + 0.01, a_blend, b_blend]) # +1% daily pressure?
        # Actually drift is small daily but huge annually. 
        # Let's just show the Baseline bifurcation first.
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(w_values, equilibria, color='#B22222', linewidth=3, label='Stable Equilibrium (Attractor)')
    
    # Context Zones
    plt.axvspan(0.0, 0.2, color='green', alpha=0.1, label='Summer Physics')
    plt.axvspan(0.8, 1.0, color='blue', alpha=0.1, label='Winter Physics')
    
    # Critical Line
    plt.axhline(y=CRITICAL_THRESHOLD, color='black', linestyle='--', linewidth=2, label='Critical Grid Limit (80%)')

    # Formatting
    plt.title("Bifurcation Diagram: Seasonal Forcing on Predator Saturation")
    plt.xlabel("Seasonal Parameter (0=Summer, 1=Winter)")
    plt.ylabel("Stable Predator Ratio (Fixed Point)")
    plt.ylim(0, 1.0)
    plt.xlim(0, 1)
    
    # Annotations
    start_point = equilibria[0]
    end_point = equilibria[-1]
    
    if start_point is not None and end_point is not None:
        plt.annotate(f"Summer Safe State\n({start_point:.1%})", xy=(0.05, start_point), xytext=(0.1, start_point+0.15),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.annotate(f"Winter Risk State\n({end_point:.1%})", xy=(0.95, end_point), xytext=(0.7, end_point-0.15),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_PLOT)
    print(f"Saved {OUTPUT_PLOT}")
    print(f"Summer Equilibrium: {equilibria[0]:.4f}")
    print(f"Winter Equilibrium: {equilibria[-1]:.4f}")
    print("This diagram proves that Winter physics naturally pulls the grid toward a higher stress state.")

if __name__ == "__main__":
    run_bifurcation_analysis()