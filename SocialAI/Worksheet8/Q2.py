import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from scipy.integrate import odeint

#UNCONSTRAINED (LIST) MODEL: ABM & DE
def list_abm(init_a=50, init_b=50, T=12):
    "ABM: Each agent produces 1 identical offspring per step (Exponential growth)"
    pop = ['a'] * init_a + ['b'] * init_b
    N_a = [init_a]
    N_b = [init_b]
    
    for t in range(T):
        offspring = pop[:]  # Faster than list comprehension for simple copy
        pop.extend(offspring)
        N_a.append(pop.count('a'))
        N_b.append(pop.count('b'))
        
    return np.array(N_a), np.array(N_b), np.arange(T + 1)

def exp_de(N0, t, r=np.log(2)):
    "DE Solution: N(t) = N0 * exp(r*t) (r=ln(2) for doubling per step)"
    return N0 * np.exp(r * t)

def get_neighbors(r, c, L):
    "Returns coordinates of 4 cardinal neighbors (non-toroidal)"
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < L and 0 <= nc < L:
            neighbors.append((nr, nc))
    return neighbors

def grid_abm(L=20, init_a=50, init_b=50, p_repro=0.1, T=200):
    "ABM: Reproduction p=0.1, requires an adjacent empty cell (Logistic growth)"
    K = L * L  # Carrying capacity
    grid = [[None for _ in range(L)] for _ in range(L)]
    
    # Init placement
    all_pos = [(i, j) for i in range(L) for j in range(L)]
    random.shuffle(all_pos)
    for pos in all_pos[:init_a]: grid[pos[0]][pos[1]] = 'a'
    for pos in all_pos[init_a:init_a + init_b]: grid[pos[0]][pos[1]] = 'b'
    
    N_a = [init_a]
    N_b = [init_b]
    
    for t in range(T):
        curr_grid = copy.deepcopy(grid)
        agents = [(r, c) for r in range(L) for c in range(L) if curr_grid[r][c] is not None]
        random.shuffle(agents)
        
        for r, c in agents:
            if random.random() < p_repro:
                neighbors = get_neighbors(r, c, L)
                empty_spots = [pos for pos in neighbors if grid[pos[0]][pos[1]] is None]
                
                if empty_spots:
                    nr, nc = random.choice(empty_spots)
                    grid[nr][nc] = curr_grid[r][c]  # Place offspring
        
        N_a.append(sum(row.count('a') for row in grid))
        N_b.append(sum(row.count('b') for row in grid))
        
    return (np.array(N_a), np.array(N_b), np.arange(T + 1), K)

def logistic_de_rate(N_vec, t, r, K):
    "DE: dN/dt = r * N * (1 - N/K)"
    # Assuming N_vec is the total population N = N_a + N_b
    # This function is used by odeint to find the continuous solution
    N = N_vec[0] # Assuming single variable for total pop for simplicity
    dNdt = r * N * (1 - N / K)
    return [dNdt]
#SIMULATION RUN & PLOTTING

# 1. LIST MODEL
N_a_list_abm, N_b_list_abm, t_list = list_abm(init_a=50, init_b=30, T=12)
N_a_list_de = exp_de(50, t_list)
N_b_list_de = exp_de(30, t_list)

# 2. GRID MODEL
N_a_grid_abm, N_b_grid_abm, t_grid, K = grid_abm(L=20, init_a=50, init_b=30, T=200)

# DE for Grid Model (Total Population)
r_fit = 0.08  # Growth rate fitted to the ABM
N0_total = N_a_grid_abm[0] + N_b_grid_abm[0]
t_cont = np.linspace(0, 200, 201)
sol_total = odeint(logistic_de_rate, [N0_total], t_cont, args=(r_fit, K)).flatten()

# Scale DE solution back to a and b using initial ratio
ratio_a = N_a_grid_abm[0] / N0_total
ratio_b = N_b_grid_abm[0] / N0_total
N_a_grid_de = sol_total * ratio_a
N_b_grid_de = sol_total * ratio_b


#PLOTTING
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
ax1, ax2, ax3, ax4 = axes.flatten()

# Plot 1: List Model (Exponential)
ax1.plot(t_list, N_a_list_abm, 'o-', label="ABM 'a'", markersize=6)
ax1.plot(t_list, N_a_list_de, '-', label="DE 'a'")
ax1.plot(t_list, N_b_list_abm, 's-', label="ABM 'b'", markersize=6)
ax1.plot(t_list, N_b_list_de, '-', label="DE 'b'")
ax1.set_title("List Model: Exponential Growth (ABM vs DE)")
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Population Size (N)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Grid Model (Logistic)
ax2.plot(t_grid, N_a_grid_abm, 'o-', label="ABM 'a'", markersize=4, alpha=0.8)
ax2.plot(t_grid, N_b_grid_abm, 's-', label="ABM 'b'", markersize=4, alpha=0.8)
ax2.plot(t_cont, N_a_grid_de, '-', label="DE 'a'")
ax2.plot(t_cont, N_b_grid_de, '-', label="DE 'b'")
ax2.axhline(y=K, color='k', linestyle='--', alpha=0.5, label=f'K={K} (Capacity)')
ax2.set_title("Grid Model: Logistic-like Growth (ABM vs DE)")
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Population Size (N)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Total Population Comparison
ax3.plot(t_list, N_a_list_abm + N_b_list_abm, 'o-', label="List ABM Total")
ax3.plot(t_grid, N_a_grid_abm + N_b_grid_abm, 's-', label="Grid ABM Total")
ax3.plot(t_cont, sol_total, '--', label="Grid DE Total")
ax3.axhline(y=K, color='r', linestyle='--', label=f"Grid Capacity K={K}")
ax3.set_title("Total Population Dynamics: List (Exp) vs Grid (Log)")
ax3.set_xlabel("Time (t)")
ax3.set_ylabel("Total Population (N)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Growth Rates
# Instantaneous Growth Rate ≈ d(ln(N))/dt
list_g_rate = np.diff(np.log(N_a_list_abm + N_b_list_abm + 1e-10))
grid_g_rate = np.diff(np.log(N_a_grid_abm + N_b_grid_abm + 1e-10))
ax4.plot(t_list[1:], list_g_rate, 'o-', label="List Model (Constant)")
ax4.plot(t_grid[1:70], grid_g_rate[:69], 's-', label="Grid Model (Density Dependent)")
ax4.axhline(y=np.log(2), color='k', linestyle='--', label=r"$r_{exp} = \ln(2) \approx 0.693$")
ax4.set_title(r"Instantaneous Growth Rate ($\frac{d\ln(N)}{dt}$)")
ax4.set_xlabel("Time (t)")
ax4.set_ylabel("Growth Rate")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()