#WORKSHEET 8
#Student ID - 24159091
#Name - Bisistha Patra 

#Question 1 -
# import matplotlib.pyplot as plt
# import random
# import copy

# def simulate_list_model(init_a=50, init_b=50, steps=10):
#     # Initialize population list (pop)
#     pop = ['a'] * init_a + ['b'] * init_b
    
#     # Track population sizes
#     pop_a = [init_a]
#     pop_b = [init_b]
#     pop_total = [init_a + init_b]
    
#     for t in range(steps):
#         # Reproduction: Each agent produces one offspring of same genotype
#         # The new generation (offspring) is simply a duplicate of the current pop.
#         offspring = [agent for agent in pop]
#         pop.extend(offspring)
        
#         # Collect sizes
#         N_a = pop.count('a')
#         N_b = pop.count('b')
#         pop_a.append(N_a)
#         pop_b.append(N_b)
#         pop_total.append(N_a + N_b)
        
#     return pop_a, pop_b, pop_total

# # Run simulation
# pop_a, pop_b, pop_total = simulate_list_model(steps=15) # Increased steps for a better plot

# # Plot dynamics
# plt.plot(pop_a, label="N_a ('a' population)")
# plt.plot(pop_b, label="N_b ('b' population)")
# plt.plot(pop_total, label="Total N (Population Size)")
# plt.xlabel("Time Steps (t)")
# plt.ylabel("Population Size (N)")
# plt.title("List of Agents Model: Unconstrained Exponential Growth")
# plt.legend()
# plt.show()


# def get_neighbors(r, c, L):
#     # L for grid size (Length/Width)
#     # 4 neighbors (up, down, left, right), non-toroidal
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#     neighbors = []
#     for dr, dc in directions:
#         nr, nc = r + dr, c + dc
#         if 0 <= nr < L and 0 <= nc < L:
#             neighbors.append((nr, nc))
#     return neighbors

# def simulate_grid_model(L=20, init_a=50, init_b=50, p_repro=0.1, steps=100):
#     # Initialize grid (lattice)
#     grid = [[None for _ in range(L)] for _ in range(L)]
    
#     # Place initial agents randomly
#     all_pos = [(i, j) for i in range(L) for j in range(L)]
#     random.shuffle(all_pos)
    
#     # Use standard list indexing/slicing for initialization
#     for r, c in all_pos[:init_a]:
#         grid[r][c] = 'a'
#     for r, c in all_pos[init_a:init_a + init_b]:
#         grid[r][c] = 'b'
    
#     # Track population sizes
#     pop_a = [init_a]
#     pop_b = [init_b]
#     pop_total = [init_a + init_b]
    
#     for t in range(steps):
#         # Current state snapshot for agent iteration
#         current_grid = copy.deepcopy(grid)
        
#         # Find all agents (pos = position)
#         agents_pos = [(r, c) for r in range(L) for c in range(L) if current_grid[r][c] is not None]
#         random.shuffle(agents_pos) # Process agents in random order
        
#         for r, c in agents_pos:
#             if random.random() < p_repro:  # Attempt reproduction with probability p_repro
                
#                 neighbors = get_neighbors(r, c, L)
#                 # Check target grid (which reflects new placements this step) for empty spots
#                 empty_spots = [pos for pos in neighbors if grid[pos[0]][pos[1]] is None]
                
#                 if empty_spots:
#                     # Place offspring on random empty spot
#                     nr, nc = random.choice(empty_spots)
#                     grid[nr][nc] = current_grid[r][c]  # Same genotype
        
#         # Collect sizes
#         N_a = sum(row.count('a') for row in grid)
#         N_b = sum(row.count('b') for row in grid)
#         pop_a.append(N_a)
#         pop_b.append(N_b)
#         pop_total.append(N_a + N_b)
        
#     return pop_a, pop_b, pop_total

# # Run simulation
# pop_a, pop_b, pop_total = simulate_grid_model()

# # Plot dynamics
# plt.plot(pop_a, label="N_a ('a' population)")
# plt.plot(pop_b, label="N_b ('b' population)")
# plt.plot(pop_total, label="Total N (Population Size)")
# plt.xlabel("Time Steps (t)")
# plt.ylabel("Population Size (N)")
# plt.title("Simple Grid World Model: Spatially Constrained Growth")
# plt.legend()
# plt.show()
#end of q1 

#Question 2 - 
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# import copy
# from scipy.integrate import odeint

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