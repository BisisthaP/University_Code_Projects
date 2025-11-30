import matplotlib.pyplot as plt
import random
import copy

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


def get_neighbors(r, c, L):
    # L for grid size (Length/Width)
    # 4 neighbors (up, down, left, right), non-toroidal
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < L and 0 <= nc < L:
            neighbors.append((nr, nc))
    return neighbors

def simulate_grid_model(L=20, init_a=50, init_b=50, p_repro=0.1, steps=100):
    # Initialize grid (lattice)
    grid = [[None for _ in range(L)] for _ in range(L)]
    
    # Place initial agents randomly
    all_pos = [(i, j) for i in range(L) for j in range(L)]
    random.shuffle(all_pos)
    
    # Use standard list indexing/slicing for initialization
    for r, c in all_pos[:init_a]:
        grid[r][c] = 'a'
    for r, c in all_pos[init_a:init_a + init_b]:
        grid[r][c] = 'b'
    
    # Track population sizes
    pop_a = [init_a]
    pop_b = [init_b]
    pop_total = [init_a + init_b]
    
    for t in range(steps):
        # Current state snapshot for agent iteration
        current_grid = copy.deepcopy(grid)
        
        # Find all agents (pos = position)
        agents_pos = [(r, c) for r in range(L) for c in range(L) if current_grid[r][c] is not None]
        random.shuffle(agents_pos) # Process agents in random order
        
        for r, c in agents_pos:
            if random.random() < p_repro:  # Attempt reproduction with probability p_repro
                
                neighbors = get_neighbors(r, c, L)
                # Check target grid (which reflects new placements this step) for empty spots
                empty_spots = [pos for pos in neighbors if grid[pos[0]][pos[1]] is None]
                
                if empty_spots:
                    # Place offspring on random empty spot
                    nr, nc = random.choice(empty_spots)
                    grid[nr][nc] = current_grid[r][c]  # Same genotype
        
        # Collect sizes
        N_a = sum(row.count('a') for row in grid)
        N_b = sum(row.count('b') for row in grid)
        pop_a.append(N_a)
        pop_b.append(N_b)
        pop_total.append(N_a + N_b)
        
    return pop_a, pop_b, pop_total

# Run simulation
pop_a, pop_b, pop_total = simulate_grid_model()

# Plot dynamics
plt.plot(pop_a, label="N_a ('a' population)")
plt.plot(pop_b, label="N_b ('b' population)")
plt.plot(pop_total, label="Total N (Population Size)")
plt.xlabel("Time Steps (t)")
plt.ylabel("Population Size (N)")
plt.title("Simple Grid World Model: Spatially Constrained Growth")
plt.legend()
plt.show()