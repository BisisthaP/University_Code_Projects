import numpy 
import matplotlib.pyplot as plt
import random 

# --- Core Model Class ---
class SchellingModel:
    def __init__(self, N, p_empty, T, bias_factor):
        self.N = N                                    # Grid size (N x N)
        self.p_empty = p_empty                        # Fraction of empty cells
        self.T = T                                    # Minimum required similarity (T = Threshold)
        self.bias_factor = bias_factor                # Factor controlling how much diversity increases tolerance

        # Define possible cell states and their initial probabilities
        cells = [0, 1, 2]
        probs = [p_empty, (1 - p_empty) / 2, (1 - p_empty) / 2]
        # Create the initial random grid
        self.grid = np.random.choice(cells, size=(N, N), p=probs)

    def _get_similarity(self, x, y):
        agent_type = self.grid[x, y]
        
        # An empty cell has no preference/similarity to measure
        if agent_type == 0:
            return 0.0

        neighbor_types = []
        # Check all 8 surrounding neighbors (using toroidal/wrap-around boundaries)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if not (i == 0 and j == 0):
                    # Calculate neighbor coordinates with wrap-around (% N)
                    nx, ny = (x + i) % self.N, (y + j) % self.N
                    
                    # Only consider occupied neighbors (Group A or B)
                    if self.grid[nx, ny] != 0:
                        neighbor_types.append(self.grid[nx, ny])

        if len(neighbor_types) == 0:
            # If no occupied neighbors, the agent is maximally happy
            return 1.0

        # Count neighbors belonging to the same group as the current cell
        similar_count = sum(1 for val in neighbor_types if val == agent_type)
        return similar_count / len(neighbor_types)

    def _is_happy(self, x, y, similarity):
        # Diversity is the opposite of similarity
        diversity = 1.0 - similarity
        
        # Calculate the effective threshold: a high diversity lowers the requirement
        # for similarity, making the agent more tolerant.
        effective_T = self.T - self.bias_factor * diversity
        
        return similarity >= effective_T

    def fraction_satisfied(self):
        happy_count = 0
        total_agents = 0
        
        for x in range(self.N):
            for y in range(self.N):
                if self.grid[x, y] != 0:
                    total_agents += 1
                    similarity = self._get_similarity(x, y)
                    if self._is_happy(x, y, similarity):
                        happy_count += 1
                        
        # Return the fraction (guard against division by zero if no agents exist)
        return happy_count / total_agents if total_agents else 0.0

    def step(self):
        unhappy_locs = []
        
        # 1. Identify all unhappy agents
        for x in range(self.N):
            for y in range(self.N):
                agent_type = self.grid[x, y]
                
                if agent_type != 0:
                    similarity = self._get_similarity(x, y)
                    
                    # Check for unhappiness using the dynamic threshold
                    if not self._is_happy(x, y, similarity):
                        unhappy_locs.append((x, y))

        # 2. Find empty cells and shuffle lists to randomize movement order
        empty_spots = list(zip(*np.where(self.grid == 0)))
        random.shuffle(empty_spots)
        random.shuffle(unhappy_locs)
        
        # 3. Move agents (max agents moved is min(unhappy, empty_cells))
        for (old_x, old_y), (new_x, new_y) in zip(unhappy_locs, empty_spots):
            # Move the agent to the new, empty spot
            self.grid[new_x, new_y] = self.grid[old_x, old_y]
            # Set the old spot to empty (0)
            self.grid[old_x, old_y] = 0


#Simmulating - 
# 1. Configuration (Matching your original parameters)
N = 50                 # Grid Size
P_EMPTY = 0.2          # Fraction of Empty Cells
T = 0.4                # Baseline Similarity Threshold
BIAS = 0.2             # Tolerance Bias Factor
STEPS = 50             # Number of Simulation Steps

# 2. Initialization
model = SchellingModel(
    N=N,
    p_empty=P_EMPTY,
    T=T,
    bias_factor=BIAS
)

satisfaction_history = []

# 3. Run the model and collect data
for step_index in range(STEPS):
    # Execute one time step
    model.step()
    satisfaction_history.append(model.fraction_satisfied())


# Plot the satisfaction data over time
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(satisfaction_history, color='#4CAF50', linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Fraction Satisfied")
plt.title(f"Satisfaction Over Time (Bias: {BIAS})", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

#final grid state
plt.subplot(1, 2, 2)

# Using a clear color map 
# (0=empty, 1=Group A, 2=Group B)
cmap = plt.cm.get_cmap('coolwarm', 3) 
plt.imshow(model.grid, cmap=cmap, vmin=0, vmax=2)
plt.title("Final Grid State", fontsize=14)
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"Simulation Complete. Final Fraction Satisfied: {satisfaction_history[-1]:.4f}")
print(f"Tolerance Bias Factor used: {BIAS}")
