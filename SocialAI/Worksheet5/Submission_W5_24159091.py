import numpy 
import matplotlib.pyplot as plt
import random 

class ShellModel:
    def __init__(self, grid_size, fraction_empty_cells, required_similarity, tolerance_bias_factor):
        self.grid_size = grid_size                            
        self.fraction_empty_cells = fraction_empty_cells              #empty cells represented as fractions
        self.required_similarity = required_similarity                # Minimum required similarity to be happy (baseline)
        self.tolerance_bias_factor = tolerance_bias_factor            # Factor controlling how much diversity increases tolerance

        # --- Grid Initialization ---
        # Define possible cell states and their initial probabilities
        cells = [0, 1, 2]
        probs = [fraction_empty_cells, (1 - fraction_empty_cells) / 2, (1 - fraction_empty_cells) / 2]
        # Create the initial random grid
        self.grid = np.random.choice(cells, size=(grid_size, grid_size), p=probs)

    def _get_similarity(self, x, y):
        """
        Helper method to calculate the fractional similarity of a cell's neighbors.
        
        Args:
            x (int): Row index of the cell.
            y (int): Column index of the cell.
            
        Returns:
            float: The fraction of neighbors belonging to the same group. 
                   Returns 1 if no occupied neighbors are present.
        """
        agent_group_type = self.grid[x, y]
        
        # An empty cell has no preference/similarity to measure
        if agent_group_type == 0:
            return 0.0

        neighboring_cells = []
        # Check all 8 surrounding neighbors (using toroidal/wrap-around boundaries)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if not (i == 0 and j == 0):
                    # Calculate neighbor coordinates with wrap-around (% grid_size)
                    nx, ny = (x + i) % self.grid_size, (y + j) % self.grid_size
                    
                    # Only consider occupied neighbors (Group A or B)
                    if self.grid[nx, ny] != 0:
                        neighboring_cells.append(self.grid[nx, ny])

        if len(neighboring_cells) == 0:
            # If no occupied neighbors, the agent is maximally happy (vacuously true)
            return 1.0

        # Count neighbors belonging to the same group as the current cell
        similar_neighbors_count = sum(1 for val in neighboring_cells if val == agent_group_type)
        return similar_neighbors_count / len(neighboring_cells)

    def _is_happy(self, x, y, local_similarity):
        """
        Checks if the agent at (x, y) is happy based on the dynamic threshold.
        """
        # Diversity is the opposite of similarity
        local_diversity = 1.0 - local_similarity
        
        # Calculate the effective threshold: a high diversity lowers the requirement
        # for similarity, making the agent more tolerant.
        effective_threshold = self.required_similarity - self.tolerance_bias_factor * local_diversity
        
        return local_similarity >= effective_threshold

    def fraction_satisfied(self):
        """
        Calculates the current fraction of agents who are satisfied/happy.
        """
        happy_count = 0
        total_agents = 0
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y] != 0:
                    total_agents += 1
                    local_similarity = self._get_similarity(x, y)
                    if self._is_happy(x, y, local_similarity):
                        happy_count += 1
                        
        # Return the fraction (guard against division by zero if no agents exist)
        return happy_count / total_agents if total_agents else 0.0

    def step(self):
        """
        Performs one simulation step: identifies unhappy agents and moves them 
        to randomly selected empty cells.
        """
        unhappy_agents_locations = []
        
        # 1. Identify all unhappy agents
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                agent_group_type = self.grid[x, y]
                
                if agent_group_type != 0:
                    local_similarity = self._get_similarity(x, y)
                    
                    # Check for unhappiness using the dynamic threshold
                    if not self._is_happy(x, y, local_similarity):
                        unhappy_agents_locations.append((x, y))

        # 2. Find empty cells and shuffle lists to randomize movement order
        available_empty_spots = list(zip(*np.where(self.grid == 0)))
        random.shuffle(available_empty_spots)
        random.shuffle(unhappy_agents_locations)
        
        # 3. Move agents (max agents moved is min(unhappy, empty_cells))
        for (old_x, old_y), (new_x, new_y) in zip(unhappy_agents_locations, available_empty_spots):
            # Move the agent to the new, empty spot
            self.grid[new_x, new_y] = self.grid[old_x, old_y]
            # Set the old spot to empty (0)
            self.grid[old_x, old_y] = 0


# --- Simulation Execution ---

# 1. Configuration (Matching your original parameters)
SIM_GRID_SIZE = 50
SIM_FRACTION_EMPTY = 0.2
SIM_REQUIRED_SIMILARITY = 0.4
SIM_TOLERANCE_FACTOR = 0.2 # The core modification parameter
NUM_SIMULATION_STEPS = 50

# 2. Initialization
current_model_instance = SchellingModel(
    grid_size=SIM_GRID_SIZE,
    fraction_empty_cells=SIM_FRACTION_EMPTY,
    required_similarity=SIM_REQUIRED_SIMILARITY,
    tolerance_bias_factor=SIM_TOLERANCE_FACTOR
)

satisfaction_levels_history = []

# 3. Run the model and collect data
for step_index in range(NUM_SIMULATION_STEPS):
    # Execute one time step
    current_model_instance.step()
    # Record the fraction of satisfied agents
    satisfaction_levels_history.append(current_model_instance.fraction_satisfied())


# --- Plotting and Visualization ---

# Plot the satisfaction data over time
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(satisfaction_levels_history, color='#4CAF50', linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Fraction Satisfied")
plt.title(f"Satisfaction Over Time (Tolerance Factor: {SIM_TOLERANCE_FACTOR})", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# Show the final grid state
plt.subplot(1, 2, 2)
# Using a clear color map (0=empty, 1=Group A, 2=Group B)
cmap = plt.cm.get_cmap('coolwarm', 3) 
plt.imshow(current_model_instance.grid, cmap=cmap, vmin=0, vmax=2)
plt.title("Final Grid State", fontsize=14)
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"Simulation Complete. Final Fraction Satisfied: {satisfaction_levels_history[-1]:.4f}")
print(f"Tolerance Bias Factor used: {SIM_TOLERANCE_FACTOR}")

