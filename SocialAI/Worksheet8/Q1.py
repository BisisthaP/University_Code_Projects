import matplotlib.pyplot as plt
import random

def simulate_list_model(initial_a=50, initial_b=50, steps=10):
    # Initialize population list
    population = ['a'] * initial_a + ['b'] * initial_b
    
    # Track population sizes
    a_counts = [initial_a]
    b_counts = [initial_b]
    total_counts = [initial_a + initial_b]
    
    for step in range(steps):
        # Each agent produces one offspring of same genotype
        offspring = [agent for agent in population]
        population.extend(offspring)
        
        # Collect sizes
        num_a = population.count('a')
        num_b = population.count('b')
        a_counts.append(num_a)
        b_counts.append(num_b)
        total_counts.append(num_a + num_b)
    
    return a_counts, b_counts, total_counts

# Run simulation
a_counts, b_counts, total_counts = simulate_list_model()

# Plot dynamics
plt.plot(a_counts, label="'a' population")
plt.plot(b_counts, label="'b' population")
plt.plot(total_counts, label="Total population")
plt.xlabel("Time Steps")
plt.ylabel("Population Size")
plt.title("List of Agents Model: Exponential Growth")
plt.legend()
plt.show()