import matplotlib.pyplot as plt
import random

def simulate_list_model(init_a=50, init_b=50, steps=10):
    # Initialize population list (pop)
    pop = ['a'] * init_a + ['b'] * init_b
    
    # Track population sizes
    pop_a = [init_a]
    pop_b = [init_b]
    pop_total = [init_a + init_b]
    
    for t in range(steps):
        # Reproduction: Each agent produces one offspring of same genotype
        # The new generation (offspring) is simply a duplicate of the current pop.
        offspring = [agent for agent in pop]
        pop.extend(offspring)
        
        # Collect sizes
        N_a = pop.count('a')
        N_b = pop.count('b')
        pop_a.append(N_a)
        pop_b.append(N_b)
        pop_total.append(N_a + N_b)
        
    return pop_a, pop_b, pop_total

# Run simulation
pop_a, pop_b, pop_total = simulate_list_model(steps=15) # Increased steps for a better plot

# Plot dynamics
plt.plot(pop_a, label="N_a ('a' population)")
plt.plot(pop_b, label="N_b ('b' population)")
plt.plot(pop_total, label="Total N (Population Size)")
plt.xlabel("Time Steps (t)")
plt.ylabel("Population Size (N)")
plt.title("List of Agents Model: Unconstrained Exponential Growth")
plt.legend()
plt.show()