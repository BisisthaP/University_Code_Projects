import random
import matplotlib.pyplot as plt
import numpy as np

locs = 10
n = 100000
treasure_loc = 0

class Observer:
    def __init__(self):
        # Initialize the visit counts to 0 for all 10 locations
        self.visits = [0] * locs

    def update(self, loc):
        # Increment the visit count for the given location
        self.visits[loc] += 1

    def get_visits(self):
        # Return the current visit counts for all locations
        return self.visits


class BayesianHunter:
    def __init__(self, obs):
        # Initial belief of the agent: equal probability for all 10 locations
        self.probabilities = [1.0 / locs] * locs
        self.observer = obs  # Observer object to track visits

    def reset_agent(self):
        # Resets the agent's beliefs to a uniform distribution (for a new agent/successful hunt).
        self.probabilities = [1.0 / locs] * locs

    def updateEmptyLocation(self, loc):
        # Set the probability of the empty location to 0
        self.probabilities[loc] = 0.0

        # Normalize the remaining probabilities to ensure they add up to 1
        total_prob = sum(self.probabilities)

        if total_prob > 0:
            # Normalize the probabilities for the remaining locations
            new_prob = [p / total_prob for p in self.probabilities]
            self.probabilities = new_prob

    def wheretogo(self):
        # Find the location with the highest probability
        max_prob = max(self.probabilities)
        
        # Collect all locations tied for the highest probability
        most_likely_locations = []
        for i, p in enumerate(self.probabilities):
            # Checking for equality with tolerance due to floating point math
            if abs(p - max_prob) < 1e-9: 
                most_likely_locations.append(i)
                
        # Random choice for the tied location
        chosen_location = random.choice(most_likely_locations)

        self.observer.update(chosen_location)
        return chosen_location


class NonBayesianHunter:
    def __init__(self, obs):
        # Initial belief of the agent: equal probability for all 10 locations
        self.probabilities = [1.0 / locs] * locs
        self.observer = obs  # Observer object to track visits

    def reset_agent(self):
        # Resets the agent's beliefs to a uniform distribution
        self.probabilities = [1.0 / locs] * locs

    def updateEmptyLocation(self, loc):
        # NO UPDATE: This agent ignores the search result, simulating a non-Bayesian/random search
        pass

    def wheretogo(self):
        # Find the location with the highest probability (which is always a tie initially)
        max_prob = max(self.probabilities)
        
        # Collect all locations tied for the highest probability
        most_likely_locations = []
        for i, p in enumerate(self.probabilities):
            if abs(p - max_prob) < 1e-9: 
                most_likely_locations.append(i)
                
        # Random choice for the tied location (results in random selection every turn)
        chosen_location = random.choice(most_likely_locations)

        self.observer.update(chosen_location)
        return chosen_location


def simulation(AgentClass, n=n, treasure_loc=treasure_loc):
    # Initializes observer and agent for a single simulation run
    obs = Observer()
    agent = AgentClass(obs)

    treasure_location = treasure_loc

    # Run the simulation for the specified number of turns (n)
    for j in range(n):
        next_loc = agent.wheretogo()

        if next_loc == treasure_location:
            # Case 1: Treasure found. Agent retires, new agent replaces them (reset beliefs)
            agent.reset_agent()
        else:
            # Case 2: Location was empty. Update beliefs based on model (or do nothing for NonBayesian)
            agent.updateEmptyLocation(next_loc)

    # Return the visit counts and the total number of turns
    return obs.get_visits(), n


def plot_comparison(bayesian_counts, non_bayesian_counts, total_turns, true_loc):
    # Convert counts to probabilities
    bayesian_probs = np.array(bayesian_counts) / total_turns
    non_bayesian_probs = np.array(non_bayesian_counts) / total_turns
    
    locations = np.arange(1, locs + 1)
    
    colors = ['blue'] * locs
    colors[true_loc] = 'red' 

    # Create two subplots side-by-side for comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: With Bayesian Update (Beliefs Updated)
    axes[0].bar(locations, bayesian_probs, color=colors)
    axes[0].set_xlabel('Location')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Figure 3a: With Belief Update (Approx. Bayesian)')
    axes[0].set_ylim(0, 0.25)
    axes[0].set_xticks(locations)
    
    for i, p in enumerate(bayesian_probs):
        axes[0].text(locations[i], p + 0.005, f"{p:.3f}", ha='center', fontsize=10)
        
    # Plot 2: Without Bayesian Update (Random Search)
    axes[1].bar(locations, non_bayesian_probs, color=colors)
    axes[1].set_xlabel('Location')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Figure 3b: Without Belief Update (Random Search)')
    axes[1].set_ylim(0, 0.25)
    axes[1].set_xticks(locations)
    
    for i, p in enumerate(non_bayesian_probs):
        axes[1].text(locations[i], p + 0.005, f"{p:.3f}", ha='center', fontsize=10)
        
    fig.suptitle('Comparison of Search Distributions (Treasure fixed at Location 1)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


bayesian_counts, total_turns = simulation(BayesianHunter)
non_bayesian_counts, _ = simulation(NonBayesianHunter)

plot_comparison(bayesian_counts, non_bayesian_counts, total_turns, treasure_loc)



#The simulation with belief updates results in a higher probability of visiting the treasure location 
#(~0.18) compared to the empty locations (~0.09), showing the effectiveness of the strategy.
#The simulation without belief updates results in a uniform probability (~0.10) for all locations, 
#as the agent is essentially choosing randomly every turn and ignoring search results.
#The comparison clearly demonstrates that updating beliefs (even simplistically) significantly 
#improves the agent's expected success rate over a purely random search.
