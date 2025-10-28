import random
import matplotlib.pyplot as plt
import numpy as np

locs = 10
n = 100000
treasure_loc = 0

class Observer:
    def __init__(self):
        self.visits = [0] * locs

    def update(self, loc):
        self.visits[loc] += 1

    def get_visits(self):
        return self.visits


class BayesianHunter:
    def __init__(self, obs):
        self.probabilities = [1.0 / locs] * locs
        self.observer = obs

    def reset_agent(self):
        self.probabilities = [1.0 / locs] * locs

    def updateEmptyLocation(self, loc):
        self.probabilities[loc] = 0.0

        total_prob = sum(self.probabilities)

        if total_prob > 0:
            new_prob = [p / total_prob for p in self.probabilities]
            self.probabilities = new_prob

    def wheretogo(self):
        max_prob = max(self.probabilities)
        
        most_likely_locations = []
        for i, p in enumerate(self.probabilities):
            if abs(p - max_prob) < 1e-9: 
                most_likely_locations.append(i)
                
        chosen_location = random.choice(most_likely_locations)

        self.observer.update(chosen_location)
        return chosen_location


class NonBayesianHunter:
    def __init__(self, obs):
        self.probabilities = [1.0 / locs] * locs
        self.observer = obs

    def reset_agent(self):
        self.probabilities = [1.0 / locs] * locs

    def updateEmptyLocation(self, loc):
        pass

    def wheretogo(self):
        max_prob = max(self.probabilities)
        
        most_likely_locations = []
        for i, p in enumerate(self.probabilities):
            if abs(p - max_prob) < 1e-9: 
                most_likely_locations.append(i)
                
        chosen_location = random.choice(most_likely_locations)

        self.observer.update(chosen_location)
        return chosen_location


def simulation(AgentClass, n=n, treasure_loc=treasure_loc):
    obs = Observer()
    agent = AgentClass(obs)

    treasure_location = treasure_loc

    for j in range(n):
        next_loc = agent.wheretogo()

        if next_loc == treasure_location:
            agent.reset_agent()
        else:
            agent.updateEmptyLocation(next_loc)

    return obs.get_visits(), n


def plot_comparison(bayesian_counts, non_bayesian_counts, total_turns, true_loc):
    bayesian_probs = np.array(bayesian_counts) / total_turns
    non_bayesian_probs = np.array(non_bayesian_counts) / total_turns
    
    locations = np.arange(1, locs + 1)
    
    colors = ['blue'] * locs
    colors[true_loc] = 'red' 

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
