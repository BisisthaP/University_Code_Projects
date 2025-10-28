import random
import matplotlib.pyplot as plt

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


class TreasureHunters:
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


def simulation_q2(n=n, treasure_loc=treasure_loc):
    obs = Observer()
    agent = TreasureHunters(obs)

    treasure_location = treasure_loc

    for j in range(n):
        next_loc = agent.wheretogo()

        if next_loc == treasure_location:
            agent.reset_agent()
        else:
            agent.updateEmptyLocation(next_loc)

    return obs.get_visits(), n


def plot_probabilities(visit_counts, total_turns, true_loc):
    probabilities = [v / total_turns for v in visit_counts]
    
    locations = [str(i + 1) for i in range(locs)]
    
    colors = ['blue'] * locs
    colors[true_loc] = 'red' 

    plt.figure(figsize=(10, 6))
    plt.bar(locations, probabilities, color=colors)
    plt.xlabel('Location')
    plt.ylabel('Probability')
    plt.title('Figure 2: Observed Visit Probability (Treasure fixed at Location 1)')
    plt.ylim(0, 0.25)
    
    for i, p in enumerate(probabilities):
        plt.text(i, p + 0.005, f"{p:.3f}", ha='center', fontsize=10)
        
    plt.show()

visit_counts, total_turns = simulation_q2()

probabilities = [v / total_turns for v in visit_counts]

plot_probabilities(visit_counts, total_turns, treasure_loc)

#This simulation approximates the expected probability of an agent visiting each location 
#when the treasure is permanently fixed at Location 1 (index 0).
#The probabilities show that the treasure location (Location 1) has a higher expected visit 
#probability (~0.18) than the empty locations (~0.09). This is because the treasure location
#is always one of the *possible* choices, whereas the empty locations are systematically 
#eliminated by an agent over time, lowering their expected visit count across all turns.
