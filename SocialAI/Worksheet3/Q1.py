import random
import matplotlib.pyplot as plt

class Observer:
    def __init__(self):
        # Initialize the visit counts to 0 for all 10 locations
        self.visits = [0] * 10

    def update(self, loc):
        # Increment the visit count for the given location
        self.visits[loc] += 1

    def get_visits(self):
        # Return the current visit counts for all locations
        return self.visits


class TreasureHunters:
    def __init__(self, obs):
        # Initial belief of the agent: equal probability for all 10 locations
        self.probabilities = [0.1] * 10
        self.observer = obs  # Observer object to track visits

    def reset_agent(self):
        # Resets the agent's beliefs to a uniform distribution (for a new agent).
        self.probabilities = [0.1] * 10

    def updateEmptyLocation(self, loc):
        # Set the probability of the empty location to 0
        self.probabilities[loc] = 0

        # Normalize the remaining probabilities to ensure they add up to 1
        total_prob = sum(self.probabilities)

        if total_prob > 0:  # Avoid division by 0
            # Normalize the probabilities for the remaining locations
            new_prob = [p / total_prob for p in self.probabilities]
            self.probabilities = new_prob

    def wheretogo(self):
        # Find the location with the highest probability
        max_prob = max(self.probabilities)
        
        # Collect all locations tied for the highest probability
        most_likely_locations = []
        for i, p in enumerate(self.probabilities):
            if p == max_prob:
                most_likely_locations.append(i)
                
        #random choice for the tied location
        chosen_location = random.choice(most_likely_locations)

        self.observer.update(chosen_location)
        return chosen_location


def simulation(n=1000):
    # Initialize the Observer and TreasureHunter objects
    obs = Observer()
    agent = TreasureHunters(obs)

    # Randomly assign a treasure location
    treasure_location = random.randint(0, 9)
    print("True treasure location:", treasure_location) # Added for clarity

    for j in range(n):
        next_loc = agent.wheretogo()

        if next_loc == treasure_location:
            # Case 1: Treasure found. Agent retires, new agent replaces them
            agent.reset_agent()
        else:
            # Case 2: Location was empty. Update beliefs.
            agent.updateEmptyLocation(next_loc)

    # Return the visit counts from the observer
    return obs.get_visits(), treasure_location


visit_counts, true_loc = simulation(n=1000)
print("Visit counts per location:", visit_counts)


# Plotting function to visualize the visit counts
def plot_visit_counts(visit_counts, true_loc):
    # Plot a bar chart of the visit counts for each location
    locations = [str(i) for i in range(10)]
    
    colors = ['blue'] * 10
    colors[true_loc] = 'red' 

    plt.bar(locations, visit_counts, color=colors)
    plt.xlabel('Location')
    plt.ylabel('Visit Count')
    plt.title('Figure 1: Number of Visits per Location')
    
    for i, v in enumerate(visit_counts):
        plt.text(i, v + 5, str(v), ha='center', fontsize=9)
        
    plt.show()

# Plot the results
plot_visit_counts(visit_counts, true_loc)