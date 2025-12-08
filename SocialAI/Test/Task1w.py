# Name - Bisistha Patra
# Student ID - 24159091
# Task 1: Implementing a Predator-Prey simulation

import random
import csv
import matplotlib.pyplot as plt

# World: 50x50 grid that wraps around (toroidal) to avoid edge effects.
GRID_SIZE = 50
SENSE_DEPTH = 3
REPRO_PREY_CHANCE = 0.15
REPRO_PRED_ENERGY = 12
EAT_ENERGY = 5
INITIAL_PRED_ENERGY = 10  # Starting energy for predators
SPAWN_PRED_ENERGY = 5  # Energy for new predators
MAX_STEPS = 1000  # Max simulation steps

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Prey(Agent):
    pass

class Predator(Agent):
    def __init__(self, x, y, energy):
        super().__init__(x, y)
        self.energy = energy

def dist(x1, y1, x2, y2):  # Chebyshev distance
    return max(abs(x1 - x2), abs(y1 - y2))

def sense_positions(agent, others, dirs):  # Sense in cardinal directions
    sensed = set()
    for dx, dy in dirs:
        for d in range(1, SENSE_DEPTH + 1):
            sx = (agent.x + dx * d) % GRID_SIZE
            sy = (agent.y + dy * d) % GRID_SIZE
            for o in others:
                if o.x == sx and o.y == sy:
                    sensed.add((sx, sy))
                    break
    return list(sensed)

def get_deltas(max_steps):  # Possible moves: 1 or 2 steps in 8 dirs
    deltas = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]
    if max_steps == 2:
        deltas += [(2*dx, 2*dy) for dx, dy in deltas if (2*dx, 2*dy) != (0, 0)]
    return deltas

def move_prey(prey, predators):
    dirs = [(0,1),(0,-1),(1,0),(-1,0)]  # Cardinal
    sensed = sense_positions(prey, predators, dirs)
    deltas = get_deltas(1)
    if not sensed:
        delta = random.choice(deltas)
    else:
        best_d = []
        max_min_d = -1
        for d in deltas:
            nx = (prey.x + d[0]) % GRID_SIZE
            ny = (prey.y + d[1]) % GRID_SIZE
            min_d = min(dist(nx, ny, px, py) for px, py in sensed)
            if min_d > max_min_d:
                max_min_d = min_d
                best_d = [d]
            elif min_d == max_min_d:
                best_d.append(d)
        delta = random.choice(best_d)
    prey.x = (prey.x + delta[0]) % GRID_SIZE
    prey.y = (prey.y + delta[1]) % GRID_SIZE

def move_predator(pred, preys):
    dirs = [(0,1),(0,-1),(1,0),(-1,0)]
    sensed = sense_positions(pred, preys, dirs)
    deltas = get_deltas(2)
    if not sensed:
        delta = random.choice(deltas)
    else:
        dists = [dist(pred.x, pred.y, px, py) for px, py in sensed]
        min_d = min(dists)
        closest = [p for p, d in zip(sensed, dists) if d == min_d]
        target = random.choice(closest)
        best_d = []
        min_after = float('inf')
        for d in deltas:
            nx = (pred.x + d[0]) % GRID_SIZE
            ny = (pred.y + d[1]) % GRID_SIZE
            after = dist(nx, ny, target[0], target[1])
            if after < min_after:
                min_after = after
                best_d = [d]
            elif after == min_after:
                best_d.append(d)
        delta = random.choice(best_d)
    cost = max(abs(delta[0]), abs(delta[1]))
    pred.x = (pred.x + delta[0]) % GRID_SIZE
    pred.y = (pred.y + delta[1]) % GRID_SIZE
    pred.energy -= cost
    # Eat
    at_pos = [p for p in preys if p.x == pred.x and p.y == pred.y]
    if at_pos:
        eaten = random.choice(at_pos)
        preys.remove(eaten)
        pred.energy += EAT_ENERGY
    # Die if <=0
    if pred.energy <= 0:
        predators.remove(pred)

def reproduce(agent, agents, is_prey):
    deltas = get_deltas(1)
    delta = random.choice(deltas)
    nx = (agent.x + delta[0]) % GRID_SIZE
    ny = (agent.y + delta[1]) % GRID_SIZE
    if is_prey:
        if random.random() < REPRO_PREY_CHANCE:
            agents.append(Prey(nx, ny))
    else:
        if agent.energy >= REPRO_PRED_ENERGY:
            agents.append(Predator(nx, ny, SPAWN_PRED_ENERGY))

def setup(prey_num, pred_num):
    occupied = set()
    preys = []
    predators = []
    for _ in range(prey_num + pred_num):
        while True:
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            if (x, y) not in occupied:
                occupied.add((x, y))
                if len(preys) < prey_num:
                    preys.append(Prey(x, y))
                else:
                    predators.append(Predator(x, y, INITIAL_PRED_ENERGY))
                break
    return preys, predators

def run_sim(set_idx, run_idx, prey_start, pred_start):
    global preys, predators
    preys, predators = setup(prey_start, pred_start)
    fn = f'sim_set{set_idx}_run{run_idx}.csv'
    with open(fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'prey_count', 'predator_count'])
        w.writerow([0, len(preys), len(predators)])
        for step in range(1, MAX_STEPS + 1):
            # Move phase
            for prey in preys[:]:
                move_prey(prey, predators)
            for pred in predators[:]:
                move_predator(pred, preys)
            # Reproduce phase
            for prey in preys[:]:
                reproduce(prey, preys, True)
            for pred in predators[:]:
                reproduce(pred, predators, False)
            # Write
            w.writerow([step, len(preys), len(predators)])
            if not preys or not predators:
                break

if __name__ == "__main__":
    pops = [(200, 20), (100, 40)]  # Choices explained in report
    for s_idx, (prey_s, pred_s) in enumerate(pops, 1):
        for r_idx in range(1, 4):
            run_sim(s_idx, r_idx, prey_s, pred_s)
    
    # Plotting section
    for s_idx in range(1, 3):
        for r_idx in range(1, 4):
            fn = f'sim_set{s_idx}_run{r_idx}.csv'
            read_file = open(fn)
            csv_reader = csv.reader(read_file)
            next(csv_reader)  # Skip header
            step_list = []
            prey_list = []
            pred_list = []
            for row in csv_reader:
                step_list.append(int(row[0]))
                prey_list.append(int(row[1]))
                pred_list.append(int(row[2]))
            read_file.close()
            
            # Plot similar to prof's example
            combined_list = prey_list + pred_list
            only_figure, (left_subplot, right_subplot) = plt.subplots(1, 2, figsize=(8, 4))
            left_subplot.plot(combined_list, linestyle='--')
            right_subplot.scatter(step_list, prey_list, color='blue', marker='^', label='Prey')
            right_subplot.scatter(step_list, pred_list, color='red', marker='*', label='Predator')
            right_subplot.legend()
            plt.savefig(f'plot_set{s_idx}_run{r_idx}.png')  # Save for report
            plt.close()  # Close to avoid overlap