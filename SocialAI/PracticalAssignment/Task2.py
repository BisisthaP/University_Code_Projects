#name - Bisistha Patra 
#Student ID - 24159091

#Task 2 
import random 
import csv 
import matplotlib.pyplot as plt

grid_size = 50 #given in the task 
MAX_STEPS = 1000 #maximum steps for the simulation

class Animal:
    def __init__(self, x, y):
        self.x = x #horizontal position
        self.y = y #vertical position 

#both Prey and Predators will inherit from the Animal base class 
class Prey(Animal):
    pass

#difference between preys and predators is the energy levels 
class Predator(Animal):
    #e = energy level
    def __init__(self, x, y, e):
        super().__init__(x, y) #call the base class constructor
        self.energy = e

#to keep the cost for moving diagonally the same as moving horizontally or vertically,
# we will use chessboard distance (Chebyshev distance)
def dist(x1,y1,x2,y2):
    return max(abs(x1 - x2), abs(y1 - y2))
    #max of the absolute differences in x and y coordinates 

#all of this is same as task 1 - no changes made (code lines above this comment)

#both prey and predator can sense 3 cells in all directions, except for prey in diagonals 
# but here will add a parameter sense_dis to indicate the max sensing distance
def sense_pos(animal, others, dirs, sense_dis):
    sensed = set()
    #using sets to handle duplicates automatically and store the coordinated of another animal is detected 
    for dx, dy in dirs: #searching through all directions ((0,1), (1,0), (0,-1), (-1,0))
        for d in range(1, sense_dis + 1): #upper bound is changed to sense_dis 
            #here, d will check against 1, 2, and 3 cells away in the current direction (dx, dy)
            sx = (animal.x + dx * d) % grid_size #dividing for wrap around 
            sy = (animal.y + dy * d) % grid_size #calculates the x and y coordinates of the sensed position 
            for other in others: # if the current agent has the same coordinates as the sensed position, 
                                 #add it to the sensed set
                if other.x == sx and other.y == sy:
                    sensed.add((sx,sy))
    return list(sensed) #list used by movement functions for the next step 
#so the only change here is the addition of sense_dis parameter

#max = maximum step size
#max for prey = 1 and for predator = 2 (given for task 1)
def get_deltas(max):
    deltas = [] #chaged - empty list to store possible movements
    for step in range(1, max + 1): #loop through step sizes from 1 to max (inclusive)
        deltas.extend([dx * step, dy * step] for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0))
        #line explaination - two for loops used to generate all combinations of dx and dy for the current step 
        #dx and dy can be -1, 0, or 1 (for negative, no movement, or positive direction)
        #the if condition excludes the case where both dx and dy are 0 (no movement)    
    return deltas
#changes made - 
#1. created an empty list deltas to store possible movements
#2. used extend method to add all possible (dx, dy) pairs for each step size to the deltas list

#add sense and move distance parameters for prey movement function
def move_prey(prey, pred, sense_dist_prey, move_dist_prey):
    dirs = [(0,1), (1,0), (0,-1), (-1,0)] #prey can only sense in 4 directions
    sensed = sense_pos(prey, pred, dirs, sense_dist_prey) #defined above 
    deltas = get_deltas(move_dist_prey) #changed - custom movement distance for prey

    if not sensed: #if no predators are sensed - list is empty 
        delta = random.choice(deltas) #random movement for preys (defined in task 1)
    else: #predator (atleast 1) is sensed
        best_d = [] #empty list to store the best move 
        max_min_d = -1 #stores the highest min distance from any predator
        for d in deltas: #loops through the 8 possible movements
            nx = (prey.x + d[0]) % grid_size #calculates new x position if the current direction = d is choosen
            ny = (prey.y + d[1]) % grid_size 
            min_d = min([dist(nx, ny, sx, sy) for sx, sy in sensed])
            if min_d > max_min_d: #current move better than best move so far - escape strategy
                max_min_d = min_d #replace and update 
                best_d = [d]
            elif min_d == max_min_d: #current move = same as best move so far - add to list best_d 
                best_d.append(d)
        delta = random.choice(best_d) #best is choosen randomly from the list of best moves 

    prey.x = (prey.x + delta[0]) % grid_size
    prey.y = (prey.y + delta[1]) % grid_size #new position after movement for the prey 

def move_pred(pred, prey, sense_dist_pred, move_dist_pred):   
    dirs = [(0,1), (1,0), (0,-1), (-1,0)] #sense in all directions + diagonals 
    sensed = sense_pos(pred, prey, dirs, sense_dist_pred) #defined above - changed to include sense distance 
    deltas = get_deltas(move_dist_pred) #changed - custom movement distance for predator just like prey 

    if not sensed: #if no preys are sensed - list is empty 
        delta = random.choice(deltas) #random movement for predators (defined in task 1)
    else: #prey (atleast 1) is sensed
        #logic that is different from prey movement - predator targets the closest prey
        dists = [dist(pred.x, pred.y, sx, sy) for sx, sy in sensed]
        min_d = min(dists)
        closest = [i for i, d in zip(sensed, dists) if d == min_d]
        target = random.choice(closest) 
        #randomly choose one of the closest preys if multiple are at same distance

        best_d = [] #empty list to store the best move 
        min_after = float('inf') #stores the lowest min distance from any prey
        for d in deltas: #loops through the 16 possible movements
            nx = (pred.x + d[0]) % grid_size #calculates new x position if the current direction = d is choosen
            ny = (pred.y + d[1]) % grid_size 
            after = dist(nx, ny, target[0], target[1])
            if after < min_after: #current move better than best move so far - chase strategy
                min_after = after #replace and update 
                best_d = [d]
            elif after == min_after: #current move = same as best move so far - add to list best_d 
                best_d.append(d)
        delta = random.choice(best_d) #best is choosen randomly from the list of best moves 

    cost = max(abs(delta[0]), abs(delta[1])) #energy cost of movement = max step size taken
    pred.x = (pred.x + delta[0]) % grid_size
    pred.y = (pred.y + delta[1]) % grid_size #new position after movement for the predator
    pred.energy -= cost #reduce energy by cost of movement (defined in task 1)

    #eating prey if on the same cell - changes made from the logic of task 1 
    at_pos = [p for p in prey if p.x == pred.x and p.y == pred.y]
    if at_pos:
        eaten = random.choice(at_pos) #randomly choose one prey to eat if multiple are on the same cell
        prey.remove(eaten) #remove the eaten prey from the prey list
        pred.energy += 5 #increase predator energy by 5 (task 1 deined)
    
    return pred.energy > 0 #return True if predator is alive after eating prey, else False
    # returning boolean instead of removing predator from the list here

def reproduce(animal, agents, is_prey, rep_prey_prob, rep_pred_energy):
    deltas = get_deltas(1) #both prey and predator reproduce to adjacent cells only
    delta = random.choice(deltas) #randomly choose a direction to reproduce
    nx = (animal.x + delta[0]) % grid_size
    ny = (animal.y + delta[1]) % grid_size #new position for the child
    
    #changes made - different conditions for prey and predator reproduction
    if is_prey and random.random() < rep_prey_prob:
        agents.append(Prey(nx, ny)) #new prey
    elif not is_prey and animal.energy >= rep_pred_energy:
        agents.append(Predator(nx, ny, 5)) #new predator with initial energy of 5

def setup(prey_count, pred_count):
    occupied = set() #to avoid placing multiple animals on the same cell initially
    preys = []
    preds = []
    for _ in range(prey_count + pred_count):
        while True:
            x = random.randint(0, grid_size - 1)
            y = random.randint(0, grid_size - 1)
            if (x,y) not in occupied:
                occupied.add((x,y))
                if len(preys) < prey_count:
                    preys.append(Prey(x, y))#initial prey placement
                else:
                    preds.append(Predator(x, y, 5)) #initial predator placement with energy 5
                break
    return preys, preds

def run_simulation(set_id, run_id, prey_start, pred_start, params):
    preys, preds = setup(prey_start, pred_start)
    fn = f'task2_sim_set{set_id}_run{run_id}.csv'
    with open(fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'prey_count', 'predator_count'])
        w.writerow([0, len(preys), len(preds)])
        for step in range(1, MAX_STEPS + 1):
            for prey in preys:
                move_prey(prey, preds, params['sense_prey'], params['move_prey'])
            preds = [pred for pred in preds if move_pred(pred, preys, params['sense_pred'], params['move_pred'])]
            for prey in preys[:]:
                reproduce(prey, preys, True, params['repro_prey'], params['repro_pred_energy'])
            for pred in preds[:]:
                reproduce(pred, preds, False, params['repro_prey'], params['repro_pred_energy'])
            w.writerow([step, len(preys), len(preds)])
            if not preys or not preds:
                break

def plot_simulation(set_id, run_id):
    fn = f'task2_sim_set{set_id}_run{run_id}.csv'
    with open(fn, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        data = [(int(row[0]), int(row[1]), int(row[2])) for row in csv_reader]
    steps, preys, preds = zip(*data)
    combined = list(preys) + list(preds)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.plot(combined, linestyle='--')
    ax2.scatter(steps, preys, color='blue', marker='^', label='Prey')
    ax2.scatter(steps, preds, color='red', marker='*', label='Predator')
    ax2.legend()
    plt.savefig(f'task2_plot_set{set_id}_run{run_id}.png')
    plt.close()

if __name__ == "__main__":
    param_sets = [
        {'repro_prey': 0.10, 'repro_pred_energy': 15, 'sense_prey': 4, 'sense_pred': 2, 'move_prey': 2, 'move_pred': 1},
        {'repro_prey': 0.20, 'repro_pred_energy': 10, 'sense_prey': 2, 'sense_pred': 4, 'move_prey': 1, 'move_pred': 3}
    ]
    prey_start, pred_start = 200, 20
    for set_id, params in enumerate(param_sets, 1):
        for run_id in range(1, 4):
            run_simulation(set_id, run_id, prey_start, pred_start, params)
            plot_simulation(set_id, run_id)