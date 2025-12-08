#name - Bisistha Patra 
#Student ID - 24159091 

import random
import csv 
import matplotlib.pyplot as plt

#WORLD 
#for task1, I choose the world to wrap around - this will avoid edge cases (like going out of bounds) 
#and maintain a uniform stimulation 

grid_size = 50 #given in the task 
MAX_STEPS = 1000 #maximum steps for the simulation

#common class for both Prey and Predator (base class)
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

#both prey and predator can sense 3 cells in all directions, except for prey in diagonals 
def sense_pos(animal, others, dirs):
    sensed = set()
    #using sets to handle duplicates automatically and store the coordinated of another animal is detected 
    for dx, dy in dirs: #searching through all directions ((0,1), (1,0), (0,-1), (-1,0))
        for d in range(1, 4): #upper bound was 4 = 3 + 1
            #here, d will check against 1, 2, and 3 cells away in the current direction (dx, dy)
            sx = (animal.x + dx * d) % grid_size #dividing for wrap around 
            sy = (animal.y + dy * d) % grid_size #calculates the x and y coordinates of the sensed position 
            for other in others: # if the current agent has the same coordinates as the sensed position, 
                                 #add it to the sensed set
                if other.x == sx and other.y == sy:
                    sensed.add((sx,sy))
    return list(sensed) #list used by movement functions for the next step 

#max = maximum step size
#max for prey = 1 and for predator = 2 (given for task 1)
def get_deltas(max):
    deltas = [(dx,dy) for dx in [-1,0,1] for dy in [-1,0,1] if (dx,dy) != (0,0)] #left, right, up, down, and diagonals
    #condition of (dx,dy) != (0,0) to avoid no movement 
    if max == 2: #for predator
        deltas += [(dx*2,dy*2) for dx in [-1,0,1] for dy in [-1,0,1] if (dx,dy) != (0,0)]
    return deltas
    #adds 8 base 1 based movements and 8 additional 2 based movements for predator

def move_prey(prey, pred):
    dirs = [(0,1), (1,0), (0,-1), (-1,0)] #prey can only sense in 4 directions
    sensed = sense_pos(prey, pred, dirs) #defined above 
    deltas = get_deltas(1) #prey can move 1 step - including diagoonals 
    if not sensed: #if no predators are sensed - list is empty 
        dx, dy = random.choice(deltas) #random movement for preys (defined in task 1)
    else: #predator (atleast 1) is sensed
        best_d = [] #empty list to store the best move 
        max_min_d = -1 #stores the highest min distance from any predator
        for d in deltas: #loops through the 8 possible movements
            nx = (prey.y + d[0]) % grid_size #calculates new x position if the current direction = d is choosen
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


#logic similar to prey movement but chase instead of escape strategy
def move_pred(pred, prey):    
    dirs = [(0,1), (1,0), (0,-1), (-1,0)] #sense in all directions + diagonals 
    sensed = sense_pos(pred, prey, dirs) #defined above 
    deltas = get_deltas(2) #predator can move 2 steps - including diagoonals 
    if not sensed: #if no preys are sensed - list is empty 
        dx, dy = random.choice(deltas) #random movement for predators (defined in task 1)
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

    #eating prey if on the same cell 
    at_pos = [p for p in prey if p.x == pred.x and p.y == pred.y]
    if at_pos:
        eaten = random.choice(at_pos) #randomly choose one prey to eat if multiple are on the same cell
        prey.remove(eaten) #remove the eaten prey from the prey list
        pred.energy += 12 #increase predator energy by 12 (task 1 deined)
    
    #die if energy <= 0
    if pred.energy <= 0:
        pred.remove(pred) #indicates predator has died

#using one fucntion for both prey and predator reproduction - only difference is the conditions (if statements)
def reproduce(animal, agents, is_prey):
    deltas = get_deltas(1) #both prey and predator reproduce to adjacent cells only
    delta = random.choice(deltas) #randomly choose a direction to reproduce
    nx = (animal.x + delta[0]) % grid_size
    ny = (animal.y + delta[1]) % grid_size #new position for the child
    if is_prey:
        if random.random() < 0.15: #15% chance of reproduction for prey
            agents.append(Prey(nx, ny)) #new prey
    else:
        if animal.energy >= 12: 
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

def run_stimulation(steps, run_id, prey_start, pred_start):
    global preys, preds
    preys, preds = setup(prey_start, pred_start)
    fn = f'sim_set{steps}_run{run_id}.csv'
    with open(fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'prey_count', 'predator_count'])
        w.writerow([0, len(preys), len(preds)])
        for step in range(1, MAX_STEPS + 1):
            #movement phase
            for prey in preys:
                move_prey(prey, preds)
            for pred in preds[:]: #copy of the list to avoid modification during iteration
                move_pred(pred, preys)
            
            #reproduction phase
            for prey in preys[:]: 
                reproduce(prey, preys, True)
            for pred in preds[:]: 
                reproduce(pred, preds, False) 
            
            #write the counts to the csv file
            w.writerow([step, len(preys), len(preds)])
            if not preys or not preds:
                break #end simulation if either preys or predators are extinct - 0 count

if __name__ == "__main__":
    pops = [(200, 20), (100, 40)]  # Choices explained in report
    for s_idx, (prey_s, pred_s) in enumerate(pops, 1):
        for r_idx in range(1, 4):
            run_stimulation(s_idx, r_idx, prey_s, pred_s)
    
    # Plotting section
    for s_idx in range(1, 3):
        for r_idx in range(1, 4):
            fn = f'sim_set{s_idx}_run{r_idx}.csv'
            read_file = open(fn)
            csv_reader = csv.reader(read_file)
            next(csv_reader)  #skip header
            step_list = []
            prey_list = []
            pred_list = []
            for row in csv_reader:
                step_list.append(int(row[0]))
                prey_list.append(int(row[1]))
                pred_list.append(int(row[2]))
            read_file.close()
            
            # Plot similar to prof's example - provided in the assignment files 
            combined_list = prey_list + pred_list
            only_figure, (left_subplot, right_subplot) = plt.subplots(1, 2, figsize=(8, 4))
            left_subplot.plot(combined_list, linestyle='--')
            right_subplot.scatter(step_list, prey_list, color='blue', marker='^', label='Prey')
            right_subplot.scatter(step_list, pred_list, color='red', marker='*', label='Predator')
            right_subplot.legend()
            plt.savefig(f'plot_set{s_idx}_run{r_idx}.png')  # Save for report
            plt.close()  #avoid overlap

    






