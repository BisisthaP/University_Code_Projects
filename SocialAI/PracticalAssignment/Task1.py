#name - Bisistha Patra 
#Student ID - 24159091 

import random
import csv 

#WORLD 
#for task1, I choose the world to wrap around - this will avoid edge cases (like going out of bounds) 
#and maintain a uniform stimulation 

grid_size = 50 #given in the task 

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
    deltas = get_deltas(1) #prey can move 1 step
    if not sensed: #if no predators are sensed 
        dx, dy = random.choice(deltas) #random movement for preys (defined in task 1)
    else:
        best_d = []
        max_min_d = -1 
        for d in deltas:
            nx = (prey.y + d[0]) % grid_size
            ny = (prey.y + d[1]) % grid_size
            min_d = min([dist(nx, ny, sx, sy) for sx, sy in sensed])
            if min_d > max_min_d:
                max_min_d = min_d
                best_d = [d]
            elif min_d == max_min_d:
                best_d.append(d)
        delta = random.choice(best_d)
    prey.x = (prey.x + delta[0]) % grid_size
    prey.y = (prey.y + delta[1]) % grid_size








