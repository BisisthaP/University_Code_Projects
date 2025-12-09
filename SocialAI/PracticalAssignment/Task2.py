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

def move_prey(prey, pred):
    dirs = [(0,1), (1,0), (0,-1), (-1,0)] #prey can only sense in 4 directions
    sensed = sense_pos(prey, pred, dirs) #defined above 
    deltas = get_deltas(1) #prey can move 1 step - including diagoonals 
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

