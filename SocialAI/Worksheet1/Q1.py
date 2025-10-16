import numpy as np 

grid_size = 3
num_agents = 3 
time_steps = 50 

#makes sure that the agents remain within the grid boundaries
def wrap(pos):
    return pos%grid_size #remainder 

#so if pos = -1 or 3 = it will wrap around to 2 or 0 

#agent movement functions

def move_clockwise(x, y):
    #clockwise order around the border (top → right → bottom → left)
    if y == 0 and x < grid_size - 1: #top 
        x += 1
    elif x == grid_size - 1 and y < grid_size - 1: #right 
        y += 1
    elif y == grid_size - 1 and x > 0: #bottom
        x -= 1
    elif x == 0 and y > 0: #left
        y -= 1
    return wrap(x), wrap(y)

def move_diagonal_ccw(x, y):
    # Moves diagonally in a diamond shape (counter-clockwise)
    moves = [(-1, -1), (-1, 1), (1, 1), (1, -1)] #such list indexes start from 0 
    index = (x + y) % 4 #remainder when divided by 4
     #index will change the values through 0,1,2,3 as the agent moves diagonally
    dx, dy = moves[index]
    return wrap(x + dx), wrap(y + dy)

def move_left(x, y):
    # Always moves left
    return wrap(x - 1), y
    #x = move left or right 
    #y = move up or down

class Agent:
    def __init__(self, name, pos, move_func, start_delay=0):
        self.name = name
        self.pos = np.array(pos)
        self.move_func = move_func
        self.delay = start_delay
        self.step = 0

    def move(self):
        # Wait until delay is over
        if self.step < self.delay: #2<2 = false 
            self.step += 1
            return tuple(self.pos) 
        #calculation of new position using the movement function
        new_x, new_y = self.move_func(*self.pos)
        self.pos = np.array([new_x, new_y]) #update position
        self.step += 1
        return tuple(self.pos) #return as a tuple 


agents = [
    Agent("A1", (0, 1), move_clockwise, start_delay=0),
    Agent("A2", (0, 1), move_diagonal_ccw, start_delay=2),
    Agent("A3", (0, 1), move_left, start_delay=4)
]
#all agents start at (0,1) which is the co-ordinates for S 

for t in range(time_steps):
    print(f"Time Step {t + 1}")
    grid = [[ "." ] * grid_size for _ in range(grid_size)]
    #empty grid initialised with . 
    occupied = {}  # track where agents move - empty dictionaries 

    for agent in agents:
        old_pos = tuple(agent.pos)
        new_pos = agent.move()

        # Check for conflicts
        if new_pos in occupied:
            print(f"Conflict at {new_pos} between {occupied[new_pos]} and {agent.name}. Move cancelled!")
            # Cancel both moves (revert)
            agent.pos = np.array(old_pos)
            for a in agents:
                if a.name == occupied[new_pos]:
                    a.pos = np.array(old_pos)
        else:
            occupied[new_pos] = agent.name

    # Update grid
    for agent in agents:
        x, y = agent.pos
        grid[y][x] = agent.name  #y is row, x is column

    for row in grid:
        print(" ".join(row))

print("Simulation complete. Observe repeating patterns or stable conflicts.")
#pattern recognised after several time steps 
#A 4 Step repeat cycle that becomes obvious after time step 22 (from t = 23)

#there two main conflict patterns - 
#one conflict between A1 and A2 at (2,2) (Happens at t=23, 27, 31, 35, etc.)
#another conflict between A2 and A3 at (1,1) (Happens at t=25, 29, 33, 37, etc.)
#also notice that every time there is a conflict, agents are always place one below the another on the same side
#and one of the agent is off the grid 
#e.g. (2,2) - A1 is at (2,2), A2 is at (2,1) and A3 is off the grid

#reason for the conflict is that the agents are trying to move into the same cell at the same time step
#as in, during the conflict situations, both agents move back to their previous positions
#and try to move into the same cell again in the next time step, causing a repeating conflict cycle 

#endo of question 1 

