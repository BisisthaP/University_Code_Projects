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
    if y == 0 and x < grid_size - 1:
        x += 1
    elif x == grid_size - 1 and y < grid_size - 1:
        y += 1
    elif y == grid_size - 1 and x > 0:
        x -= 1
    elif x == 0 and y > 0:
        y -= 1
    return wrap(x), wrap(y)

def move_diagonal_ccw(x, y):
    # Moves diagonally in a diamond shape (counter-clockwise)
    moves = [(-1, -1), (-1, 1), (1, 1), (1, -1)] #such list indexes start from 0 
    index = (x + y) % 4 #remainder 
    dx, dy = moves[index]
    return wrap(x + dx), wrap(y + dy)

def move_left(x, y):
    # Always moves left
    return wrap(x - 1), y

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
        new_x, new_y = self.move_func(*self.pos)
        self.pos = np.array([new_x, new_y])
        self.step += 1
        return tuple(self.pos)


agents = [
    Agent("A1", (0, 1), move_clockwise, start_delay=0),
    Agent("A2", (0, 1), move_diagonal_ccw, start_delay=2),
    Agent("A3", (0, 1), move_left, start_delay=4)
]
#all agents start at (0,1) which is the co-ordinates for S 

for t in range(time_steps):
    print(f"\n Time Step {t + 1}")
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)] 
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

print("\nSimulation complete. Observe repeating patterns or stable conflicts.")
