#Question 1 implementation - 
import numpy as np 

grid_size = 3
num_agents = 3 
time_steps = 50

def wrap(pos):
    return pos%grid_size  
    #uses the remainder operator for the wrapping around the grid 
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
#all agents start at (0,1) which is the co-ordinates for S, as per the question  

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

#Question 2 implementation -
import numpy as np
import random 

GRID_SIZE = 5
TIME_STEPS = 100
NUM_SCENARIOS = 3

def is_valid(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def move_up(x, y):
    return x, y - 1

def move_down_right(x, y):
    return x + 1, y + 1

def move_horizontal_cycle(x, y):
    return x + 1, y

def move_random(x, y):
    dx, dy = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    #move right = (1,0)
    #move left = (-1,0)
    #move up = (0,-1)
    #move down = (0,1)
    return x + dx, y + dy #change in the directions of x and y 

class Agent:
    def __init__(self, name, pos, move_func, direction=1):
        self.name = name
        if pos is not None: #if starting position is valid, set it to a numpy array and it should use integers
            self.pos = np.array(pos, dtype=int)
        else:
            self.pos = None       #if no starting position - set it to Nonen 
        self.move_func = move_func
        self.is_active = pos is not None
        self.direction = direction

    def get_move_coords(self):
        # If the agent isn't active (it's "dead"), it can't calculate a move
        if not self.is_active:
            return None
        
        x, y = self.pos
        
        #calculate the next position based on the the current stored directions (+1, -1)
        if self.move_func == move_horizontal_cycle:
            return x + self.direction, y
        else:
            return self.move_func(x, y)

    def update_state(self, new_pos):
        # Check if the target position is off the grid 
        if not is_valid(new_pos):
            self.is_active = False #agent is no longer active 
            self.pos = None #position is none 
            return False
        
        self.pos = np.array(new_pos)
        
        if self.move_func == move_horizontal_cycle:
            if self.pos[0] == GRID_SIZE - 1 and self.direction == 1: #hitting the right edge 
                self.direction = -1
            elif self.pos[0] == 0 and self.direction == -1: #hitting the left edge
                self.direction = 1
                
        return True

def run_simulation(agents_list, scenario_id):
    # Filter the initial list to make sure we only have active agents to start
    active_agents = [a for a in agents_list if a.is_active] 
    print(f"\n--- Scenario {scenario_id} Start (Grid Size: {GRID_SIZE}x{GRID_SIZE}) ---")
    
    #handles the common steps once an agent successfully finds its spot
    def move_into_target(agent, target_pos, current_pos, grid_map, moved_agents_this_turn):
        agent.update_state(target_pos) # Final position update
        moved_agents_this_turn.add(agent.name)
        del grid_map[current_pos] # Clear the old spot
        grid_map[target_pos] = agent # Claim the new spot

    for t in range(TIME_STEPS):
        # Setup for the current step (T+1)
        move_order = [a for a in active_agents if a.is_active] # Get agents still alive
        grid_map = {tuple(a.pos): a for a in active_agents if a.is_active} # Map position -> agent
        moved_agents_this_turn = set() # Track who has moved to prevent cheating
        
        for agent in move_order:
            # Skip if dead or if already pushed/moved this turn
            if not agent.is_active or agent.name in moved_agents_this_turn:
                continue
            
            current_pos = tuple(agent.pos)
            target_pos = agent.get_move_coords() # Where the agent WANTS to go
            
            # SCENARIO 1: INVALID AGENT (Moving off the grid)
            if not is_valid(target_pos):
                agent.update_state(target_pos) # This step sets is_active = False
                moved_agents_this_turn.add(agent.name)
                if current_pos in grid_map:
                    del grid_map[current_pos]
                active_agents.remove(agent) # Agent removed from the game list
                continue

            # SCENARIO 2: CONFLICT 
            if target_pos in grid_map:
                pushed_agent = grid_map[target_pos] # The agent in the way
                
                # Calculate where the pushed agent would land 
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
                pushed_target = (target_pos[0] + dx, target_pos[1] + dy)
                
                # Check A: Deadlock? (Spot behind is blocked too)
                if pushed_target in grid_map:
                    continue # None of the agents will move this turn
                
                # Check B: Attempt the Push
                push_successful = pushed_agent.update_state(pushed_target)
                
                if not push_successful:
                    # Pushed agent died as it was - pushed off the edge
                    active_agents.remove(pushed_agent)
                else:
                    # Pushed agent moved 
                    moved_agents_this_turn.add(pushed_agent.name)
                    del grid_map[target_pos] # Clear original spot
                    grid_map[pushed_target] = pushed_agent # Update map for pushed agent
                
                # Now the primary agent can move into the vacated spot
                move_into_target(agent, target_pos, current_pos, grid_map, moved_agents_this_turn)

            # SCENARIO 3: FREE MOVE (Target spot is empty)
            else:
                move_into_target(agent, target_pos, current_pos, grid_map, moved_agents_this_turn)

        # Final list creation to remove any dead agents 
        new_active_agents = [] 
        for agent in active_agents:
            if agent.is_active: # Only keep agents that are still active
                new_active_agents.append(agent)
                
        active_agents = new_active_agents # Replace list for the next time step

    print(f"\nSimulation finished for Scenario {scenario_id}.")
    remaining = len(active_agents)
    print(f"Agents remaining after {TIME_STEPS} steps: {remaining}")
    return remaining

def create_agents(start_states):
    s = start_states 
    return [
        Agent("A1", s[0], move_up),
        Agent("A2", s[1], move_down_right),
        Agent("A3", s[2], move_horizontal_cycle),
        Agent("A4", s[3], move_random)
    ]

SCENARIOS = {
    1: [(2, 4), (0, 0), (2, 2), (4, 4)], 
    2: [(0, 4), (0, 3), (0, 2), (0, 1)], 
    3: [(2, 2), (3, 2), (4, 2), (0, 0)]  
}

results = {}
for i in range(1, NUM_SCENARIOS + 1):
    start_states = SCENARIOS[i]
    agents = create_agents(start_states)
    remaining_agents = run_simulation(agents, i)
    results[f"Scenario {i}"] = remaining_agents

print("--- Summary of Results ---")
for scenario, count in results.items():
    print(f"{scenario}: {count} agents remaining")

#agents remaining after 100 steps: Scenario 1: 2,
# Scenario 2: 1, 
# Scenario 3: 3

#why ?
#The random (A4) and down-right (A2) movers are often the first to be lost, 
# frequently falling off the boundary for scenario 1

#The agents start close together, leading to early pushing and 
# a brief period of instability before A1 (Up) quickly moves out of the crowd's path for scenarion 2

#For scenario 3, the configuration provided enough space that all agents survived the 100 steps. 
#No death causing pushes or boundary exits occurred within the time limit.
