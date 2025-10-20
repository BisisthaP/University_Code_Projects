#Name - Bisistha Patra 
#Student ID - 24159091

#Question 1 
#part 1 
import random 
class Agent:
    def __init__(self, is_leader: bool):
        self.x = random.randint(-50,50)
        self.y = random.randint(-50,50)
        self.dx = 0 
        self.dy = 0 
        self.is_leader = is_leader #used to identify the leader agent 
        #true for leader, false for follower

    #part 2 
    def move(self, dt: float):
        self.x += (self.dx * dt) #following - distance = velocity * time 
        self.y += (self.dy * dt)
    
#part 4 
def change_velocity_lead(A: Agent, max_chg):
    delta_vx = random.uniform(-max_chg, max_chg) #change in the velocity x direction 
    delta_vy = random.uniform(-max_chg, max_chg) #change in the velocity y direction

    A.dx += delta_vx #updating the agent's velocity 
    A.dy += delta_vy

#part 5
def change_velocity_follower(lead: Agent, follower: Agent, rate: float):
    diff_x = lead.x - follower.x 
    diff_y = lead.y - follower.y  
    #to create the pull towards the leader, we need to find the difference
    
    #main formula:
    #new velocity = old velocity + rate*(diff btw leader and follower position)
    follower.dx += diff_x * rate 
    follower.dy += diff_y * rate

#additional function to calculate the distance between two agents
def distance(A,B : Agent):
    return ((A.x - B.x)**2 + (A.y - B.y)**2)**0.5 #Euclidean distance formula

#part 6 
def simulate(chg,rate):
    leader = Agent(True) #initialise the leader 
    followers = [Agent(False) for x in range(9)] #initialise the 9 followers 

    print("Initial Positions")
    print(f"Leader Position: ({leader.x:.2f}, {leader.y:.2f})")
    for i in range(9):
        print(f"Follower {i+1} Position: ({followers[i].x:.2f}, {followers[i].y:.2f})")
    print("")

    print('Simulating...')
    for i in range(100):
        change_velocity_lead(leader,chg) #changing the leader velocity randomly
        leader.move(1.0) #updating the leader position

        for follower in followers:
            change_velocity_follower(leader, follower, rate) #updating follower velocity
            follower.move(1.0) #updating follower position
    
    if (i + 1) % 10 == 0:
            print("")
            print(f"Step {i+1}: Leader Pos: ({leader.x:.2f}, {leader.y:.2f})")
            for j in range(9):
                print(f"  Follower {j+1}: ({followers[j].x:.2f}, {followers[j].y:.2f})")




