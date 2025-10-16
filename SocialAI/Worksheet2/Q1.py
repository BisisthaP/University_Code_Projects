#StudentID = 24159091 
#Studentname = "Bisistha Patra" 

import random 

#part 1 - implementing a simple agent class for initialization and movement defination of all agents
class Agent:
    def __init__(self, answer:bool):
        self.x = random.uniform(-50.0, 50.0)
        self.y = random.uniform(-50.0, 50.0)
        self.dx = 0 #velocity in x direction
        self.dy = 0 #velocity in y direction
        self.is_leader = answer 
    
    #part 2 - move function to update agent position based on a velocity 
    def move(self,dt:float):
        #main thoughts: distance = velocity * time
        self.x += (self.dx * dt) 
        self.y += (self.dy * dt)

#part 4 - function to change the velocity of the leader agent randomly within a range
def change_velocity_lead(A:Agent, max_chg):
    delta_vx = random.uniform(-max_chg, max_chg) #so in max_chg = 1, delta_vx can be any value between -1 to +1
    delta_vy = random.uniform(-max_chg, max_chg)

    A.dx += delta_vx #updating the velocity of the agent - x direction
    A.dy += delta_vy #upating in the y-direction 

#part 5 - function that updates the velocity of all the fellow agents 
#rate = learning rate 
def change_velocity_follower(lead:Agent, follower:Agent, rate:float):
    #different between the leader and follower 
    diff_x = lead.x - follower.x 
    diff_y = lead.y - follower.y 

    follower.dx += diff_x * rate 
    follower.dy += diff_y * rate
    #main formula: new velocity = old velocity + rate * (difference between leader and follower position)

def distance(A:Agent, B:Agent):
    return ((A.x - B.x)**2 + (A.y - B.y)**2)**0.5 #Euclidean distance formula - root of x*2+y*2

def simulate(chg_user, rate, time_steps):
    #can also be user defined inputs 
    # chg_user = float(input("Enter the max change in velocity: "))
    # rate = float(input("Enter the learning rate for followers: "))
    # time_steps = int(input("Enter the number of time steps to simulate: "))

    #initializing the leader and follower agents
    leader = Agent(True)
    followers = [Agent(False) for _ in range(9)]
    
    print("Initial Positions")
    print(f"Leader Position: ({leader.x:.2f}, {leader.y:.2f})")
    for i in range(9):
        print(f"Follower {i+1} Position: ({followers[i].x:.2f}, {followers[i].y:.2f})")
    print("")

    print("Simulating...")
    for i in range(time_steps):
        #changes the leader velocity randomly with the user defined range 
        change_velocity_lead(leader, chg_user)
        leader.move(1.0) #updating the leader position based on the new velocity
        #print(f"Leader Position: ({leader.x:.2f}, {leader.y:.2f})")

        #updating the follower positions based on the leader position and learning rate
        for j in range(9):
            follower = followers[j]
            change_velocity_follower(leader, follower, rate)
            follower.move(1.0)
            #print(f"Follower {j+1} Position: ({follower.x:.2f}, {follower.y:.2f})")
        
        if (i + 1) % 10 == 0:
            print("")
            print(f"Step {i+1}: Leader Pos: ({leader.x:.2f}, {leader.y:.2f})")
            for j in range(9):
                print(f"  Follower {j+1}: ({followers[j].x:.2f}, {followers[j].y:.2f})")
    
    print("")
    print("Simulation complete.")
    print("Code ran 100 iterations with 1 leader and 9 followers.")
    print("Most distance between leader and followers is")
    max_dist = 0 
    for i in range(9):
        dist = distance(leader, followers[i])
        max_dist = max(max_dist, dist)
    
    print("")
    print(f"Final Leader Position: ({leader.x:.2f}, {leader.y:.2f})")
    for j in range(9):
        print(f"  Follower {j+1} Final positions:  ({followers[j].x:.2f}, {followers[j].y:.2f})")
    print(f"Maximum distance between leader and any follower is: {max_dist:.4f}")

chg_user = 1.0
rate = 0.1
time_steps = 100
simulate(chg_user, rate, time_steps)