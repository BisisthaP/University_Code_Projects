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




