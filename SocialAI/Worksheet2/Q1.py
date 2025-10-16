#StudentID = 24159091 
#Studentname = "Bisistha Patra" 

import random 

#part 1 - implementing a simple agent class for initialization and movement defination of all agents
class Agent:
    def __init__(self, answer:bool):
        self.x = 0
        self.y = 0 
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
