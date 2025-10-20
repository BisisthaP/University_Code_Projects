# # Name - Bisistha Patra 
# # Student ID - 24159091

# # Question 1 
# # part 1 
# import random 
# class Agent:
#     def __init__(self, is_leader: bool):
#         self.x = random.randint(-50,50)
#         self.y = random.randint(-50,50)
#         self.dx = 0 
#         self.dy = 0 
#         self.is_leader = is_leader #used to identify the leader agent 
#         #true for leader, false for follower

#     #part 2 
#     def move(self, dt: float):
#         self.x += (self.dx * dt) #following - distance = velocity * time 
#         self.y += (self.dy * dt)
    
# #part 4 
# def change_velocity_lead(A: Agent, max_chg):
#     delta_vx = random.uniform(-max_chg, max_chg) #change in the velocity x direction 
#     delta_vy = random.uniform(-max_chg, max_chg) #change in the velocity y direction

#     A.dx += delta_vx #updating the agent's velocity 
#     A.dy += delta_vy

# #part 5
# def change_velocity_follower(lead: Agent, follower: Agent, rate: float):
#     diff_x = lead.x - follower.x 
#     diff_y = lead.y - follower.y  
#     #to create the pull towards the leader, we need to find the difference
    
#     #main formula:
#     #new velocity = old velocity + rate*(diff btw leader and follower position)
#     follower.dx += diff_x * rate 
#     follower.dy += diff_y * rate

# #additional function to calculate the distance between two agents
# def distance(A,B : Agent):
#     return ((A.x - B.x)**2 + (A.y - B.y)**2)**0.5 #Euclidean distance formula

# #part 6 
# def simulate(chg,rate):
#     leader = Agent(True) #initialise the leader 
#     followers = [Agent(False) for x in range(9)] #initialise the 9 followers 

#     print("Initial Positions")
#     print(f"Leader Position: ({leader.x:.2f}, {leader.y:.2f})")
#     for i in range(9):
#         print(f"Follower {i+1} Position: ({followers[i].x:.2f}, {followers[i].y:.2f})")
#     print("")

#     print('Simulating...')
#     for i in range(100):
#         change_velocity_lead(leader,chg) #changing the leader velocity randomly
#         leader.move(1.0) #updating the leader position

#         for follower in followers:
#             change_velocity_follower(leader, follower, rate) #updating follower velocity
#             follower.move(1.0) #updating follower position
    
#     if (i + 1) % 10 == 0:
#             print("")
#             print(f"Step {i+1}: Leader Pos: ({leader.x:.2f}, {leader.y:.2f})")
#             for j in range(9):
#                 print(f"  Follower {j+1}: ({followers[j].x:.2f}, {followers[j].y:.2f})")

#     print("")
#     print("Simulation is complete")
#     print(f"Final Leader Position: ({leader.x:.2f}, {leader.y:.2f})")
#     for j in range(9):
#         print(f"  Follower {j+1} Final positions:  ({followers[j].x:.2f}, {followers[j].y:.2f})")
    
#     print("")
#     print("Most distant follower from the leader:")
#     max_dist = -1 
#     for j in range(9):
#         follower = followers[j]
#         dist = distance(leader, follower)
#         if dist > max_dist:
#             max_dist = dist 
#             agent_num = j + 1

#     print(f"Follower {agent_num} and leader have the max distance of {max_dist:.4f}")

# #main code running and testing
# chg = 2.0 #max change in leader velocity
# rate = 0.05 #learning rate for followers
# simulate(chg, rate)

#Question 2 

#part 1 
def function(x:int):
    return 3 * (x**2)

#part 2
def derivative(x:int):
    return 6 * x

#part 3 
def cal_delta(alpha, gradient):
    return - (alpha * gradient)

#part 4
def threshold_check(gradient, threshold):
    return abs(gradient) < threshold

#part 5 
def gradient_descent(x_init, alpha, theta, max_iters):

    #x_init = starting point 
    #alpha = learning rate
    #theta = threshold for gradient check
    x = x_init
    iterations = 0 

    while iterations < max_iters: 
        #loop will run until iterations = 1000 or gradient check is satisfied - minimum gradient is reached 

        #calculation of the current values of y and gradient
        curr_val = function(x)
        gradient = derivative(x)

        print(f"Iteration {iterations}: x = {x:.6f}, f(x) = {curr_val:.6f}, f'(x) = {gradient:.6f}")

        if threshold_check(gradient, theta):
            print(f"Converged after {iterations} iterations!")
            print(f"Minimum found at x = {x:.6f}, f(x) = {curr_val:.6f}")
            return x, curr_val, iterations
        
        #update x value using the calculated delta
        delta = cal_delta(alpha, gradient)
        x += delta

        iterations += 1
    
    #if max iterations reached without convergence
    final_value = function(x)
    print(f"Reached maximum iterations ({max_iters})")
    print(f"Final result: x = {x:.6f}, f(x) = {final_value:.6f}")
    return x, final_value, iterations

