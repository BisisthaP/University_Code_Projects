# #Student Name - Bisistha Patra 
# #ID - 24159091 

# import numpy as np
# import matplotlib.pyplot as plt

# # question 1 
# # The function takes:
# # - f: the derivative function dy/dt = f(t, y), where y is a list or array.
# # - x0: initial conditions as a list or array.
# # - t_array: array of time points.
# def euler_forward_scalar(f, x0, t_array):
#     x = np.zeros(len(t_array)) # Initialize solution array - [0., 0., ..., 0.]
#     x[0] = x0 # Set initial condition - very first value - either 1,2 or 3 (from the practical)
#     for i in range(1, len(t_array)): #loop over time steps 
#         dt = t_array[i] - t_array[i-1] #time difference 
#         x[i] = x[i-1] + dt * f(t_array[i-1], x[i-1]) # x_new = x_old + (slope) × (time_step) - main formula 
#     return x #return the new solution array 

# #example 1 - from the lab worksheet
# def deriv1(t, x):
#     return (2 - x)

# #working manual example from the lab worksheet
# dt_lab = 1.0
# t_lab  = np.array([0.0, 1.0, 2.0, 3.0, 4.0])   # t0 to t4 → gives x0, x1, x2, x3, x4

# initial_con = [1.0, 2.0, 3.0]

# plt.figure(figsize=(10, 6))

# #call the manual fucntion for each initial condition and plot the results
# for x0 in initial_con:
#     solution = euler_forward_scalar(deriv1, x0, t_lab)
    
#     print("")
#     print("Starting from x0 =", x0)
#     for i in range(len(t_lab)):
#         print(f"  t = {t_lab[i]:.1f}  -->  x ~= {solution[i]:.6f}")
    
#     # Plot
#     plt.plot(t_lab, solution, 'o-', label=f'x₀ = {x0}')

# plt.title('Euler method with ∆t = 1 (exactly as done by hand in lab)')
# plt.xlabel('t')
# plt.ylabel('x(t)')
# plt.legend()
# plt.grid(True)
# plt.show()

# #exponential decay example
# def deriv2(t, x):
#     return -x

# #exponential growth example 
# def deriv3(t, x):
#     return x

# # List of DEs
# derivs = [(deriv1, 'dx/dt = 2 - x'), (deriv2, 'dx/dt = -x'), (deriv3, 'dx/dt = x')]

# #Reproduce lab hand calculations (Δt=1, t=0 to 4, x0 to x4)
# dt_lab = 1.0
# t_lab = np.arange(0.0, 5.0, dt_lab)

# plt.figure(figsize=(12, 6 * len(derivs)))

# for idx, (deriv, title) in enumerate(derivs, 1):
#     print("")
#     print("Reproducing lab results for {title}")
#     plt.subplot(len(derivs), 1, idx)
#     for x0 in initial_con:
#         solution = euler_forward_scalar(deriv, x0, t_lab)
        
#         print("")
#         print("Starting from x0 = {x0}")
#         for i in range(len(t_lab)):
#             print(f"  t = {t_lab[i]:.1f}  -->  x ~= {solution[i]:.6f}")
        
#         plt.plot(t_lab, solution, 'o-', label=f'x₀ = {x0}')
    
#     plt.title(f'Euler method with ∆t = 1 (as in lab): {title}')
#     plt.xlabel('t')
#     plt.ylabel('x(t)')
#     plt.legend()
#     plt.grid(True)

# plt.tight_layout()
# plt.show()

# # Part 2: Longer period (t=0 to 20) with smaller Δt=0.1
# dt_small = 0.1
# t_long = np.arange(0.0, 20.1, dt_small)

# plt.figure(figsize=(12, 6 * len(derivs)))

# for idx, (deriv, title) in enumerate(derivs, 1):
#     plt.subplot(len(derivs), 1, idx)
#     for x0 in initial_con:
#         solution = euler_forward_scalar(deriv, x0, t_long)
#         plt.plot(t_long, solution, '-', label=f'x₀ = {x0}')
    
#     plt.title(f'Longer trajectory with smaller ∆t = 0.1: {title}')
#     plt.xlabel('t')
#     plt.ylabel('x(t)')
#     plt.legend()
#     plt.grid(True)

# plt.tight_layout()
# plt.show()

# # Part 3: Explore larger Δt=2.0 (t=0 to 20)
# dt_large = 2.0
# t_large = np.arange(0.0, 20.1, dt_large)

# plt.figure(figsize=(12, 6 * len(derivs)))

# for idx, (deriv, title) in enumerate(derivs, 1):
#     plt.subplot(len(derivs), 1, idx)
#     for x0 in initial_con:
#         solution_large = euler_forward_scalar(deriv, x0, t_large)
#         plt.plot(t_large, solution_large, 'o--', label=f'x₀ = {x0} (∆t=2.0)')
        
#         # Reference with small dt
#         solution_small = euler_forward_scalar(deriv, x0, t_long)
#         plt.plot(t_long, solution_small, '-', label=f'x₀ = {x0} (∆t=0.1, ref)', alpha=0.5)
    
#     plt.title(f'Effect of larger ∆t = 2.0 vs 0.1: {title}')
#     plt.xlabel('t')
#     plt.ylabel('x(t)')
#     plt.legend()
#     plt.grid(True)

# plt.tight_layout()
# plt.show()

# # The above code implements the Euler method to numerically solve first-order
# # ordinary differential equations (ODEs). It demonstrates the method's behavior
# # across different initial conditions (x_0) and rate functions (dx/dt).
# # The code specifically illustrates:
# # 1. Convergence to a stable equilibrium (dx/dt = 2 - x).
# # 2. Rapid decay and stability towards zero (dx/dt = -2x).
# # 3. Exponential, unstable growth (dx/dt = x).
# # The step size (Delta t = 1.0) is consistent across all examples.

#question 2 - 

# #from question 1 
# def euler_forward_vector(f, y0, t_array, *args):
#     n_steps = len(t_array)
#     n_vars  = len(y0)
#     y = np.zeros((n_steps, n_vars))
#     y[0] = y0
#     for i in range(1, n_steps):
#         dt = t_array[i] - t_array[i-1]
#         y[i] = y[i-1] + dt * f(t_array[i-1], y[i-1], *args)
#     return y

# # SIR model from the slides 
# # S + i + R = 1 at all times
# # ds/dt = -β S i
# # di/dt = β S i - γ i
# # dr/dt = γ i

# def sir_slide_model(t, y, beta, gamma):
#     S, i, R = y                    # i is lowercase to match the slide!
#     dSdt = -beta * S * i
#     didt =  beta * S * i - gamma * i
#     dRdt =  gamma * i
#     return np.array([dSdt, didt, dRdt])

# # Parameters (typical values that match the shape in your slide)
# beta  = 0.5      # transmission rate (larger because we work in proportions)
# gamma = 0.1      # recovery rate → average infectious period = 10 days

# # Initial conditions in proportions (exactly like the slide plot)
# S0 = 0.999
# i0 = 0.001       # one infected person in a population of ~1000
# R0 = 0.0
# y0 = np.array([S0, i0, R0])

# # Time array
# dt = 0.2
# t  = np.arange(0, 100.1, dt)   # long enough to see full epidemic

# # Solve the system
# solution = euler_forward_vector(sir_slide_model, y0, t, beta, gamma)

# S = solution[:, 0]
# i = solution[:, 1]
# R = solution[:, 2]


# # 1. Tabulate results (every 10 days)
# print("SIR Model (exactly as in lecture slide) - Proportions")
# print("Day\tS\t\ti\t\tR")
# print("-" * 45)
# for day in range(0, len(t), 50):        # 50 steps × 0.2 = 10 days
#     print(f"{t[day]:.1f}\t{S[day]:.4f}\t{i[day]:.4f}\t{R[day]:.4f}")


# # 2. Plot - identical style to the slide
# plt.figure(figsize=(10, 6))
# plt.plot(t, S, label='Susceptible (S)', color='blue')
# plt.plot(t, i, label='Infected (i)',     color='orange')
# plt.plot(t, R, label='Recovered (R)',    color='green')

# plt.title('SIR Model - Exactly as Shown in Lecture Slide', fontsize=14)
# plt.xlabel('Time')
# plt.ylabel('Proportion of Population')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.ylim(0, 1)
# plt.xlim(0, 100)