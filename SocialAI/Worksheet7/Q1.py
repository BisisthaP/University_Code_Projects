#Student Name - Bisistha Patra 
#ID - 24159091 

import numpy as np
import matplotlib.pyplot as plt

# The function takes:
# - f: the derivative function dy/dt = f(t, y), where y is a list or array.
# - x0: initial conditions as a list or array.
# - t_array: array of time points.
def euler_forward_scalar(f, x0, t_array):
    x = np.zeros(len(t_array)) # Initialize solution array - [0., 0., ..., 0.]
    x[0] = x0 # Set initial condition - very first value - either 1,2 or 3 (from the practical)
    for i in range(1, len(t_array)): #loop over time steps 
        dt = t_array[i] - t_array[i-1] #time difference 
        x[i] = x[i-1] + dt * f(t_array[i-1], x[i-1]) # x_new = x_old + (slope) × (time_step) - main formula 
    return x #return the new solution array 

#example 1 - from the lab worksheet
def deriv1(t, x):
    return (2 - x)

#working manual example from the lab worksheet
dt_lab = 1.0
t_lab  = np.array([0.0, 1.0, 2.0, 3.0, 4.0])   # t0 to t4 → gives x0, x1, x2, x3, x4

initial_con = [1.0, 2.0, 3.0]

plt.figure(figsize=(10, 6))

#call the manual fucntion for each initial condition and plot the results
for x0 in initial_con:
    solution = euler_forward_scalar(deriv1, x0, t_lab)
    
    print("")
    print("Starting from x0 =", x0)
    for i in range(len(t_lab)):
        print(f"  t = {t_lab[i]:.1f}  -->  x ~= {solution[i]:.6f}")
    
    # Plot
    plt.plot(t_lab, solution, 'o-', label=f'x₀ = {x0}')

plt.title('Euler method with ∆t = 1 (exactly as done by hand in lab)')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.grid(True)
plt.show()

#exponential decay example
def deriv2(t, x):
    return -x

#exponential growth example 
def deriv3(t, x):
    return x

