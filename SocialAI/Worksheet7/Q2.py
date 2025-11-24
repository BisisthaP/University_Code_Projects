import matplotlib.pyplot as plt
import numpy as np

def euler_forward_vector(f, y0, t_array, *args):
    n_steps = len(t_array)
    n_vars  = len(y0)
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    for i in range(1, n_steps):
        dt = t_array[i] - t_array[i-1]
        y[i] = y[i-1] + dt * f(t_array[i-1], y[i-1], *args)
    return y

# SIR model from the slides 
# S + i + R = 1 at all times
# ds/dt = -β S i
# di/dt = β S i - γ i
# dr/dt = γ i

def sir_slide_model(t, y, beta, gamma):
    S, i, R = y                    # i is lowercase to match the slide!
    dSdt = -beta * S * i
    didt =  beta * S * i - gamma * i
    dRdt =  gamma * i
    return np.array([dSdt, didt, dRdt])

# Parameters (typical values that match the shape in your slide)
beta  = 0.5      # transmission rate (larger because we work in proportions)
gamma = 0.1      # recovery rate → average infectious period = 10 days

# Initial conditions in proportions (exactly like the slide plot)
S0 = 0.999
i0 = 0.001       # one infected person in a population of ~1000
R0 = 0.0
y0 = np.array([S0, i0, R0])

# Time array
dt = 0.2
t  = np.arange(0, 100.1, dt)   # long enough to see full epidemic

# Solve the system
solution = euler_forward_vector(sir_slide_model, y0, t, beta, gamma)

S = solution[:, 0]
i = solution[:, 1]
R = solution[:, 2]


# 1. Tabulate results (every 10 days)
print("SIR Model (exactly as in lecture slide) - Proportions")
print("Day\tS\t\ti\t\tR")
print("-" * 45)
for day in range(0, len(t), 50):        # 50 steps × 0.2 = 10 days
    print(f"{t[day]:.1f}\t{S[day]:.4f}\t{i[day]:.4f}\t{R[day]:.4f}")


# 2. Plot - identical style to the slide
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible (S)', color='blue')
plt.plot(t, i, label='Infected (i)',     color='orange')
plt.plot(t, R, label='Recovered (R)',    color='green')

plt.title('SIR Model - Exactly as Shown in Lecture Slide', fontsize=14)
plt.xlabel('Time')
plt.ylabel('Proportion of Population')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.xlim(0, 100)