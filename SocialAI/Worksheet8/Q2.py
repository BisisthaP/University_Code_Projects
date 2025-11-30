import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from scipy.integrate import odeint

#UNCONSTRAINED (LIST) MODEL: ABM & DE
def list_abm(init_a=50, init_b=50, T=12):
    """ABM: Each agent produces 1 identical offspring per step (Exponential growth)"""
    pop = ['a'] * init_a + ['b'] * init_b
    N_a = [init_a]
    N_b = [init_b]
    
    for t in range(T):
        offspring = pop[:]  # Faster than list comprehension for simple copy
        pop.extend(offspring)
        N_a.append(pop.count('a'))
        N_b.append(pop.count('b'))
        
    return np.array(N_a), np.array(N_b), np.arange(T + 1)

def exp_de(N0, t, r=np.log(2)):
    """DE Solution: N(t) = N0 * exp(r*t) (r=ln(2) for doubling per step)"""
    return N0 * np.exp(r * t)