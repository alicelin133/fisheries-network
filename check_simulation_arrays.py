"""To look at what Simulation_arrays.py is doing. Is it reaching the e_msr
and e_nash values for delta=0 and delta=1 cases, because it doesn't seem to
be doing so right now."""

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
plt.switch_backend('Qt5Agg')

import Simulation_arrays

def calculate_e_msr(n_fishers, q, r, K, price, cost):
    """Calculates value of e_msr (maximum sustainable rent)."""
    return r * (price * q * K * n_fishers - n_fishers * cost) / (2 * price * q * q * K * n_fishers)

def calculate_e_nash(e_msr, n_fishers):
    """Calculates value of Nash equilibrium level of effort."""
    return e_msr * 2 * n_fishers / (1 + n_fishers)

def e_avg_nodes(sim):
    """Returns average curve of *sim*'s e_data vs time. Averages
    across all nodes and returns an array of e values over time."""
    e_avg = np.mean(sim.e_data, axis=0)
    return e_avg

if __name__ == '__main__':
    start_time = time.time()

    # Assign other parameter values for simulations
    n_fishers = 20
    q = 1
    r = 0.05
    K = 200
    R_0 = np.full(n_fishers, K/2)
    e_0 = np.linspace(0, r/q, num=n_fishers)
    price = 1
    cost = 0.5
    noise = 0.0001
    num_feedback = 50
    payoff_discount = 0.5
    num_steps = 1000

    deltas = np.linspace(0,1,10,endpoint=True)

    e_vals = np.zeros((deltas.size, num_steps))
    for i in range(deltas.size):
        sim = Simulation_arrays.Simulation_arrays(n_fishers, deltas[i], q, r, K, R_0,
           e_0, price, cost, noise, num_feedback, payoff_discount, num_steps)
        sim.simulate()
        e_vals[i] = e_avg_nodes(sim)

    colors = np.linspace(0,1,deltas.size,endpoint=False)

    for i in range(deltas.size):
        plt.plot(np.arange(num_steps), e_vals[i], color=(1, colors[i], colors[i]))

    e_msr = calculate_e_msr(n_fishers, q, r, K, price, cost)
    e_nash = calculate_e_nash(e_msr, n_fishers)
    
    plt.axhline(y=e_msr, ls='--', c='c', linewidth=1)
    plt.axhline(y=e_nash, ls='--', c='m', linewidth=1)

    plt.text(0.1, e_msr, "$e_{MSR}$", va='bottom')
    plt.text(0.1, e_nash, "$e_{Nash}$", va='top')

    plt.xlabel("Time step")
    plt.ylabel("Effort")
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()