"""To look at what Simulation_arrays.py is doing. Is it reaching the e_msr
and e_nash values for delta=0 and delta=1 cases, because it doesn't seem to
be doing so right now."""

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
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

def cc(arg):
    """Borrowed from an example from matplotlib"""
    return mcolors.to_rgba(arg, alpha=0.3)

def cc_lite(arg):
    """Borrowed from an example from matplotlib"""
    return mcolors.to_rgba(arg, alpha=0.2)

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

    # Assign delta values
    deltas = np.linspace(0, 0.1, 10, endpoint=True)

    # Run simulations at different delta values to determine effort values
    e_vals = np.zeros((deltas.size, num_steps))
    for i in range(deltas.size):
        sim = Simulation_arrays.Simulation_arrays(n_fishers, deltas[i], q, r, K, R_0,
           e_0, price, cost, noise, num_feedback, payoff_discount, num_steps)
        sim.simulate()
        e_vals[i] = e_avg_nodes(sim)

    # Calculate analytical e_MSR and e_Nash effort levels
    e_msr = calculate_e_msr(n_fishers, q, r, K, price, cost)
    e_nash = calculate_e_nash(e_msr, n_fishers)

    # Plotting stuff
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = np.arange(num_steps)
    ys = e_vals
    ys[:, 0] = e_msr * 0.8 # ensures face color extends to bottom of plot
    ys[:, -1] = e_msr * 0.8
    zs = deltas
    verts = []
    for i in range(zs.size):
        verts.append(list(zip(xs, ys[i])))
    
    colors = [cc('r') for i in range(deltas.size)]
    lines = LineCollection(verts, linewidth=0.5, facecolors=colors)
    ax.add_collection3d(lines, zs=zs, zdir='y')

    # Making horizontal planes for the e_nash and e_msr effort levels
    plane = [(0,deltas[0]), (0.0001, deltas[-1]),
             (num_steps - 0.001, deltas[-1]), (num_steps, deltas[0])]
    horizs = [plane, plane]
    planes = PolyCollection(horizs, facecolors=[cc_lite('b'), cc_lite('b')])
    ax.add_collection3d(planes, zs=[e_msr, e_nash], zdir='z')

    ax.set_xlabel("Time step")
    ax.set_ylabel("Deltas")
    ax.set_zlabel("Effort Level")
    ax.set_xlim3d(0, num_steps)
    ax.set_ylim3d(0,deltas[-1])
    ax.set_zlim3d(e_msr * 0.8, e_nash * 1.2)

    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()