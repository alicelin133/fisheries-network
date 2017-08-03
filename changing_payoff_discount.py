"""Runs simulations, varying payoff_discount parameter, and plots
one effort vs. delta plot per value of payoff_discount in a 3D plot."""

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

def cc_lite(arg):
    """Borrowed from an example from matplotlib"""
    return mcolors.to_rgba(arg, alpha=0.2)

if __name__ == '__main__':
    start_time = time.time()

    # Assign parameter values for simulations
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
    num_steps = 1000

    # Assign values for *payoff_discount*
    payoff_discounts = np.linspace(0.04, 0.06, 5, endpoint=True)

    # Assign values for *delta*
    deltas = np.linspace(0, 0.5, 11, endpoint=True)

    # Create simulations
    e_vals = np.zeros((payoff_discounts.size, deltas.size))
    for i in range(payoff_discounts.size):
        for j in range(deltas.size):
            sim = Simulation_arrays.Simulation_arrays(n_fishers, deltas[j],
                  q, r, K, R_0, e_0, price, cost, noise, num_feedback,
                  payoff_discounts[i], num_steps)
            sim.simulate()
            e_vals[i][j] = np.mean(sim.e_data[:,100:])
            print("{}th simulation done of {}!".format(i * payoff_discounts.size + j,
                payoff_discounts.size * deltas.size))

    # Plotting stuff
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = deltas
    ys = e_vals
    zs = payoff_discounts

    verts = []
    for i in range(payoff_discounts.size):
        verts.append(list(zip(xs, ys[i])))
    
    lines = LineCollection(verts, linewidth=1)
    ax.add_collection3d(lines, zs=zs, zdir='y')

    plane = [(deltas[0], payoff_discounts[0]), (deltas[0] + 0.0001, payoff_discounts[-1]),
             (deltas[-1] - 0.001, payoff_discounts[-1]), (deltas[-1], payoff_discounts[0])]
    horizs = [plane, plane]
    planes = PolyCollection(horizs, facecolors=[cc_lite('b'), cc_lite('b')])
    ax.add_collection3d(planes, zs=[e_msr, e_nash], zdir='z')

    ax.set_xlabel("Delta")
    ax.set_ylabel("Payoff discount")
    ax.set_zlabel("Asymptotic effort level")
    ax.set_xlim3d(deltas[0], deltas[-1])
    ax.set_ylim3d(payoff_discounts[0], payoff_discounts[-1])
    ax.set_zlim3d(e_msr * 0.8, e_nash * 1.2)

    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()