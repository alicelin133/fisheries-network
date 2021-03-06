"""Loads data from changing_payoff_discount.py, and plots these data in
3D as several curves of effort vs. delta, each curve representing one
of several payoff_discount values, which forms the third axis."""

import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
from matplotlib import cm
import matplotlib.colors as mcolors
plt.switch_backend('Qt5Agg')

import Simulation_arrays

def get_e_msr_nash(sim):
    """Takes a Simulation_arrays *sim* as argument and calculates value
    of e_msr (maximum sustainable rent) and e_nash (nash equilibrium)
    levels of effort. Returns the pair of values e_msr, e_nash
    in that order.
    WARNING: if any of the values r, n_fishers, q, K, price, or cost change,
    this function is NOT GOOD because now you have different values of e_msr
    and e_nash depending on the simulation."""
    r = sim.r
    price = sim.price
    q = sim.q
    K = sim.K
    n_fishers = sim.n_fishers
    cost = sim.cost
    e_msr = r * (price * q * K * n_fishers - n_fishers * cost) / \
            (2 * price * q * q * K * n_fishers)
    e_nash = e_msr * 2 * n_fishers / (1 + n_fishers)
    return e_msr, e_nash

def cc_lite(arg):
    """Borrowed from an example from matplotlib"""
    return mcolors.to_rgba(arg, alpha=0.2)

if __name__ == '__main__':
    start_time = time.time()

    # Loading numpy arrays
    path = "/Users/alicelin/Documents/fish/fisheries-network/data/"
    deltas = np.load(path + "deltas-seed15-changing_payoff_discount.npy")
    e_vals = np.load(path + "e_vals-seed15-changing_payoff_discount.npy")
    payoff_discounts = np.load(path + "payoff_discounts-seed15-changing_payoff_discount.npy")
    print(e_vals)
    
    # Calculating e_msr, e_nash
    with open(path + "sim-seed15-changing_payoff_discount.file", 'rb') as f:
        sim = pickle.load(f)
    e_msr, e_nash = get_e_msr_nash(sim)

    # Plotting stuff
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = deltas
    ys = e_vals
    zs = payoff_discounts

    verts = []
    for i in range(payoff_discounts.size):
        verts.append(list(zip(xs, ys[i])))
    
    # Picking colors to use as lines
    cmap = cm.get_cmap('Spectral')
    # Normalize the payoff_discounts by dividing by its max value
    color_vals = np.divide(payoff_discounts, np.amax(payoff_discounts))
    rgba = cmap(color_vals)

    # Plotting all the effort vs. delta curves
    lines = LineCollection(verts, colors=rgba, linewidth=1)
    ax.add_collection3d(lines, zs=zs, zdir='y')

    # Creating planes for e_msr and e_nash
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
    ax.set_yticks(payoff_discounts)
    ax.set_xticks(deltas)
    ax.set_xticklabels(deltas, rotation=30) 

    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()
