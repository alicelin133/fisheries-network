"""Loads numpy arrays from lattice_e_vs_delta.py and complete_e_vs_delta.py
and plots an e vs. delta graph so that the behavior of the two different
graph structures can be compared."""

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import time
plt.switch_backend('Qt5Agg')

import lattice_e_vs_delta

def get_vals(raw):
    """Takes a 3D numpy array *raw* and calculates the mean data values for
    each value of delta. Returns a 1D numpy array of these mean values. The
    mean is calculated by averaging across all nodes from time step 100
    until the end."""
    num_deltas = raw.shape[0] # Num 2D sheets, 1 per delta
    data = np.zeros(num_deltas)
    for i in range(num_deltas):
        data[i] = np.mean(raw[i][:,100:])
    return data

def main():
    # Get values to plot
    e_l_raw = np.load("/Users/alicelin/Documents/fish/fisheries-network" + \
        "/data/e-lattice-4x4-num_deltas21-deltas0_0to0_1-q1-r0_05-K20" + \
        "0-price1-cost0_5-noise0_0001-R_0100_0-num_feedback25-num_ste" + \
        "ps1000-v0.npy")
    R_l_raw = np.load("/Users/alicelin/Documents/fish/fisheries-network" + \
        "/data/R-lattice-4x4-num_deltas21-deltas0_0to0_1-q1-r0_05-K20" + \
        "0-price1-cost0_5-noise0_0001-R_0100_0-num_feedback25-num_ste" + \
        "ps1000-v0.npy")
    e_c_raw = np.load("/Users/alicelin/Documents/fish/fisheries-netwo" + \
        "rk/data/e-complete-n16-num_deltas21-deltas0_0to0_1-q1-r0_05-" + \
        "K200-price1-cost0_5-noise0_0001-R_0100_0-num_feedback25-num_" + \
        "steps1000-v0.npy")
    R_c_raw = np.load("/Users/alicelin/Documents/fish/fisheries-netwo" + \
        "rk/data/R-complete-n16-num_deltas21-deltas0_0to0_1-q1-r0_05-" + \
        "K200-price1-cost0_5-noise0_0001-R_0100_0-num_feedback25-num_" + \
        "steps1000-v0.npy")
    deltas = np.load("/Users/alicelin/Documents/fish/fisheries-networ" + \
        "k/data/deltas-lattice-4x4-num_deltas21-deltas0_0to0_1-q1-r0_" + \
        "05-K200-price1-cost0_5-noise0_0001-R_0100_0-num_feedback25-n" + \
        "um_steps1000-v0.npy")
    e_l = get_vals(e_l_raw)
    e_c = get_vals(e_c_raw)
    R_l = get_vals(R_l_raw)
    R_c = get_vals(R_c_raw)

    # Calculae e_msr, e_nash to plot as hlines
    e_msr, e_nash = lattice_e_vs_delta.get_e_msr_nash(16, 0.05, 1, 200, 1, 0.5)
    
    # Making figure
    plt.style.use('bmh')
    rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # Plotting effort vs. delta
    ax1.plot(deltas, e_l, color=(0.2,0.2,0.6), marker='o', markevery=1, markersize=5, linewidth=0.5, label='e, 4-by-4 Lattice')
    ax1.plot(deltas, e_c, color=(0.4,0.6,1), marker='o', markevery=1, markersize=5, linewidth=0.5, label='e, $K_{16}$')
    ax1.set_xticks(np.linspace(deltas[0],deltas[-1],11))
    ax1.set_xlabel("$\delta$")
    ax1.set_ylabel("Equilibrium effort", color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_title("Resource, effort vs. $\delta$ on a 4-by-4 lattice and $K_{16}$")
    ax1.grid(b=False)
    ax1.legend(loc=7)

    # Plotting e_msr, e_nash as hlines
    ax1.axhline(y=e_msr, linestyle='--', color='c', linewidth=0.5)
    ax1.axhline(y=e_nash, linestyle='--', color='c', linewidth=0.5)
    ax1.text(0, e_msr * 1.01, '$e_{MSR}$', va='bottom', color='c')
    ax1.text(0.09, e_nash * 0.99, '$e_{Nash}$', va='top', color='c')

    # Plotting resource level vs. delta
    ax2 = ax1.twinx()
    ax2.plot(deltas, R_l, color=(0.6,0,0.2), marker='o', markevery=1, markersize=5, linewidth=0.5, label='R, 4-by-4 Lattice')
    ax2.plot(deltas, R_c, color=(1,0.28,0.1), marker='o', markevery=1, markersize=5, linewidth=0.5, label='R, $K_{16}$')
    ax2.set_ylabel("Equilibrium resource level", color='r')
    ax2.tick_params('y', colors='r')
    ax2.grid(b=False)
    ax2.legend(loc=4)

    plt.show()
    
if __name__ == '__main__':
    main()