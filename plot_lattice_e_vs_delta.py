"""Loads e and deltas array from lattice_e_vs_delta.py, processes e,
then plots and saves the average e value vs. delta using matplotlib."""

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import time
plt.switch_backend('Qt5Agg')

import lattice_e_vs_delta # TODO: use this to calculate e_msr, e_nash and plot these as hlines

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
    e_raw = np.load("/Users/alicelin/Documents/fish/fisheries-network" + \
        "/data/e-lattice-4x4-num_deltas21-deltas0_0to0_1-q1-r0_05-K20" + \
        "0-price1-cost0_5-noise0_0001-R_0100_0-num_feedback25-num_ste" + \
        "ps1000-v0.npy")
    R_raw = np.load("/Users/alicelin/Documents/fish/fisheries-network" + \
        "/data/R-lattice-4x4-num_deltas21-deltas0_0to0_1-q1-r0_05-K20" + \
        "0-price1-cost0_5-noise0_0001-R_0100_0-num_feedback25-num_ste" + \
        "ps1000-v0.npy")
    deltas = np.load("/Users/alicelin/Documents/fish/fisheries-networ" + \
        "k/data/deltas-lattice-4x4-num_deltas21-deltas0_0to0_1-q1-r0_" + \
        "05-K200-price1-cost0_5-noise0_0001-R_0100_0-num_feedback25-n" + \
        "um_steps1000-v0.npy")
    es = get_vals(e_raw)
    Rs = get_vals(R_raw)
    
    # Making figure
    plt.style.use('bmh')
    rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # Plotting effort vs. delta
    ax1.plot(deltas, es, 'b', marker='o', markevery=1, markersize=5, linewidth=0.5)
    ax1.set_xticks(np.linspace(deltas[0],deltas[-1],11))
    ax1.set_xlabel("$\delta$")
    ax1.set_ylabel("Equilibrium effort", color='b')
    ax1.set_title("Effort vs. $\delta$ on a 4-by-4 lattice")
    ax1.tick_params('y', colors='b')

    # Plotting resource level vs. delta
    ax2 = ax1.twinx()
    ax2.plot(deltas, Rs, 'r', marker='o', markevery=1, markersize=5, linewidth=0.5)
    ax2.set_ylabel("Equilibrium resource level", color='r')
    ax2.tick_params('y', colors='r')

    plt.show()
    
if __name__ == '__main__':
    main()