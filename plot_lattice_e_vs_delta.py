"""Loads e and deltas array from lattice_e_vs_delta.py, processes e,
then plots and saves the average e value vs. delta using matplotlib."""

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import time
plt.switch_backend('Qt5Agg')

import lattice_e_vs_delta

def get_e_vals(e_raw):
    """Takes a 3D numpy array *e_raw* and calculates the mean e value for
    each value of delta. Returns a 1D numpy array of these e values."""
    num_deltas = e_raw.shape[0] # Num 2D sheets, 1 per delta
    es = np.zeros(num_deltas)
    for i in range(num_deltas):
        es[i] = np.mean(e_raw[i][:,100:])
    return es

def main():
    e_raw = np.load("/Users/alicelin/Documents/fish/fisheries-network" + \
        "/data/e-lattice-4x4-num_deltas21-deltas0_0to1_0-q1-r0_05-K20" + \
        "0-price1-cost0_5-noise0_0001-R_0100_0-num_feedback25-num_ste" + \
        "ps1000-v0.npy")
    deltas = np.load("/Users/alicelin/Documents/fish/fisheries-networ" + \
        "k/data/deltas-lattice-4x4-num_deltas21-deltas0_0to1_0-q1-r0_" + \
        "05-K200-price1-cost0_5-noise0_0001-R_0100_0-num_feedback25-n" + \
        "um_steps1000-v0.npy")
    es = get_e_vals(e_raw)
    
    # Making figure
    plt.style.use('ggplot')
    rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(deltas, es, marker='o', markevery=1, markersize=5, linewidth=0.5)
    ax.set_xticks(np.linspace(0,1,11))
    ax.grid(b=True, alpha=0.5)
    ax.set_xlabel("$\delta$")
    ax.set_ylabel("Equilibrium effort")
    ax.set_title("Effort vs. $\delta$ on a 4-by-4 lattice")
    plt.show()
    
if __name__ == '__main__':
    main()