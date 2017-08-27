"""Loads .npy file data from many_deltas.p, plots a clean curve of effort
vs delta.

Depends on:
numpy
matplotlib.pyplot
os
"""

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
import os

def plot_e_vs_delta(deltas, e_avg):
    """Plots the values of e_avg vs. values of delta."""
    plt.plot(deltas,e_avg)
    plt.xlabel("Delta (degree of fish movement")
    plt.ylabel("Average effort level of territories")
    plt.title("Optimal Effort Level vs. Delta")

def write_fname(fig_info, deltas, sim_info, path):
    """Pulls data attributes from the *sim* Simulation object to create a
    file name."""
    string_list = ["fig-deltas",
    fig_info,
    "num_deltas" + str(deltas.size),
    "delta_lo" + str(deltas[0]),
    "delta_hi" + str(deltas[-1]),
    sim_info,
    "v"]
    fname = "-".join(split_join(string_list))
    for root, dirs, files in os.walk(path):
        filenames = files
    matching_fname = [s for s in filenames if fname in s]
    fname = path + fname + str(len(matching_fname)) + ".png"
    return fname

def split_join(list_of_numbers):
    """Helper function for write_fname() method. Replaces any periods
    in the list elements with underscores and returns a new list."""
    new_list = []
    for number in list_of_numbers:
        s = str(number).split(".")
        new_s = "_".join(s)
        new_list.append(new_s)
    return new_list

def load_arrays(path, num_v):
    """Loads data from versions of the same simulation from a given path
    ending in "v" and returns an array averaging each point of all versions."""
    arrays = []
    for i in range(num_v):
        fname = path + str(i) + ".npy"
        arrays.append(np.load(fname))
    e_avg = np.average(np.stack(arrays, axis=0), axis=0)
    return e_avg

def main():
    num_v = 5

    # Loading [0,0.2) segment
    # remember that path ends in "v"
    epath1 = "/Users/alicelin/Documents/fish/fisheries-network/data/npy-d" + \
    "eltas-e-num_deltas20-delta_lo0_0-delta_hi0_19-n20-delta0_0-q1-r0_04-" + \
    "K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_0h" + \
    "i0_04-num_feedback50-p_discount0_5-num_steps1000-v"
    e_avg1 = load_arrays(epath1, num_v)

    # Loading [0.2,0.4) segment
    epath2 = "/Users/alicelin/Documents/fish/fisheries-network/data/npy-d" + \
    "eltas-e-num_deltas20-delta_lo0_2-delta_hi0_39-n20-delta0_2-q1-r0_04-" + \
    "K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_0h" + \
    "i0_04-num_feedback50-p_discount0_5-num_steps1000-v"
    e_avg2 = load_arrays(epath2, num_v)

    # Loading [0.4,0.6) segment
    epath3 = "/Users/alicelin/Documents/fish/fisheries-network/data/npy-d" + \
    "eltas-e-num_deltas20-delta_lo0_4-delta_hi0_59-n20-delta0_4-q1-r0_04-" + \
    "K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_0h" + \
    "i0_04-num_feedback50-p_discount0_5-num_steps1000-v"
    e_avg3 = load_arrays(epath3, num_v)
    
    # Loading [0.6,0.8) segment
    epath4 = "/Users/alicelin/Documents/fish/fisheries-network/data/npy-d" + \
    "eltas-e-num_deltas20-delta_lo0_6-delta_hi0_79-n20-delta0_6-q1-r0_04-" + \
    "K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_0h" + \
    "i0_04-num_feedback50-p_discount0_5-num_steps1000-v"
    e_avg4 = load_arrays(epath4, num_v)

    # Loading [0.8,1] segment
    # remember that path ends in "v"
    epath5 = "/Users/alicelin/Documents/fish/fisheries-network/data/npy-d" + \
    "eltas-e-num_deltas21-delta_lo0_8-delta_hi1_0-n20-delta0_8-q1-r0_04-K" + \
    "250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_0hi" + \
    "0_04-num_feedback50-p_discount0_5-num_steps1000-v"
    e_avg5 = load_arrays(epath5, num_v)

    # Entire set of e_avg values
    e_avg = np.concatenate((e_avg1, e_avg2, e_avg3, e_avg4, e_avg5))

    # Putting together delta values
    dpath1 = "/Users/alicelin/Documents/fish/fisheries-network/data/npy-de" + \
    "ltas-deltas-num_deltas20-delta_lo0_0-delta_hi0_19-n20-delta0_0-q1-r0_" + \
    "04-K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_" + \
    "0hi0_04-num_feedback50-p_discount0_5-num_steps1000-v0.npy"
    dpath2 = "/Users/alicelin/Documents/fish/fisheries-network/data/npy-de" + \
    "ltas-deltas-num_deltas20-delta_lo0_2-delta_hi0_39-n20-delta0_2-q1-r0_" + \
    "04-K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_" + \
    "0hi0_04-num_feedback50-p_discount0_5-num_steps1000-v0.npy"
    dpath3 = "/Users/alicelin/Documents/fish/fisheries-network/data/npy-de" + \
    "ltas-deltas-num_deltas20-delta_lo0_4-delta_hi0_59-n20-delta0_4-q1-r0_" + \
    "04-K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_" + \
    "0hi0_04-num_feedback50-p_discount0_5-num_steps1000-v0.npy"
    dpath4 = "/Users/alicelin/Documents/fish/fisheries-network/data/npy-de" + \
    "ltas-deltas-num_deltas20-delta_lo0_6-delta_hi0_79-n20-delta0_6-q1-r0_" + \
    "04-K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_" + \
    "0hi0_04-num_feedback50-p_discount0_5-num_steps1000-v0.npy"
    dpath5 = "/Users/alicelin/Documents/fish/fisheries-network/data/npy-de" + \
    "ltas-deltas-num_deltas21-delta_lo0_8-delta_hi1_0-n20-delta0_8-q1-r0_0" + \
    "4-K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_0" + \
    "hi0_04-num_feedback50-p_discount0_5-num_steps1000-v0.npy"
    d1 = np.load(dpath1)
    d2 = np.load(dpath2)
    d3 = np.load(dpath3)
    d4 = np.load(dpath4)
    d5 = np.load(dpath5)

    deltas = np.concatenate((d1, d2, d3, d4, d5))

    # Loading in-between effort arrays
    extra_e_path = "/Users/alicelin/Documents/fish/fisheries-network/da" + \
    "ta/npy-deltas-e-num_deltas100-delta_lo0_005-delta_hi0_995-n20-delta0" + \
    "_005-q1-r0_04-K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0" + \
    "-e_0lo0_0-e_0hi0_04-num_feedback50-p_discount0_5-num_steps1000-v"
    extra_e_avg = load_arrays(extra_e_path, num_v)

    # Loading in-between delta array
    extra_deltas_path = "/Users/alicelin/Documents/fish/fisheries-network" + \
    "/data/npy-deltas-deltas-num_deltas100-delta_lo0_005-delta_hi0_995-n2" + \
    "0-delta0_005-q1-r0_04-K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_" + \
    "0hi250_0-e_0lo0_0-e_0hi0_04-num_feedback50-p_discount0_5-num_steps10" + \
    "00-v0.npy"
    extra_deltas = np.load(extra_deltas_path)

    e_avg_final = np.concatenate((e_avg, extra_e_avg))
    deltas_final = np.concatenate((deltas, extra_deltas))
    plt.scatter(deltas_final, e_avg_final, s=1)

    # Plotting figure
    #plot_e_vs_delta(deltas, e_avg)

    # Save figure generated to a file name
    path = "/Users/alicelin/Documents/fish/fisheries-network/data/"
    fig_info = "e_vs_delta-scatter"
    sim_info = "n20-delta0" + \
    "_005-q1-r0_04-K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0" + \
    "-e_0lo0_0-e_0hi0_04-num_feedback50-p_discount0_5-num_steps1000" # copied from e_path
    fig_fname = write_fname(fig_info, deltas, sim_info, path)
    print(fig_fname) # easily added to data log spreadsheet
    plt.savefig(fig_fname)
    plt.show()

if __name__ == "__main__":
    main()