"""Loads and plots .npy file data from many_deltas.py.

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

def main():
    e_path = "/Users/alicelin/Documents/fish/fisheries-network/data/npy-deltas-e-num_deltas20-delta_lo0_0-delta_hi0_19-n20-delta0_0-q1-r0_04-K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_0hi0_04-num_feedback50-p_discount0_5-num_steps1000-v0.npy"
    deltas_path = "/Users/alicelin/Documents/fish/fisheries-network/data/npy-deltas-deltas-num_deltas20-delta_lo0_0-delta_hi0_19-n20-delta0_0-q1-r0_04-K250_0-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_0hi0_04-num_feedback50-p_discount0_5-num_steps1000-v0.npy"
    e_avg = np.load(e_path)
    deltas = np.load(deltas_path)
    plot_e_vs_delta(deltas, e_avg)

    # Save figure generated to a file name
    path = "/Users/alicelin/Documents/fish/fisheries-network/data/"
    fig_info = "e_vs_delta"
    sim_info = "n20-delta0_0-q1-r0_04-K250_0" + \
    "-price1-cost0_5-noise1e-05-R_0lo250_0-R_0hi250_0-e_0lo0_0-e_0hi0_04-nu" + \
    "m_regrowth50-num_steps60" # copied from e_path
    fig_fname = write_fname(fig_info, deltas, sim_info, path)
    print(fig_fname) # easily added to data log spreadsheet
    # plt.savefig(fig_fname)
    plt.show()

if __name__ == "__main__":
    main()