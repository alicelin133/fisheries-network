"""Given a Simulation object, number of steps, and directory path,
sim_fig.save_fig() creates a file name for the figure using object data
attributes and saves the figure under this file name as a PNG in the
argument directory."""

import matplotlib.pyplot as plt
import os
import Simulation

def write_fname(sim, num_steps, path):
    """Pulls data attributes from the Simulation object to create a file name."""
    string_list = ["n" + str(sim.n_fishers),
    "delta" + str(sim.delta),
    "q" + str(sim.q),
    "r" + str(sim.r),
    "K" + str(sim.K),
    "price" + str(sim.price),
    "cost" + str(sim.cost),
    "noise" + str(sim.noise),
    "R_0lo" + str(sim.R_0[0]),
    "R_0hi" + str(sim.R_0[-1]),
    "e_0lo" + str(sim.e_0[0]),
    "e_0hi" + str(sim.e_0[-1]),
    "num_regrowth" + str(sim.num_feedback),
    "num_steps" + str(num_steps),
    "v"]
    fname = "-".join(split_join(string_list))
    for root, dirs, files in os.walk(path):
        filenames = files
    matching_fname = [s for s in filenames if fname in s]
    fname += str(len(matching_fname))
    return fname

def add_path(path, fname):
    """Takes a path *path* and a file name *fname* and concatenates them
    to get the full path for a new file."""
    path_fname = path + fname
    return path_fname
    
def split_join(list_of_numbers):
    """Helper function for write_fname() method. Replaces any periods
    in the list elements with underscores and returns a new list."""
    new_list = []
    for number in list_of_numbers:
        s = str(number).split(".")
        new_s = "_".join(s)
        new_list.append(new_s)
    return new_list

def save_fig(sim, num_steps, path):
    """Saves figure to a generated file name in a path directory."""
    fname = add_path(path, write_fname(sim, num_steps, path) + ".png")
    print(fname)
    plt.savefig(fname)

    