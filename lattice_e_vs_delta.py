"""For each of several values for delta, creates and runs many
Simulation_lattice object instances. Averages the e_data, R_data, and
U_data attributes of the Simulation_lattice object for all the nodes
of all instances of one value of delta, and stacks these averages with
the average arrays of all other values of delta."""

import numpy as np
import os
import time
import math

import Simulation_lattice

def get_e_msr_nash(n_fishers, r, q, K, price, cost):
    """Takes parameters and returns the two values
    of e_msr (maximum sustainable rent) and e_nash (nash equilibrium)
    levels of effort. Returns the pair of values e_msr, e_nash
    in that order.
    WARNING: if any of the values r, n_fishers, q, K, price, or cost change,
    this function is NOT GOOD because now you have different values of e_msr
    and e_nash depending on the simulation."""
    e_msr = r * (price * q * K * n_fishers - n_fishers * cost) / \
            (2 * price * q * q * K * n_fishers)
    e_nash = e_msr * 2 * n_fishers / (1 + n_fishers)
    return e_msr, e_nash

def create_sims(num_sims, network_dims, delta, q, r, K, R_0, e_0, e_nash,
                price, cost, noise, num_feedback, num_steps):
    """Creates *num_sims* Simulation_lattice objects with the given
    parameters, runs the simulation, then averages the results of each
    node of each object over time."""
    n_fishers = network_dims[0] * network_dims[1]
    e = np.zeros((n_fishers, num_steps))
    R = np.zeros((n_fishers, num_steps))
    U = np.zeros((n_fishers, num_steps))
    for i in range(num_sims):
        # Set seed for replicable pseudo-RNG
        np.random.seed(i)

        # Create current simulation
        sim = Simulation_lattice.Simulation_lattice(network_dims, delta, q, r, K,
              R_0, e_0, price, cost, noise, num_feedback, num_steps)
        # Assign "left" half to be have e_nash, "right" is e_msr
        for node in sim.G.nodes(data=False):
            if node[1] < network_dims[1] / 2:
                sim.G.node[node]['e'] = e_nash
        sim.simulate()
        print("Sim {} of {} done for delta = {}!".format(i, num_sims, delta))
        e = np.add(e, sim.e_data)
        R = np.add(R, sim.R_data)
        U = np.add(U, sim.U_data)
    e = e / num_sims # Calculate mean
    R = R / num_sims
    U = U / num_sims
    return e, R, U

def write_fname(path, network_dims, deltas, q, r, K, price, cost,
    noise, R_0, num_feedback, num_steps):
    """Creates an identifier for this simulation by taking info about
    the simulations and turning it into a string."""
    string_list = ["lattice",
    "{}x{}".format(network_dims[0], network_dims[1]),
    "num_deltas" + str(deltas.size),
    "deltas{}to{}".format(deltas[0], deltas[-1]),
    "q" + str(q),
    "r" + str(r),
    "K" + str(K),
    "price" + str(price),
    "cost" + str(cost),
    "noise" + str(noise),
    "R_0" + str(R_0[0]),
    "num_feedback" + str(num_feedback),
    "num_steps" + str(num_steps),
    "v"]
    fname = "-".join(split_join(string_list))
    for root, dirs, files in os.walk(path):
        filenames = files
    matching_fname = [s for s in filenames if fname in s]
    fname = fname + str(len(matching_fname)) + ".npy"
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
    start_time = time.time()

    # Assign parameter values
    # (network_dims, delta, q, r, K, R_0, e_0, price, cost, noise, num_feedback, num_steps)
    network_dims = [6,6]
    n_fishers = network_dims[0] * network_dims[1]
    q = 1
    r = 0.06
    K = 200
    R_0 = np.full(n_fishers, K/2)
    price = 1
    cost = 0.5
    e_msr, e_nash = get_e_msr_nash(n_fishers, r, q, K, price, cost)
    e_0 = np.full(n_fishers, e_msr) # half will be changed to e_nash
    noise = 0.0001
    num_feedback = 25
    num_steps = 1000

    # Assign delta values
    deltas = np.linspace(0,0.1,21)

    # Create lattice simulations for each delta value
    num_sims = 50
    e_list = []
    R_list = []
    U_list = []
    for delta in deltas:
        e, R, U = create_sims(num_sims, network_dims, delta, q, r, K, R_0, e_0, e_nash,
                    price, cost, noise, num_feedback, num_steps)
        e_list.append(e)
        R_list.append(R)
        U_list.append(U)
    e_to_save = np.stack(e_list)
    R_to_save = np.stack(R_list)
    U_to_save = np.stack(U_list)

    # Save e, R, U arrays to disk
    path = "/Users/alicelin/Documents/fish/fisheries-network/data/"
    data_info = "half-e_msr-half-e_nash"
    fname = write_fname(path, network_dims, deltas, q, r,
        K, price, cost, noise, R_0, num_feedback, num_steps)
    e_fname = path + "e-" + fname
    R_fname = path + "R-" + fname
    U_fname = path + "U-" + fname
    deltas_fname = path + "deltas-" + fname
    print("{} \n {} \n {} \n {}".format(e_fname, R_fname, U_fname, deltas_fname))
    np.save(e_fname, e_to_save)
    np.save(R_fname, R_to_save)
    np.save(U_fname, U_to_save)
    np.save(deltas_fname, deltas)

    # Time stuff
    seconds = time.time() - start_time
    minutes = math.floor(seconds / 60)
    seconds_mod = seconds - minutes * 60
    print("--- {} min {} sec ---".format(minutes, seconds_mod))

if __name__ == '__main__':
    main()