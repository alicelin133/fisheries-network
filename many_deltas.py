"""Creates many Simulation_arrays objects which vary only in delta, and runs the
simulations over the same number of time steps. Calculates the average long-
term effort for each value of delta, and saves the results to a binary numpy
file. Similarly calculates and saves average resource level and payoff.

Depends on:
Simulation_arrays.py
numpy
matplotlib.pyplot
time
os
"""

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
import time
import os
import math

import Simulation_arrays

def calculate_e_msr(n_fishers, q, r, K, price, cost):
    """Calculates value of e_msr (maximum sustainable rent)."""
    return r * (price * q * K * n_fishers - n_fishers * cost) / (2 * price * q * q * K * n_fishers)

def calculate_e_nash(e_msr, n_fishers):
    """Calculates value of Nash equilibrium level of effort."""
    return e_msr * 2 * n_fishers / (1 + n_fishers)

def create_sims(n_fishers, deltas, q, r, K, R_0, e_0, price, cost, noise, num_feedback, payoff_discount, num_steps):
    """Creates list of Simulation objects varying in delta."""
    sims_list = []
    i = 1
    for delta in deltas:
        sim = Simulation_arrays.Simulation_arrays(n_fishers, delta, q, r, K, R_0, e_0, price,
                                    cost, noise, num_feedback, payoff_discount, num_steps)
        sim.simulate()
        print("{} of {} simulations done!".format(i, deltas.size))
        i += 1
        sims_list.append(sim)
    return sims_list

def calculate_e_avg(sims_list):
    """Returns array of average effort levels corresponding to different
    values of delta. Average is calculated by averaging all values after the
    first 50 time steps."""
    e_avg = np.zeros(len(sims_list))
    for i in range(len(sims_list)):
        e_avg[i] = np.average(sims_list[i].e_data[:,50:])
    return e_avg

def calculate_R_avg(sims_list):
    """Returns array of average resource levels corresponding to different
    values of delta. Average is calculated by averaging all values after the
    first 50 time steps."""
    R_avg = np.zeros(len(sims_list))
    for i in range(len(sims_list)):
        R_avg[i] = np.average(sims_list[i].R_data[:,50:])
    return R_avg

def calculate_U_avg(sims_list):
    """Returns array of average fisher utility corresponding to different
    values of delta. Average is calculated by averaging all values after the
    first 50 time steps."""
    U_avg = np.zeros(len(sims_list))
    for i in range(len(sims_list)):
        U_avg[i] = np.average(sims_list[i].U_data[:,50:])
    return U_avg

def write_fname(sim, num_steps, deltas, path, data):
    """Pulls data attributes from the *sim* Simulation_arrays object to create a
    file name."""
    string_list = ["npy-deltas",
    data,
    "num_deltas" + str(deltas.size),
    "delta_lo" + str(deltas[0]),
    "delta_hi" + str(deltas[-1]),
    "n" + str(sim.n_fishers),
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
    "num_feedback" + str(sim.num_feedback),
    "p_discount" + str(sim.payoff_discount),
    "num_steps" + str(num_steps),
    "v"]
    fname = "-".join(split_join(string_list))
    for root, dirs, files in os.walk(path):
        filenames = files
    matching_fname = [s for s in filenames if fname in s]
    fname = path + fname + str(len(matching_fname)) + ".npy"
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
    # Assign parameter values: n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise, num_feedback,
    # num_steps, deltas
    n_fishers = 20
    q = 1
    r = 0.04
    K = 5000 / n_fishers
    price = 1
    cost = 0.5
    noise = 0.00001
    num_feedback = 50
    payoff_discount = 0.5
    num_steps = 1000 # for the avg, needs to be > 50
    R_0 = np.full(n_fishers,K)
    e_msr = calculate_e_msr(n_fishers, q, r, K, price, cost)
    e_nash = calculate_e_nash(e_msr, n_fishers)
    e_max = r/q
    e_0 = np.linspace(0, e_max, n_fishers)
    num_deltas = 100
    deltas = np.linspace(0.005, 0.995, num_deltas, endpoint=True)
    
    # Create Simulations and plot results
    sims_list = create_sims(n_fishers, deltas, q, r, K, R_0, e_0, price, cost,
                            noise, num_feedback, payoff_discount, num_steps)
    e_avg = calculate_e_avg(sims_list)
    R_avg = calculate_R_avg(sims_list)
    U_avg = calculate_U_avg(sims_list)

    # Save data from Simulations to .npy files
    path = "/Users/alicelin/Documents/fish/fisheries-network/data/"
    e_fname = write_fname(sims_list[0], num_steps, deltas, path, "e")
    R_fname = write_fname(sims_list[0], num_steps, deltas, path, "R")
    U_fname = write_fname(sims_list[0], num_steps, deltas, path, "U")
    dels_fname = write_fname(sims_list[0], num_steps, deltas, path, "deltas")
    np.save(e_fname, e_avg)
    np.save(R_fname, R_avg)
    np.save(U_fname, U_avg)
    np.save(dels_fname, deltas)
    print("Effort: {} \n Resource: {} \n Payoff: {} \n Deltas: {}".format(
        e_fname, R_fname, U_fname, dels_fname))
    seconds = time.time() - start_time
    minutes = math.floor(seconds / 60)
    seconds_mod = seconds - minutes * 60
    print("--- {} min {} sec ---".format(minutes, seconds_mod))

if __name__ == "__main__":
    main()