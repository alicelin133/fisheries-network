"""Creates many Simulation objects which vary only in delta, and runs the
simulations over the same number of time steps. Calculates the average long-
term effort for each value of delta, and saves the results to a binary numpy
file. Similarly calculates and saves average resource level and payoff.

Depends on:
Simulation.py
numpy
matplotlib.pyplot
time
"""

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
import time

import Simulation
import sim_fig

def calculate_e_msr(n_fishers, q, r, K, price, cost):
    """Calculates value of e_msr (maximum sustainable rent)."""
    return r * (price * q * K * n_fishers - n_fishers * cost) / (2 * price * q * q * K * n_fishers)

def calculate_e_nash(e_msr, n_fishers):
    """Calculates value of Nash equilibrium level of effort."""
    return e_msr * 2 * n_fishers / (1 + n_fishers)

def create_sims(n_fishers, deltas, q, r, K, R_0, e_0, price, cost, noise, num_feedback, num_steps):
    """Creates list of Simulation objects varying in delta."""
    sims_list = []
    for delta in deltas:
        sim = Simulation.Simulation(n_fishers, delta, q, r, K, R_0, e_0, price,
                                    cost, noise, num_feedback)
        sim.simulate(num_steps)
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

def calculate_pi_avg(sims_list):
    """Returns array of average fisher payoffs corresponding to different
    values of delta. Average is calculated by averaging all values after the
    first 50 time steps."""
    pi_avg = np.zeros(len(sims_list))
    for i in range(len(sims_list)):
        pi_avg[i] = np.average(sims_list[i].pi_data[:,50:])
    return pi_avg

def input_parameters():
    """Allows user to input parameter values for Simulations."""
    n_fishers = int(input("Number of fishers (int): "))
    q = float(input("q (in [0,1]): "))
    r = float(input("r (in [0,1]): "))
    K = float(input("K (total): "))
    price = float(input("price: "))
    cost = float(input("cost: "))
    noise = float(input("noise: "))
    num_feedback = int(input("num_feedback (number of feedback loops per strategy update): "))
    num_steps = int(input("num_steps (number of time steps, > 50): "))
    R_0 = eval(input("R_0 (numpy array, initial resource levels): "))
    e_0 = eval(input("e_0 (numpy array, initial effort levels): "))
    e_msr = calculate_e_msr(n_fishers, q, r, K, price, cost)
    e_nash = calculate_e_nash(e_msr, n_fishers)
    num_deltas = int(input("num_deltas (number of delta values to test): "))
    delta_min = float(input("Minimum delta value: "))
    delta_max = float(input("Maximum delta value: "))
    deltas = np.linspace(delta_min, delta_max, num_deltas, endpoint=True)

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
    num_steps = 60 # for the avg, needs to be > 50
    R_0 = np.full(n_fishers,K)
    e_msr = calculate_e_msr(n_fishers, q, r, K, price, cost)
    e_nash = calculate_e_nash(e_msr, n_fishers)
    e_max = r/q
    e_0 = np.linspace(0, e_max, n_fishers)
    # Assign delta values
    num_deltas = 10
    deltas = np.linspace(0, 1, num_deltas, endpoint=True)

    yes_no = input("Would you like to input your own parameters? (Y/N) ")
    while (yes_no != "Y" and yes_no != "N"):
        yes_no = input("Please type Y or N: ")
    if yes_no == "Y":
        input_parameters()
    
    # Create Simulations and plot results
    sims_list = create_sims(n_fishers, deltas, q, r, K, R_0, e_0, price, cost,
                            noise, num_feedback, num_steps)
    e_avg = calculate_e_avg(sims_list)
    R_avg = calculate_R_avg(sims_list)
    pi_avg = calculate_pi_avg(sims_list)

    # Save data from Simulations to .npy files
    path = "/Users/alicelin/Documents/fish/fisheries-network/data/"
    # txt_fname_list = []
    # for i in range(len(sims_list)):
    # txt_fname = "txt-deltas-" + num_deltas + "-" +
    #             sim_fig.write_fname(sims_list[-1], num_steps, path)

    print("--- %s minutes ---" % ((time.time() - start_time)/60.0))

if __name__ == "__main__":
    main()