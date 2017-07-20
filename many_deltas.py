"""Creates many Simulation objects which vary only in delta, and runs the
simulations over the same number of time steps. Plots the average
effort for each Simulation object as a function of delta. The average takes
into account all but the first 50 time steps.

Depends on:
Simulation.py
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

def create_sims(n_fishers, deltas, q, r, K, R_0, e_0, price, cost, noise, num_feedback):
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
    values of delta."""
    e_avg = np.zeros(len(sims_list))
    for i in range(len(sims_list)):
        e_avg[i] = np.average(sims_list[i].e_data[:,50:])
    return e_avg

def plot_e_vs_delta(deltas, e_avg):
    """Plots the values of e_avg vs. values of delta."""
    plt.plot(deltas,e_avg)
    plt.xlabel("Delta (degree of fish movement")
    plt.ylabel("Average effort level of territories")
    plt.title("Optimal Effort Level vs. Delta")

if __name__ == "__main__":
    start_time = time.time()
    # Assign parameter values: n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise, num_feedback
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
    e_0 = np.linspace(0, 1, n_fishers)

    # Assign delta values
    num_deltas = 5
    deltas = np.linspace(0, 1, num_deltas, endpoint=True)
    
    # Create Simulations and plot results
    sims_list = create_sims(n_fishers, deltas, q, r, K, R_0, e_0, price, cost, noise, num_feedback)
    e_avg = calculate_e_avg(sims_list)
    plot_e_vs_delta(deltas, e_avg)

    # Save figure generated to a file name
    path = "/Users/alicelin/Documents/fish/fisheries-network/data/"
    fig_fname = "fig-deltas" + num_deltas + "-" +
                sim_fig.write_fname(sims_list[-1], num_steps, path)
    fig_fname_with_path = path + fig_fname
    print(fig_fname_with_path) # easily added to data log spreadsheet
    plt.savefig(fig_fname_with_path)

    # Save data from Simulations to txt files
    # txt_fname = "txt-deltas-" + sim_fig.write_fname(sims_list[-1], num_steps, path)

    print("--- %s minutes ---" % ((time.time() - start_time)/60.0))
    plt.show()