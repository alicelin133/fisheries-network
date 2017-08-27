"""Runs simulations for varying levels of delta to determine average effort.
100 simulations per value of delta, with fixed seed for reproducible results.
"""
import numpy as np
import matplotlib.pyplot as plt
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

def get_e_avg(sim):
    """Returns average value of *sim*'s e_data from one simulation. Averages
    across all nodes and includes all time steps except the first 100."""
    e_avg = np.average(sim.e_data[:,100:])
    return e_avg

def create_sims(num_sims, n_fishers, deltas, q, r, K, R_0, e_0, price, cost,
                noise, num_feedback, payoff_discount, num_steps):
    """Creates *num_sims* Simulation_array objects and runs the simulations.
    Returns the average *e_data* array over time. Sets the seed for
    reproducibility."""
    e_avgs = np.zeros(deltas.size)
    for j in range(deltas.size):
        e_vals = np.zeros(num_sims)
        for i in range(num_sims):
            np.random.seed(i)
            print("Delta: {} Number: {}".format(deltas[j], i))
            sim = Simulation_arrays.Simulation_arrays(n_fishers, deltas[j], q, r, K, R_0,
                e_0, price, cost, noise, num_feedback, payoff_discount, num_steps)
            sim.simulate()
            e_vals[i] = get_e_avg(sim)
        e_avgs[j] = np.mean(e_vals)
    return e_avgs

if __name__ == '__main__':
    start_time = time.time()

    # Assign delta values
    deltas = np.linspace(0, 1, 3, endpoint=True)

    # Assign other parameter values for simulations
    n_fishers = 20
    q = 1
    r = 0.05
    K = 200
    R_0 = np.full(n_fishers, K/2)
    e_0 = np.linspace(0, r/q, num=n_fishers)
    price = 1
    cost = 0.5
    noise = 0.00001
    num_feedback = 500
    payoff_discount = 0.5
    num_steps = 1000

    # Create simulations for each delta value
    num_sims = 1
    e = create_sims(num_sims, n_fishers, deltas, q, r, K, R_0, e_0, price,
                    cost, noise, num_feedback, payoff_discount, num_steps)
    
    # Plot results
    plt.plot(deltas, e)

    # Add lines to indicate e_msr and e_nash levels
    e_msr = calculate_e_msr(n_fishers, q, r, K, price, cost)
    e_nash = calculate_e_nash(e_msr, n_fishers)
    plt.axhline(y=e_msr, linewidth=1)
    plt.axhline(y=e_nash, linewidth=1)
    plt.text(0.1, e_msr, '$e_{msr}$', va='bottom')
    plt.text(0.1, e_nash, '$e_{Nash}$', va='bottom')
    plt.xlabel("Delta")
    plt.ylabel("Effort Level")
    plt.title("Effort vs. Delta")

    seconds = time.time() - start_time
    minutes = math.floor(seconds / 60)
    seconds_mod = seconds - minutes * 60
    print("--- {} min {} sec ---".format(minutes, seconds_mod))

    plt.show()