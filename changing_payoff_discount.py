"""Runs simulations, varying payoff_discount parameter, and plots
one effort vs. delta plot per value of payoff_discount in a 3D plot."""

import time
import numpy as np
import pickle

import Simulation_arrays

if __name__ == '__main__':
    start_time = time.time()

    # Assign parameter values for simulations
    n_fishers = 20
    q = 1
    r = 0.05
    K = 200
    R_0 = np.full(n_fishers, K/2)
    e_0 = np.linspace(0, r/q, num=n_fishers)
    price = 1
    cost = 0.5
    noise = 0.0001
    num_feedback = 50
    num_steps = 1000

    # Assign values for *payoff_discount*
    payoff_discounts = np.linspace(0, 1, 6, endpoint=True)

    # Assign values for *delta*
    deltas = np.linspace(0, 0.1, 10, endpoint=True)

    # Assign seed value (idk)
    seed = 15

    # Create simulations
    e_vals = np.zeros((payoff_discounts.size, deltas.size))
    for i in range(payoff_discounts.size):
        for j in range(deltas.size):
            np.random.seed(seed)
            sim = Simulation_arrays.Simulation_arrays(n_fishers, deltas[j],
                  q, r, K, R_0, e_0, price, cost, noise, num_feedback,
                  payoff_discounts[i], num_steps)
            sim.simulate()
            e_vals[i][j] = np.mean(sim.e_data[:,100:])
            print("{}th simulation done of {}!".format(i * deltas.size + j,
                payoff_discounts.size * deltas.size)) # keep track of progress

    # Save data
    path = "/Users/alicelin/Documents/fish/fisheries-network/data/"
    fname_deltas = path + "deltas-seed" + str(seed) + "-changing_payoff_discount.npy"
    fname_e_vals = path + "e_vals-seed" + str(seed) + "-changing_payoff_discount.npy"
    fname_payoff_discounts = path + "payoff_discounts-seed" + str(seed) + \
                            "-changing_payoff_discount.npy"
    fname_sim = path + "sim-seed" + str(seed) + "-changing_payoff_discount.file"
    np.save(fname_deltas, deltas)
    np.save(fname_e_vals, e_vals)
    np.save(fname_payoff_discounts, payoff_discounts)

    # Save simulation data attributes to get e_msr in separate file
    with open(fname_sim, 'wb') as f:
        pickle.dump(sim, f, pickle.HIGHEST_PROTOCOL)

    print("--- %s seconds ---" % (time.time() - start_time))