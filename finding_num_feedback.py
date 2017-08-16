"""Finds the optimal value of num_feedback as a parameter
by graphing the value of R vs number of resource updates."""

# USING: LATTICE

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import time
plt.switch_backend('Qt5Agg')
import math

import Simulation_lattice
import Simulation_arrays

def get_Rs(sim, max_feedback):
    Rs = np.zeros(max_feedback)
    for i in range(max_feedback):
        sim.harvest(i)
        sim.regrowth()
        sim.leakage()
        for nood in sim.G.nodes(data=False):
            Rs[i] += sim.G.node[nood]['R']
    return Rs


def main():
    # Assign parameter values besides e_0
    # network_dims, delta, q, r, K, R_0, e_0, price, cost, noise, num_feedback, num_steps
    network_dims = [6,6]
    n_fishers = network_dims[0] * network_dims[1]
    delta = 0.01
    q = 1
    r = 0.05
    K = 200
    R_0 = np.full(n_fishers, K/2)
    price = 1
    cost = 0.5
    noise = 0.0001
    num_feedback = 1 # not actually important
    num_steps = 1 # not actually important

    max_feedback = 150

    # Random initial efforts
    for i in range(20):
        e_rand = np.random.random(n_fishers) * r/q
        sim_rand = Simulation_lattice.Simulation_lattice(network_dims, delta, q,
            r, K, R_0, e_rand, price, cost, noise, num_feedback, num_steps)
        Rs_rand = get_Rs(sim_rand, max_feedback)
        plt.plot(Rs_rand, color='r', linewidth=0.5)
    plt.plot(Rs_rand, color='r', linewidth=0.5, label='random')
    

    # Half e_msr, half e_nash
    e_msr = Simulation_lattice.calculate_e_msr(n_fishers, q, r, K, price, cost)
    e_nash = Simulation_lattice.calculate_e_nash(e_msr, n_fishers)
    e_half = np.concatenate((np.full(math.floor(n_fishers/2), e_msr),
        np.full(math.ceil(n_fishers/2), e_nash)))
    sim_half = Simulation_lattice.Simulation_lattice(network_dims, delta, q,
        r, K, R_0, e_half, price, cost, noise, num_feedback, num_steps)
    Rs_half = get_Rs(sim_half, max_feedback)
    plt.plot(Rs_half, label='half e_msr half e_nash', linewidth=0.5)

    # Plot labels
    plt.style.use('bmh')
    plt.xlabel("Number of resource updates (num_feedback)")
    plt.ylabel("Sum of nodes' resource levels")
    plt.title("Resource level vs. number of ecological feedback loops")
    plt.legend()
    plt.grid(b=True)
    plt.show()
    

if __name__ == '__main__':
    main()