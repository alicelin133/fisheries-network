"""Creates many Simulation objects which vary only in delta, and runs the
simulations over the same number of time steps. Plots the average
effort for each Simulation object as a function of delta.

Depends on:
Simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
import time

import Simulation

start_time = time.time()

# assign parameter values:
# (n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise, num_feedback)
n_fishers = 20
q = 1
r = 0.04
K = 5000 / n_fishers
price = 1
cost = 0.5
noise = 0.00005
num_feedback = 10
num_steps = 1000
R_0 = np.full(n_fishers,K)
e_msy = r * (price * q * K * n_fishers - n_fishers * cost) / (2 * price * q * q * K * n_fishers)
print("e_msy = {}".format(e_msy))
e_nash = e_msy * 2 * n_fishers / (1 + n_fishers)
print("e_nash = {}".format(e_nash))
e_0 = np.linspace(0, e_nash, n_fishers)

num_deltas = 200
deltas = np.linspace(0, 1, num_deltas, endpoint=True)
e_end = np.zeros(num_deltas)
sims_list = []
for delta in deltas:
    sim = Simulation.Simulation(n_fishers, delta, q, r, K, R_0, e_0, price,
                                cost, noise, num_feedback)
    sim.simulate(num_steps)
    sims_list.append(sim)
for i in range(len(sims_list)):
    e_end[i] = np.average(sims_list[i].e_data[:,-1])
plt.plot(deltas,e_end)
plt.xlabel("Delta (degree of fish movement")
plt.ylabel("Average effort level of territories")
plt.title("Optimal Effort Level vs. Delta")
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()