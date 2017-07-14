"""Uses Simulation class to run a simulation given the parameters below."""

import numpy as np
import matplotlib.pyplot as plt
import time
import Simulation

start_time = time.time()

# Assigning parameter values
n_fishers = 50
delta = 0
q = 1
r = 0.01
K = 5000 / n_fishers
price = 1
cost = 0.5
noise = 0.00005
num_regrowth = 10
num_steps = 1000
R_0 = np.full(n_fishers,K)
# creating initial distribution across 2 effort levels
f_msy = 0.5
e_msy = r * (price * q * K * n_fishers - n_fishers * cost) / (2 * price * q * q * K * n_fishers)
print("e_msy = {}".format(e_msy))
e_nash = e_msy * 2 * n_fishers / (1 + n_fishers)
print("e_nash = {}".format(e_nash))
num_msy = round(f_msy * n_fishers)
num_nash = n_fishers - num_msy
# e_0 = np.concatenate((np.full(num_msy, e_msy), np.full(num_nash, e_nash)), axis=0)
e_0 = np.linspace(0, e_nash, n_fishers)

# Creating Simulation object
sim0 = Simulation.Simulation(n_fishers, 0, q, r, K, R_0, e_0, price, cost, noise, num_regrowth)
sim0.simulate(num_steps)

sim05 = Simulation.Simulation(n_fishers, 0.5, q, r, K, R_0, e_0, price, cost, noise, num_regrowth)
sim05.simulate(num_steps)

sim1 = Simulation.Simulation(n_fishers, 1, q, r, K, R_0, e_0, price, cost, noise, num_regrowth)
sim1.simulate(num_steps)

# Plotting avg effort vs. time
e_avg0 = np.average(sim0.e_data, axis=0)
e_avg05 = np.average(sim05.e_data, axis=0)
e_avg1 = np.average(sim1.e_data, axis=0)
plt.plot(np.arange(num_steps), e_avg0, label="delta = 0")
plt.plot(np.arange(num_steps), e_avg05, label="delta=0.5")
plt.plot(np.arange(num_steps), e_avg1, label="delta=1")
plt.xlabel("Time steps")
plt.ylabel("Effort")
plt.yticks(np.linspace(0.004,0.01, 20))
plt.title("Average Effort vs. Time")
plt.legend()
plt.grid(True)
print(np.average(sim05.R_data[-1]))

print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
