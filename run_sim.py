"""Uses Simulation class to run a simulation given the parameters below."""

import numpy as np
import matplotlib.pyplot as plt
import time
import Simulation

start_time = time.time()

# Assigning parameter values
n_fishers = 10
delta = 1
q = 1
r = 0.01
K = 5000
price = 1
cost = 0.5
noise = 0
num_steps = 10000
R_0 = np.full(n_fishers,K)
# creating initial distribution of cooperators and defectors
f_c = 1
e_c = r * (price * q * K - cost) / (2 * price * q * q * K * n_fishers)
e_d = e_c * 2 * n_fishers / (n_fishers + 1)
num_c = round(f_c * n_fishers)
num_d = n_fishers - num_c
e_0 = np.concatenate((np.full(num_c, e_c), np.full(num_d, e_d)), axis=0)

# Creating Simulation object
sim = Simulation.Simulation(n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise)
sim.simulate(num_steps)
fig = plt.figure()
plt.suptitle("Full fish movement")

# Plotting resource levels vs. time
ax1 = fig.add_subplot(2,2,1)
for i in range(sim.n_fishers):
    ax1.plot(np.arange(num_steps), sim.R_data[i])
ax1.set_xlabel("Time step")
ax1.set_ylabel("Resource (K = {})".format(sim.K))
ax1.set_title("Territory Resource Levels vs. Time")
ax1.grid(True)

# Plotting avg payoff vs. time
pi_avg = np.average(sim.pi_data, axis=0)
ax2 = fig.add_subplot(2,2,2)
ax2.plot(np.arange(num_steps), pi_avg)
ax2.set_xlabel("Time steps")
ax2.set_ylabel("Average payoff")
ax2.set_title("Average Payoff vs. Time")
ax2.grid(True)

# Plotting avg effort vs. time
e_avg = np.average(sim.e_data, axis=0)
ax3 = fig.add_subplot(2,2,3)
ax3.plot(np.arange(num_steps), e_avg)
ax3.set_xlabel("Time steps")
ax3.set_ylabel("Effort")
ax3.set_title("Average Effort vs. Time")
extraticks = [e_c, e_d]
ax3.set_yticks(extraticks)
ax3.grid(True)

# Plotting all efforts vs. time
ax4 = fig.add_subplot(2,2,4)
for i in range(sim.n_fishers):
    ax4.plot(sim.e_data[i])
ax4.set_xlabel("Time steps")
ax4.set_ylabel("Effort")
ax4.set_title("Effort vs. Time")
ax4.grid(True)
fig.subplots_adjust(wspace=0.3, hspace=0.4)

print("Last time step avg payoff: {}".format(pi_avg[-1]))
print("Last time step avg effort: {}".format(e_avg[-1]))
print("Last time step avg R: {}".format(np.average(sim.R_data[-1])))
R_theory = K * (1 - q * e_d * n_fishers / r)
print("Theoretical R: {}".format(R_theory))
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
