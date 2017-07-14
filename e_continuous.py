"""Uses Simulation class to run a simulation given the parameters below."""

import numpy as np
import matplotlib.pyplot as plt
import time
import Simulation

start_time = time.time()

# Assigning parameter values
n_fishers = 100
delta = 0
r = 0.01
q = 1
K = 5000 / n_fishers
K_tot = 5000
price = 1
cost = 0
noise = 0.0005
num_regrowth = 2
num_steps = 500
R_0 = np.full(n_fishers,K)
# creating initial distribution of effort
e_msy = r * (price * q * K_tot - n_fishers * cost) / (2 * price * q * q * K_tot)
print("e_msy = {}".format(e_msy))
e_nash = e_msy * 2 * n_fishers / (1 + n_fishers)
print("e_nash = {}".format(e_nash))
e_0 = np.linspace(0, e_nash, n_fishers)


# Creating Simulation object
sim = Simulation.Simulation(n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise, num_regrowth)
sim.simulate(num_steps)
fig = plt.figure()
plt.suptitle("No fish movement")

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
ax3.set_yticks(ax3.get_yticks())
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
R_msy = K * (1 - q * e_msy / r)
R_nash = K * (1 - q * e_nash / r)
print("Theoretical R for e_msy: {}".format(R_msy))
print("Theoretical R for e_nash: {}".format(R_nash))
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
