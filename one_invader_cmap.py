"""Creates a colormap of each fisher's final payoff after many steps,
where an invader with a different effort level is placed in a lattice
otherwise filled with residents of a uniform effort level."""

import Sim_no_update as Sim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.switch_backend('Qt5Agg')

import time

def main():
    # set seed for pseudo RNG
    seed = 17
    np.random.seed(seed)
    # set parameter values
    m = 100
    n = 1
    delta = 1
    q = 1
    r = 0.2
    R_0 = np.full((m, n), 0.5)
    p = 1
    w = 0.5
    num_feedback = 15
    copy_noise = 0.0005 # not actually used here
    gm = False
    num_steps = 100
    wellmixed = False # False means that the fish move along the lattice i.e. not well-mixed
    # creating e_0
    e_msr = Sim.calculate_e_msr(m, n, q, r, p, w)
    e_nash = Sim.calculate_e_nash(e_msr, m, n)
    e_0 = np.full((m,n), e_msr)
    mid = (int(m/2), int(n/2))
    e_0[mid] = e_nash # one invader has strategy e_nash

    params = {'m': m,'n': n, 'delta': delta, 'q': q, 'r': r, 'R_0': R_0,
              'e_0': e_0, 'p': p, 'w': w, 'num_feedback': num_feedback,
              'copy_noise': copy_noise, 'gm': gm, 'num_steps': num_steps,
              'wellmixed': wellmixed}

    mysim = Sim.Sim_no_update(params)
    mysim.run_sim()
    
    # making colormap of payoffs
    print("Nash: {} MSR: {}".format(e_nash, e_msr))
    print(mysim.pi_data[-1])
    print(mysim.pi_data[-1][mid])
    data = mysim.pi_data[-1]
    fig, ax = plt.subplots()
    ax.set_title("Payoff after {} steps, e_nash invader vs e_msr residents".format(num_steps))
    cax = ax.imshow(data, interpolation='nearest', cmap=cm.coolwarm)
    cbar = fig.colorbar(cax, ticks=[np.amin(data), np.amax(data)])
    cbar.ax.set_yticklabels(['{0:.5f}'.format(np.amin(data)), '{0:.5f}'.format(np.amax(data))])  # vertically oriented colorbar

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("---{} sec---".format(end - start))
    plt.show()