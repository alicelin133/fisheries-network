"""Creates and runs many Sim_no_update objects to create an invasibility plot
in which a mutant with a different effort level is placed in a lattice
otherwise filled with residents of a uniform effort level, then compares payoffs
of the mutant and the average resident to make PIP."""

import Sim_no_update as Sim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.switch_backend('Qt5Agg')

import time

def main():
    # set parameters
    m = 6
    n = 6
    delta = 0.99
    q = 1
    r = 0.05
    e_0 = 0
    R_0 = np.full((m, n), 0.5)
    p = 1
    w = 0.5
    num_feedback = 10
    copy_noise = 0.0005
    gm = False
    num_steps = 10

    params = {'m': m,'n': n, 'delta': delta, 'q': q, 'r': r, 'R_0': R_0,
              'e_0': e_0, 'p': p, 'w': w, 'num_feedback': num_feedback,
              'copy_noise': copy_noise, 'gm': gm, 'num_steps': num_steps}

    # Assign efforts
    e_msr = Sim.calculate_e_msr(m, n, q, r, p, w)
    e_nash = Sim.calculate_e_nash(e_msr, m, n)
    print("e_nash: {}".format(e_nash))
    # Range of efforts used for mutant/resident strategies
    num_levels = 25
    res_levels = np.linspace(e_msr, r/q, num=num_levels, endpoint=True)
    mut_levels = np.flip(res_levels, 0)
    isInvadable = np.zeros((num_levels, num_levels)) # (i,j) can mutant i invade resident j

    # Compute pairwise invasibility
    for i in range(num_levels): # mutant
        for j in range(num_levels): # resident
            e_0 = np.full((m, n), res_levels[j]) # resident strategy
            e_0 = e_0.reshape((m,n))
            mutant = (int(m/2), int(n/2))
            e_0[mutant] = mut_levels[i] # mutant strategy
            params['e_0'] = e_0
            # create and run the simulation
            mysim = Sim.Sim_no_update(params)
            mysim.run_sim()
            isInvadable[i][j] = bool(mysim.pi_data[-1][mutant] > np.mean(mysim.pi_data[-1]))
    print(isInvadable)
    
    # Create pairwise invasibility plot
    fig, ax = plt.subplots()
    cax = ax.imshow(isInvadable, cmap = cm.gray, extent = [res_levels[0],
        res_levels[-1], mut_levels[-1], mut_levels[0]])
    ax.set_xlabel("Resident effort level")
    ax.set_ylabel("Mutant effort level")
    ax.set_title("Pairwise Invasibility Plot, delta = {}".format(delta))

    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Residents win', 'Mutant wins'])

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("---{} sec---".format(end - start))
    plt.show()