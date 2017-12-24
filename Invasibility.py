"""Modifies the Simulation_2d_arrays object to create invasibility plots
in which an invader with a different effort level is placed in a lattice
otherwise filled with residents of a uniform effort level."""

import Sim_no_update as Sim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.switch_backend('Qt5Agg')

import time

def main():
    # set parameters
    m = 8
    n = 8
    delta = 0
    q = 1
    r = 0.2
    R_0 = np.full((m, n), 0.5)
    p = 1
    w = 0.5
    num_feedback = 10
    copy_noise = 0.0005
    gm = False
    num_steps = 20

    # Assign efforts
    e_msr = Sim.calculate_e_msr(m, n, q, r, p, w)
    e_nash = Sim.calculate_e_nash(e_msr, m, n)
    # Range of efforts used for mutant/resident strategies
    num_levels = 20
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
            # create and run the simulation
            mysim = Sim.Sim_no_update(m, n, delta, q, r, R_0, e_0, p, w,
                num_feedback, copy_noise, gm, num_steps)
            mysim.run_sim()
            isInvadable[i][j] = bool(mysim.pi_data[-1][mutant] > np.mean(mysim.pi_data[-1]))
    print(isInvadable)
    
    # Create pairwise invasibility plot
    fig, ax = plt.subplots()
    plt.imshow(isInvadable, cmap = cm.gray)
    ax.set_xlabel("Resident effort level")
    ax.set_ylabel("Mutant effort level")

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("---{} sec---".format(end - start))
    plt.show()