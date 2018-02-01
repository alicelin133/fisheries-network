"""Creates and pickles the info for many PIPs."""

import Sim_no_update as Sim
import Pip

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
plt.switch_backend('Qt5Agg')

import time

def plot_e_vs_d(deltas, e_list, e_msr, e_nash):
    """Given the numpy array *deltas* and the list *e_list*, makes a plot."""
    fig, ax = plt.subplots()
    ax.plot(deltas, e_list)
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel('Stable effort level')
    ax.set_title(r'ESS vs. $\delta$')
    ax.axhline(y=e_msr, linestyle='--', color='c', linewidth=0.5)
    ax.axhline(y=e_nash, linestyle='--', color='c', linewidth=0.5)
    ax.text(deltas[-1], e_msr * 1.01, '$e_{MSR}$', va='bottom', ha='right', color='c')
    ax.text(deltas[0], e_nash * 0.99, '$e_{Nash}$', va='top', color='c')

def main():
    e_list = []
    matrix_list = []

    # set parameters other than delta
    m = 6
    n = 6
    delta = 0 # will be changed later
    q = 1
    r = 0.05
    e_0 = 0 # will be assigned a meaningful value later
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
    
    num_levels = 20

    # choose delta values to be tested
    num_deltas = 2
    start = 0
    end = 0.005
    deltas = np.linspace(start, end, num=num_deltas, endpoint=False)

    # generate data
    for delta in deltas:
        params['delta'] = delta
        pip1 = Pip.Pip(params, num_levels)
        e_msr = pip1.e_msr
        e_nash = pip1.e_nash
        pip1.create_matrix()
        e_list.append(pip1.get_eq())
        matrix_list.append(pip1.get_matrix())
    
    # save *e_list* and each matrix in *matrix_list*
    path = '/Users/alicelin/Documents/fish/fisheries-network/Pairwise Invasibility Plots/80x80_data/'
    e_list_fname = path + 'd' + str(start).split('.')[-1] + 'to' + \
                    str(end).split('.')[-1] + 'e_list'
    np.savetxt(e_list_fname, e_list, fmt='%9.8f')
    for i in range(len(matrix_list)):
        matrix_fname = path + 'd' + str(deltas[i]).split('.')[1] + 'matrix'
        np.savetxt(matrix_fname, matrix_list[i], fmt='%d')

    # plot effort vs delta
    plot_e_vs_d(deltas, e_list, e_msr, e_nash)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("---{} sec---".format(end - start))
    plt.show()