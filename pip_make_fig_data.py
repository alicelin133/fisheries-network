"""Creates and pickles the info for many PIPs."""

import Sim_no_update as Sim
import Pip

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
plt.switch_backend('Qt5Agg')

import time

def plot_e_vs_d(deltas, e_list):
    """Given the numpy array *deltas* and the list *e_list*, makes a plot."""
    fig, ax = plt.subplots()
    ax.plot(deltas, e_list)
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel('Stable effort level')
    ax.set_title(r'ESS vs. $\delta$')

def main():
    e_list = []
    matrix_list = []

    # Set parameters other than delta
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
    
    num_levels = 40

    # delta values to be tested
    num_deltas = 3
    deltas = np.linspace(0, 0.1, num=num_deltas)

    for delta in deltas:
        params['delta'] = delta
        pip1 = Pip.Pip(params, num_levels)
        pip1.create_matrix()
        e_list.append(pip1.get_eq())
        matrix_list.append(pip1.get_matrix())
    
    # save *e_list* and each matrix in *matrix_list*
    path = '/Users/alicelin/Documents/fish/fisheries-network/data/test-save/'
    e_list_fname = path + 'd' + str(deltas[0]).split('.')[1] + 'to' +
                    str(deltas[-1]).split('.')[1] + 'e_list'
    np.savetxt(e_list_fname, e_list)
    for i in range(len(matrix_list)):
        matrix_fname = path + 'd' + str(deltas[i]).split('.')[1] + 'matrix'
        np.savetxt(matrix_fname, matrix_list[i])

    # plot effort vs delta
    plot_e_vs_d(deltas, e_list)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("---{} sec---".format(end - start))
    plt.show()