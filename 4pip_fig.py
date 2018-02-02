"""Loads invasiion_matrix data from txt files saved by pip_make_fig_data.py
and plots 4 PIPs in one figure."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
plt.switch_backend('Qt5Agg')

def main():
    ds = [0,0,0,0]
    # load the corresponding matrices from their txt files
    path = '/Users/alicelin/Documents/fish/fisheries-network/Pairwise Invasibility Plots/80x80_data/'
    fnames = [path + 'd' + str(d).split('.')[-1] + 'matrix' for d in ds]
    # parse the header
    with open(fnames[0]) as f:
        line = f.readline()
        e_vals = np.fromstring(line[2:], sep=' ')
        e_msr, e_nash, e_start, e_end = e_vals
    matrices = [np.loadtxt(fname) for fname in fnames]

    # plot each matrix as a subplot
    fig, axs = plt.subplots(2, 2)
    ls = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        axs[ls[i]].imshow(matrices[i], cmap = cm.gray, extent = [e_start,
            e_end, e_start, e_end])
        axs[ls[i]].plot([e_msr], [e_msr], marker='o', markersize=2, color="red")
        axs[ls[i]].plot([e_nash], [e_nash], marker='o', markersize=2, color="red")
        axs[ls[i]].set_title(r'$\delta$ = {}'.format(ds[i]), fontsize=8)
        axs[ls[i]].set_xlabel(r'$e_{res}$', fontsize=7)
        axs[ls[i]].set_ylabel(r'$e_{mut}$', fontsize=7)
        axs[ls[i]].tick_params(axis='both', labelsize=5)
    fig.subplots_adjust(hspace=0.4, wspace=0.001)

if __name__ == '__main__':
    main()
    plt.show()