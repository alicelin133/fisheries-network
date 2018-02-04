"""Loads invasiion_matrix data from txt files saved by pip_make_fig_data.py
and plots 4 PIPs in one figure."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.patches as mpatches
plt.switch_backend('Qt5Agg')

def main():
    ds = [0,0.02,0.1,0.4]
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
    white = (1, 1, 1)
    green = (120/255,209/255,110/255, 0.5)
    mycmap = colors.ListedColormap([white, green], name='mycmap')
    for i in range(4):
        axs[ls[i]].imshow(matrices[i], cmap=mycmap, extent = [e_start,
            e_end, e_start, e_end])
        axs[ls[i]].plot([e_msr], [e_msr], marker='x', markersize=2, color='black')
        axs[ls[i]].plot([e_nash], [e_nash], marker='x', markersize=2, color='black')
        axs[ls[i]].set_title(r'$\delta$ = {}'.format(ds[i]), fontsize=8)
        axs[ls[i]].set_xlabel(r'$e_{res}$', fontsize=8)
        axs[ls[i]].set_ylabel(r'$e_{mut}$', fontsize=8)
        axs[ls[i]].tick_params(axis='both', labelsize=6)
    fig.subplots_adjust(hspace=0.4, wspace=0.001)

if __name__ == '__main__':
    main()
    plt.show()