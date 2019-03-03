'''Creates the 9-box figure using data generated and saved from
9box_data.py.'''

import Nine_box_data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
plt.switch_backend('Qt5Agg')

import time

import nine_box_setup # style for plot

def load_data(path):
    # loading data
    deltas = np.loadtxt(path + 'deltas')
    list_wm = [np.loadtxt(path + 'wellmixed-e_list'), np.loadtxt(path + 'wellmixed-R_list'),
               np.loadtxt(path + 'wellmixed-pi_list')]
    list_l = [np.loadtxt(path + 'lattice-e_list'), np.loadtxt(path + 'lattice-R_list'),
               np.loadtxt(path + 'lattice-pi_list')]
    list_r = [np.loadtxt(path + 'ring-e_list'), np.loadtxt(path + 'ring-R_list'),
               np.loadtxt(path + 'ring-pi_list')]
    return deltas, list_wm, list_l, list_r

if __name__ == '__main__':
    pathq1 = '/Users/alicelin/Documents/fish/fisheries-network/Figures/9box/PIP80x80/q1/'
    pathq2 = '/Users/alicelin/Documents/fish/fisheries-network/Figures/9box/PIP80x80/q2/'
    pathq06 = '/Users/alicelin/Documents/fish/fisheries-network/Figures/9box/PIP80x80/q0_6/'
    num_lines = 3 # number of lines on each plot

    # colors from colormap
    e_cmap = cm.get_cmap('Blues')
    R_cmap = cm.get_cmap('Purples')
    pi_cmap = cm.get_cmap('Greens')
    linecolors = []
    for x in np.linspace(0, 1, num=num_lines+1, endpoint=False)[1:]:
        colors = [e_cmap(x), R_cmap(x), pi_cmap(x)]
        linecolors.append(colors)

    # plotting
    fig, axs = plt.subplots(3,3, sharex='col', sharey='row')

    # plot sets of data in order
    paths = [pathq06, pathq1, pathq2]
    labels = ['$q = 0.6$', '$q = 1$', '$q = 2$']
    for j in range(len(paths)):
        deltas, list_wm, list_l, list_r = load_data(paths[j])
        for i in range(3):
            axs[i,0].plot(deltas, list_wm[i], color=linecolors[j][i])
            axs[i,1].plot(deltas, list_l[i], color=linecolors[j][i])
            axs[i,2].plot(deltas, list_r[i], color=linecolors[j][i], label=labels[j])
    
    #for i in range(3):
    #    axs[i,0].plot(deltas, list_wm[i], color=linecolors[0][i])
    #    axs[i,1].plot(deltas, list_l[i], color=linecolors[0][i])
    #    axs[i,2].plot(deltas, list_r[i], color=linecolors[0][i], label=r'$q = 1$')

    # # first set of data
    # deltas, list_wm, list_l, list_r = load_data(pathq1)
    # for i in range(3):
    #     axs[i,0].plot(deltas, list_wm[i], color=linecolors[0][i])
    #     axs[i,1].plot(deltas, list_l[i], color=linecolors[0][i])
    #     axs[i,2].plot(deltas, list_r[i], color=linecolors[0][i], label=r'$q = 1$')
    
    # deltas, list_wm, list_l, list_r = load_data(pathq2)
    # for i in range(3):
    #     axs[i,0].plot(deltas, list_wm[i], color=linecolors[1][i])
    #     axs[i,1].plot(deltas, list_l[i], color=linecolors[1][i])
    #     axs[i,2].plot(deltas, list_r[i], color=linecolors[1][i], label=r'$q = 2$')
    
    # deltas, list_wm, list_l, list_r = load_data(pathq06)
    # for i in range(3):
    #     axs[i,0].plot(deltas, list_wm[i], color=linecolors[2][i])
    #     axs[i,1].plot(deltas, list_l[i], color=linecolors[2][i])
    #     axs[i,2].plot(deltas, list_r[i], color=linecolors[2][i], label=r'$q = 0.6$')
    
    for i in range(3):
        # legend for all lines
        axs[i,2].legend(bbox_to_anchor=(1.02,1.02), loc='upper left')

    # labels and titles for plot
    # fig.suptitle(r'Equilibrium Values vs. $\delta$')
    axs[0,0].set_ylabel('effort') # intsxn pt of PIP
    axs[1,0].set_ylabel('resource') # avg over residents for situations near intsxn pt of PIP
    axs[2,0].set_ylabel('payoff') # per time step, avg over residents for situations near intsxn pt of PIP
    #axs[2,0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    #plt.ticklabel_format(axis='y', style='sci')
    #axs[2,1].ticklabel_format(axis='y', style='sci')
    axs[2,0].set_xlabel(r'$\delta$')
    axs[2,1].set_xlabel(r'$\delta$')
    axs[2,2].set_xlabel(r'$\delta$')
    axs[0,0].set_title('well-mixed')
    axs[0,1].set_title('lattice')
    axs[0,2].set_title('ring')
    fig.subplots_adjust(right=0.85, hspace=0.1, wspace=0.1)
    plt.show()
    