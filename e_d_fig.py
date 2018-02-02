"""Loads data generated from pip_make_fig_data.py and concatenates
the data to plot equilibrium effort as a function of delta."""

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

def main():
    path = '/Users/alicelin/Documents/fish/fisheries-network/Pairwise Invasibility Plots/80x80_data/'
    intervals = ['0to1', '1to2']
    fnames_e = [path + 'd' + s + 'e_list' for s in intervals]
    fnames_d = [path + 'd' + s + 'deltas' for s in intervals]
    # extract values of e_msr and e_nash in header of txt files
    with open(fnames_e[0]) as f:
        line = f.readline()
        e_vals = np.fromstring(line[2:], sep=' ')
        e_msr, e_nash = e_vals
    # load values for efforts and deltas
    es = [np.loadtxt(fname) for fname in fnames_e]
    ds = [np.loadtxt(fname) for fname in fnames_d]
    e = np.concatenate(es)
    d = np.concatenate(ds)
    # plot concatenated values
    fig, ax = plt.subplots()
    ax.plot(d, e)
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel('Stable effort level')
    ax.set_title(r'ESS vs. $\delta$')
    # add hlines marking e_msr and e_nash
    ax.axhline(y=e_msr, linestyle='--', color='c', linewidth=0.5)
    ax.axhline(y=e_nash, linestyle='--', color='c', linewidth=0.5)
    ax.text(d[-1], e_msr * 1.01, '$e_{MSR}$', va='bottom', ha='right', color='c')
    ax.text(d[0], e_nash * 0.99, '$e_{Nash}$', va='top', color='c')

if __name__ == '__main__':
    main()
    plt.show()