'''Generates the data for the 9-box figure for the paper.'''
import Sim_no_update as Sim
import Pip

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
plt.switch_backend('Qt5Agg')

import time

class Nine_box_data(object):
    '''Creates and saves the data which will go in the figure. Note that
    this will only create the data for ONE fishing regime and ONE kind of
    graph structure (e.g. well-mixed, ring, lattice).
    Attributes:
    params = dict of parameter values, initial setup for the fisheries, 
             whatever is needed for Sim_no_update.py
    deltas = np array of delta values to be tested
    num_levels = int, number of different effort levels to be tested for
                 the PIP, i.e. the granularity of the PIP.
    prefix = string, specifies the kind of data being saved to the file,
             used to name files saved to computer, e.g. 'wellmixed', 'ring',
             'lattice'
    path = string, specifies location in computer to save data
    '''
    def __init__(self, params, deltas, num_levels, path, prefix):
        self.params = params
        self.deltas = deltas
        self.num_levels = num_levels
        self.path = path
        self.prefix = prefix
        self.make_data() # all the work is done in the constructor
        self.save_lists()

    def save_lists(self):
        '''Saves the data generated in make_data() to the *pathname* location.
        Namely, it saves *e_list*, *R_list*, *pi_list*, and *deltas*.'''
        e_fname = self.path + self.prefix + '-' + 'e_list'
        R_fname = self.path + self.prefix + '-' + 'R_list'
        pi_fname = self.path + self.prefix + '-' + 'pi_list'
        deltas_fname = self.path + 'deltas'
        np.savetxt(e_fname, self.e_list, fmt='%10.9f')
        np.savetxt(R_fname, self.R_list, fmt='%10.9f')
        np.savetxt(pi_fname, self.pi_list, fmt='%10.9f')
        np.savetxt(deltas_fname, self.deltas, fmt='%10.9f')

    def save_matrix(self, matrix, delta):
        '''Saves matrix which creates one PIP. Used in self.make_data().'''
        fname = self.path + self.prefix + '-pip-matrix-for-delta' + \
                str(delta).split('.')[0] + '_' + str(delta).split('.')[-1]
        np.savetxt(fname, matrix, fmt='%d')

    def make_data(self):
        '''Generates data for plot for one type of graph structure,
        e.g. well-mixed, lattice, or ring.'''
        e_list = [] # stores equil. effort level for different delta values
        R_list = [] # stores equil. resource level corresponding to equil. effort
        pi_list = [] # stores equil. payoff corresponding to equil. effort
        for delta in self.deltas:
            start1 = time.time()
            self.params['delta'] = delta
            pip1 = Pip.Pip(self.params, self.num_levels)
            pip1.create_matrix()
            self.save_matrix(pip1.get_matrix(), delta)
            e_list.append(pip1.e_eq)
            R_list.append(pip1.R_eq)
            pi_list.append(pip1.pi_eq)
            print("done: delta = {}".format(delta))
            end1 = time.time()
            print("---{} sec---".format(end1 - start1))
        self.e_list = e_list
        self.R_list = R_list
        self.pi_list = pi_list

def main():
    # initial parameters
    m = 4
    n = 4
    delta = 0 # will be changed later
    q = 2
    r = 0.05
    e_0 = 0 # will be assigned a meaningful value later
    R_0 = np.full((m, n), 0.5)
    p = 1
    w = 0.5
    num_feedback = 10
    copy_noise = 0.0005
    gm = False
    num_steps = 10
    wellmixed = False # NEED TO CHANGE THIS FOR THE WELL MIXED CASE
    params = {'m': m,'n': n, 'delta': delta, 'q': q, 'r': r, 'R_0': R_0,
              'e_0': e_0, 'p': p, 'w': w, 'num_feedback': num_feedback,
              'copy_noise': copy_noise, 'gm': gm, 'num_steps': num_steps,
              'wellmixed': False}
    num_levels = 10

    # choose delta values to be tested
    num_deltas = 5
    start = 0
    end = 0.5
    deltas = np.linspace(start, end, num=num_deltas, endpoint=False)

    # 16 fishers total
    # case 1: well-mixed
    params['wellmixed'] = True
    path = '/Users/alicelin/Documents/fish/fisheries-network/Figures/9box/trial1/'
    prefix = 'wellmixed'
    trial1 = Nine_box_data(params, deltas, num_levels, path, prefix)

    # case 2: ring
    m_ring = 16
    n_ring = 1
    params['m'] = m_ring
    params['n'] = n_ring
    params['wellmixed'] = False

    # case 3: nxn lattice
    params['m'] = m
    params['n'] = n
    params['wellmixed'] = False

if __name__ == '__main__':
    main()