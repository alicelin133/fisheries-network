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
    def __init__(self, params, deltas, num_levels):
        self.params = params
        self.deltas = deltas
        self.num_levels = num_levels
        self.make_data()

    def save_lists(self):
        '''Saves the data generated in make_data() to the *pathname* location.'''

    def save_matrix(self):
        '''Saves matrix which creates one PIP.'''

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
            #e_vals = [pip1.e_msr, pip1.e_nash, pip1.res_levels[0], pip1.res_levels[-1]]
            e_list.append(pip1.e_eq)
            R_list.append(pip1.R_eq)
            pi_list.append(pip1.pi_eq)
            #save_matrix(path, pip1.get_matrix(), delta, e_vals)
            # TODO: save matrices for all of these to file
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
    num_levels = 40

    # choose delta values to be tested
    num_deltas = 25
    start = 0.5
    end = 1
    deltas = np.linspace(start, end, num=num_deltas, endpoint=False)

    # 16 fishers total
    # case 1: well-mixed
    params['wellmixed'] = True
    fname = '/Users/alicelin/Documents/fish/fisheries-network/Figures/9box/trial1/wellmixed'

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