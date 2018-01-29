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

class Pip(object):
    """Creates a pairwise invasibility plot.
    params = dict of parameter values
    num_levels = int, number of effort levels to test (amt of detail in plot)
    """
    def __init__(self, params, num_levels):
        self.params = params
        self.num_levels = num_levels
        m = params['m']
        n = params['n']
        q = params['q']
        r = params['r']
        p = params['p']
        w = params['w']
        # create array of efforts to be tested, between e_msr and r/q
        self.e_msr = Sim.calculate_e_msr(m, n, q, r, p, w)
        self.e_nash = Sim.calculate_e_nash(self.e_msr, m, n)
        # TODO: REMEMBER THAT YOU CHANGED THE LOWER LIMIT FOR RES LEVELS BELOW -->
        self.res_levels = np.linspace(self.e_msr * 0.8, r/q, num=num_levels, endpoint=True)
        self.mut_levels = np.flip(self.res_levels, 0)
        self.mutant = (int(m/2), int(n/2)) # needed in self.create_matrix()
        # stores PIP data
        self.invasion_matrix = np.empty((num_levels, num_levels))
    
    def assign_efforts(self, efforts):
        """By default, the efforts range between e_msr and r/q, the maximum
        sustainable effort level. But it is possible to change these if
        desired, using this method."""
        self.res_levels = efforts
        self.mut_levels = np.flip(self.res_levels, 0)

    def create_matrix(self):
        """For each possible pair of mutant/resident effort levels,
        creates and runs a Sim_no_update object. Creates a matrix where (i,j)
        corresponds to a coordinate in a plot, 1 for mutant having higher payoff,
        0 for resident having equal or higher payoff. The matrix is saved as
        *invasion_matrix* attribute."""
        for i in range(self.num_levels): # mutant
            for j in range(self.num_levels): # resident
                e_0 = self.make_e_0(i, j)
                self.params['e_0'] = e_0
                sim1 = Sim.Sim_no_update(self.params)
                sim1.run_sim()
                self.invasion_matrix[i,j] = bool(sim1.pi_data[-1][self.mutant] > np.mean(sim1.pi_data[-1]))

    def make_e_0(self, i, j):
        """Helper method, returns custom e_0 parameter for the Sim_no_update
        object created to fill position (i,j) in *self.invasion_matrix*.
        In this e_0 2D np array, all positions have value *self.res_levels[j]*
        except for the position (m/2, n/2), which has value
        *self.mut_levels[i]*."""
        m = self.params['m']
        n = self.params['n']
        e_0 = np.full((m, n), self.res_levels[j]) # resident strategy
        e_0 = e_0.reshape((m, n))
        self.mutant = (int(m/2), int(n/2))
        e_0[self.mutant] = self.mut_levels[i] # mutant strategy
        return e_0

    def get_matrix(self):
        """Returns the *invasion_matrix* attribute."""
        return self.invasion_matrix

    def plot_pip(self):
        """Plots *self.invasion_matrix* using a colormap. Black = residents win
        White = mutant wins."""
        fig, ax = plt.subplots()
        cax = ax.imshow(self.invasion_matrix, cmap = cm.gray, extent = [self.res_levels[0],
            self.res_levels[-1], self.mut_levels[-1], self.mut_levels[0]])
        ax.set_xlabel("Resident effort level")
        ax.set_ylabel("Mutant effort level")
        ax.set_title("Pairwise Invasibility Plot, delta = {}".format(self.params['delta']))
        cbar = fig.colorbar(cax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Residents win', 'Mutant wins'])

def main():
    # Set parameters
    m = 6
    n = 6
    delta = 0.001
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

    # Create and plot Pip object
    pip1 = Pip(params, num_levels)
    print(pip1.e_msr)
    efforts = np.linspace(0, pip1.params['r'] / pip1.params['q'],
                          num=num_levels)
    pip1.assign_efforts(efforts)
    pip1.create_matrix()
    pip1.plot_pip()

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("---{} sec---".format(end - start))
    plt.show()