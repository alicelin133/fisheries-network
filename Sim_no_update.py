import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

import time

class Sim_no_update(object):
    """A grid network of territories, where each territory harvests fish,
    and fish move between territories along edges. The network is implicit.
    Data attributes:
        m = number of rows in lattice of fishers
        n = number of columns in lattice of fishers
        delta = level of fish movement in [0,1]
        q = "catchability" of fish resource
        r = growth factor of fish within territory
        R_0 = mxn numpy array, initial resource levels for each fisher
        e_0 = mxn numpy array, initial effort levels for each fishers
        p = price per unit of harvest
        w = cost per unit of effort
        num_feedback: number cycles of harvest/regrowth/leakage per time step
        gm = boolean, True = global mutation, False = local mutation
        num_steps = number of time steps that the simulation runs
        R_t = mxn numpy array, current resource levels for each fisher
        e_t = mxn numpy array, current effort levels for each fisher
        R_data = mxnxnum_steps np array, saves R_t for all t
        e_data = mxnxnum_steps np array, saves e_t for all t
        neighbors = mxnxDEGREEx2 np array, index [i][j] returns fisher's 4
            neighbor coordinates
    """
    def __init__(self, params):
        """Creates a Simulation_2d_arrays object."""
        self.DEGREE = 4 # can change this later
        self.m = params['m']
        self.n = params['n']
        self.delta = params['delta']
        self.q = params['q']
        self.r = params['r']
        self.R_0 = np.copy(params['R_0']) # used for naming files later
        self.e_0 = np.copy(params['e_0'])
        self.p = params['p']
        self.w = params['w']
        self.num_feedback = params['num_feedback']
        self.copy_noise = params['copy_noise']
        self.gm = params['gm']
        self.num_steps = params['num_steps']
        self.R_t = np.copy(params['R_0'])
        self.e_t = np.copy(params['e_0'])
        self.pi_t = np.zeros((self.m,self.n))
        self.R_data = np.zeros((self.num_steps, self.m, self.n))
        self.e_data = np.zeros((self.num_steps, self.m, self.n))
        self.pi_data = np.zeros((self.num_steps, self.m, self.n))
        self.neighbors = np.zeros((self.m, self.n, self.DEGREE, 2), dtype=int)
        self.get_neighbors()
        self.dR = np.zeros((self.m, self.n)) # used in leakage()
        self.R_from_neighbors = np.zeros((self.m, self.n)) # used in leakage()

    def get_neighbors(self):
        """Fills the self.neighbors array with the neighbors. So
        that neighbors[i][j] returns an array of the neighbors of
        the fisher at (i,j)."""
        for i in range(self.m):
            for j in range(self.n):
                neighbors = np.zeros((self.DEGREE,2), dtype=int)
                neighbors[0] = (i, (j + 1) % self.n)
                neighbors[1] = (i, (j - 1) % self.n)
                neighbors[2] = ((i + 1) % self.m, j)
                neighbors[3] = ((i - 1) % self.m, j)
                self.neighbors[i][j] = neighbors

    def run_sim(self):
        """Runs a discrete simulation of the fisheries network for
        self.num_steps time steps."""
        for t in range(self.num_steps):
            self.save_data(t)
            self.update_resource()

    def save_data(self, t):
        """Stores current data arrays."""
        self.R_data[t] = self.R_t
        self.e_data[t] = self.e_t
        self.pi_data[t] = self.pi_t

    def update_resource(self):
        """Runs the harvest, regrowth, and leakage functions self.num_feedback
        number of times."""
        for i in range(self.num_feedback):
            self.harvest()
            self.regrowth()
            self.leakage()

    def harvest(self):
        """Updates R_t based on the efforts e_t of the fishers.
        Also calculates payoffs for each fisher."""
        harvest = self.R_t * self.q * self.e_t
        self.R_t = self.R_t - harvest
        self.get_payoff(harvest)

    def get_payoff(self, harvest):
        """Calculates payoff for each fisher, where pi = (pqR - w)e"""
        self.pi_t = self.p * harvest - self.w * self.e_t

    def regrowth(self):
        """Updates R_t based on the logistic growth of fish."""
        self.R_t = self.R_t * (1 + self.r * (1 - self.R_t))

    def leakage(self):
        """Updates R_t based on fish movement across edges of the lattice."""
        self.dR = self.delta * self.R_t / (self.DEGREE + 1)
        for i in range(self.m):
            for j in range(self.n):
                n = self.neighbors[i][j]
                dR = self.dR[i][j]
                for k in range(self.DEGREE):
                    self.R_from_neighbors[n[k][0],n[k][1]] += dR
                    # TODO: NEED TO TEST THIS
        self.R_t = self.R_t - self.DEGREE * self.dR + self.R_from_neighbors
        self.R_from_neighbors = np.zeros((self.m,self.n))

def calculate_e_msr(m, n, q, r, price, cost):
    """Calculates value of e_msr (maximum sustainable rent)."""
    return r * (price * q * n * m - n * m * cost) / (2 * price * q * q * n * m)

def calculate_e_nash(e_msr, m, n):
    """Calculates value of Nash equilibrium level of effort."""
    return e_msr * 2 * m * n / (1 + m * n)

def main():
    """Testing."""
    start_time = time.time()
    
    # set seed for pseudo RNG
    seed = 17
    np.random.seed(seed)
    # set parameter values
    m = 10
    n = 10
    delta = 1
    q = 1
    r = 0.2
    R_0 = np.full((m, n), 0.5)
    p = 1
    w = 0.5
    num_feedback = 15
    copy_noise = 0.0005
    gm = False
    num_steps = 51
    # creating e_0
    e_msr = calculate_e_msr(m, n, q, r, p, w)
    e_nash = calculate_e_nash(e_msr, m, n)
    e_0 = np.linspace(0, e_nash, num=m * n)
    e_0 = e_0.reshape((m,n))
    print("e_msr: {}".format(e_msr))
    print("e_nash: {}".format(e_nash))

    params = {'m': m,'n': n, 'delta': delta, 'q': q, 'r': r, 'R_0': R_0,
              'e_0': e_0, 'p': p, 'w': w, 'num_feedback': num_feedback,
              'copy_noise': copy_noise, 'gm': gm, 'num_steps': num_steps}

    mysim = Sim_no_update(params)
    mysim.run_sim()
    print("avg e: {}".format(np.mean(mysim.e_data[50:])))
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(np.arange(num_steps), np.average(mysim.e_data, axis=(1,2)))
    ax1.set_title("Effort")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Average effort")
    ax1.grid(True)

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(np.arange(num_steps), np.average(mysim.R_data, axis=(1,2)))
    ax2.set_title("Resource")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Average resource")

    fig.subplots_adjust(wspace=0.3, hspace=0.5)

    plt.show()

if __name__ == '__main__':
    main()
                