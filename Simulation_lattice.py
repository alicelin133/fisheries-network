import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

import time

class Simulation_lattice(object):
    """A grid network of territories of fishers, where each territory harvests fish,
    and fish move between territories. The network has the following properties:
    Network attributes:
        G: grid graph in networkx, each node corresponds to a territory, each edge
            corresponds to a connection b/w territories that fish can leak through
        H: complete graph in networkx, each node corresponds to a territory, each edge
            corresponds to a possible neighbor to which a territory could
            compare payoffs.
        network_dims: list, the dimensions of the network
        n_fishers: the number of territories in the total network
        delta: float, extent to which territories are ecologically
            connected
        q: float, "catchability" of fish resource
        r: float, growth factor of fish population 
        K: float, carrying capacity of fish population
        price: float, price received per unit of fish
        cost: float, cost per unit of effort invested in fishing
        noise: float, switching strategies occurs with +/- *noise*
        num_feedback: int, number of cycles of harvest/regrowth/leakage per time step
        R_data: np array, R_data[i][t] = resource of territory i at time t
        e_data: np array, e_data[i][t] = effort of territory i at time t
        U_data: np array, U_data[i][t] = utility of territory i at time t
    Node (territory) attributes:
        R: float, territory's resource level
        e: float, territory's effort level
        dR: float, for internal use in calculating leakage.
        payoffs: np.array of floats, territory's payoffs from each iteration of feedback loop
        U: float, territory's utility from current time step
    """

    def __init__(self, network_dims, delta, q, r, K, R_0, e_0, price, cost, noise, num_feedback, num_steps):
        """Return a Simulation object with a grid graph with *network_dim* dimensions,
        a leakage factor of *delta*, a fish catchability factor of *q*, and
        individual nodes with *R_0[i]* resource level and *e_0[i]* effort level."""
        self.check_sim_attributes(network_dims, delta, q, r, K, R_0, e_0, price, cost, noise)
        self.G = nx.grid_graph(dim=network_dims, periodic=True)
        self.network_dims = network_dims
        self.n_fishers = self.get_prod(network_dims)
        self.H = nx.complete_graph(self.n_fishers)
        self.delta = delta
        self.q = q
        self.r = r
        self.K = K
        self.price = price
        self.cost = cost
        self.noise = noise
        self.num_feedback = num_feedback
        self.R_0 = np.copy(R_0)
        self.e_0 = np.copy(e_0)
        for nood in self.G.nodes(data=False):
            self.G.node[nood]['R'] = self.R_0[self.get1D(nood)]
            self.G.node[nood]['e'] = self.e_0[self.get1D(nood)]
            self.G.node[nood]['dR'] = 0.0
            self.G.node[nood]['U'] = 0.0
            self.G.node[nood]['e_new'] = 0.0
        self.num_steps = num_steps
        self.e_data = np.zeros((self.n_fishers, num_steps))
        self.R_data = np.zeros((self.n_fishers, num_steps))
        self.U_data = np.zeros((self.n_fishers, num_steps))
    
    def get_prod(self, num_list):
        """Returns the product of the elements in *list*. Used in
        constructor to determine value of n_fishers from grid graph
        dimensions."""
        product = 1
        for i in num_list:
            product = product * i
        return product
    
    def get1D(self, coord_pair):
        """Given an ordered pair, convert to an integer identifier."""
        return coord_pair[0] * self.network_dims[0] + coord_pair[1]

    def HtoG(self, node_id):
        """Given a node identifier for strategy graph H, return the
        corresponding node in ecology graph G."""
        coord1 = node_id % self.network_dims[0]
        coord0 = (node_id - coord1) / self.network_dims[0]
        return (coord0, coord1)

    def check_sim_attributes(self, network_dims, delta, q, r, K, R_0, e_0, price, cost, noise):
        """Checks that values of data attributes given in parameters to 
        constructor are in the correct format."""
        for dim in network_dims:
            if dim < 0 or not isinstance(dim, int):
                raise ValueError("network_dims must be positive integer values")
        if delta < 0 or delta > 1:
            raise ValueError("delta must be in [0,1]")
        if q < 0 or q > 1:
            raise ValueError("q must be in [0,1]")
        if R_0.size != self.get_prod(network_dims):
            raise ValueError("R_0 length must match self.n_fishers")
        if e_0.size != self.get_prod(network_dims):
            raise ValueError("e_0 length must match self.n_fishers")
        for i in range(e_0.size):
            if e_0[i] < 0 or e_0[i] > 1:
                raise ValueError("entry {} in e_0 must be in [0,1]".format(i))
        
    def simulate(self):
        """Runs a discrete simulation of the fisheries network for self.num_steps
        time steps."""
        # e_data[nood][t] = e of node nood at time step t
        for t in range(self.num_steps):
            for nood in self.G.nodes(data=False):
                self.e_data[self.get1D(nood)][t] = self.G.node[nood]['e']
                self.R_data[self.get1D(nood)][t] = self.G.node[nood]['R']
                self.U_data[self.get1D(nood)][t] = self.G.node[nood]['U']
            self.update_resource()
            self.update_strategy()

    def update_resource(self):
        """Runs the harvest(), regrowth(), and leakage() methods in a loop
        self.num_feedback number of times. Also calculates the utility for
        each fisher based on payoffs from all self.num_feedback harvests
        combined."""
        for i in range(self.num_feedback):
            self.harvest(i)
            self.regrowth()
            self.leakage()

    def harvest(self, num_iteration):
        """Updates resource R for each territory based on e, the territory's
        effort level, for a single time step. Also calculates payoff from
        harvest."""
        for nood in self.G.nodes(data=False):
            R = self.G.node[nood]['R']
            e = self.G.node[nood]['e']
            harvest = self.q * R * e
            if harvest > R: # case 1: fisher wants more fish than he could take
                harvest = R
                self.G.node[nood]['R'] = 0
                print("Resource is {} for node {}".format(R, nood))
            else: # case 2: fisher wants less fish than he could take
                self.G.node[nood]['R'] = R - harvest
            # Only save the payoff if it's the last iteration of the time step
            if num_iteration == self.num_feedback - 1:
                self.G.node[nood]['U'] = self.price * harvest - self.cost * e

    def regrowth(self):
        """Updates resource R for each territory based on the fish population's
        logistic growth, occurs after the time step's harvest."""
        for i in self.G.nodes(data=False):
            R = self.G.node[i]['R']
            self.G.node[i]['R'] = R + self.r * R * (1 - R/self.K)
            
    def leakage(self):
        """Updates resource R for each territory based on resource leakage
        between territories. Each territory i gives (delta * R / self.G.degree(i))
        amount of resource to each of its adjacent territories."""
        # calculate how much fish go to each territory
        for nood in self.G.nodes(data=False):
            R = self.G.node[nood]['R']
            deg = self.G.degree(nood)
            for neighbor in self.G[nood]:
                self.G.node[neighbor]['dR'] += R / (deg + 1)
        # update resource levels for each territory
        for nood in self.G.nodes(data=False):
            R = self.G.node[nood]['R']
            deg = self.G.degree(nood)
            dR = self.G.node[nood]['dR']
            R_new = R * (1 - self.delta * deg / (deg + 1)) + self.delta * dR
            self.G.node[nood]['R'] = R_new
            self.G.node[nood]['dR'] = 0 # RESET dR TO 0 FOR EACH NODE
    
    def update_strategy(self):
        """Selects one fisher randomly for possible strategy change. With
        small probability, new strategy is randomly chosen from [0,1].
        Otherwise, select a second fisher to compare payoff against. If 2nd
        fisher has lower payoff, do nothing. If 2nd fisher has higher payoff,
        1st fisher switches to effort level of 2nd fisher with prob.
        proportional to difference of payoffs, and with some noise. Repeat
        process self.n_fishers times."""
        for nood in self.G.nodes(data=False): # initialize e_new array
            self.G.node[nood]['e_new'] = self.G.node[nood]['e']
        for i in range(self.n_fishers):
            index1 = np.random.randint(self.n_fishers)
            fisher1 = self.H.nodes(data=False)[index1]
            index2 = np.random.randint(self.H.degree(fisher1))
            fisher2 = self.H.neighbors(fisher1)[index2]
            U1 = self.G.node[self.HtoG(fisher1)]['U']
            # global mutation
            prob_mutation = 0
            rand = np.random.random()
            if rand < prob_mutation:
                self.G.node[self.HtoG(fisher1)]['e_new'] = np.random.random()
            else: # switching strategies based on payoff
                U2 = self.G.node[self.HtoG(fisher2)]['U']
                if U1 < U2:
                    if U1 != 0 or U2 != 0:
                        # Probability that fisher1 switches to fisher2's strategy
                        prob_switch = (U2 - U1) / (abs(U1) + abs(U2))
                    else:
                        prob_switch = 0 # in case both payoffs are 0
                    rand = np.random.random()
                    if rand < prob_switch:
                        diff = np.random.uniform(-1 * self.noise, self.noise)
                        e_new = self.G.node[self.HtoG(fisher2)]['e'] + diff
                        # ensure that e_new stays in [0,1]
                        if e_new < 0:
                            e_new = 0
                        elif e_new > 1:
                            e_new = 1
                        self.G.node[self.HtoG(fisher1)]['e_new'] = e_new
        for i in self.G.nodes(data=False):
            self.G.node[i]['e'] = self.G.node[i]['e_new']

def calculate_e_msr(n_fishers, q, r, K, price, cost):
    """Calculates value of e_msr (maximum sustainable rent)."""
    return r * (price * q * K * n_fishers - n_fishers * cost) / (2 * price * q * q * K * n_fishers)

def calculate_e_nash(e_msr, n_fishers):
    """Calculates value of Nash equilibrium level of effort."""
    return e_msr * 2 * n_fishers / (1 + n_fishers)

def main():
    """Performs unit testing."""
    start_time = time.time()

    # Setting seed for pseudo-RNG
    seed = 17
    np.random.seed(seed)   

    # Parameters: n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise,
    #   num_feedback, num_steps
    network_dims = [5,5]
    n_fishers = network_dims[0] * network_dims[1]
    delta = 1
    q = 1
    r = 0.06
    K = 200
    R_0 = np.full(n_fishers, K/2)
    # e_0 = np.random.random(n_fishers) * r/q
    e_0 = np.linspace(0, r/q, n_fishers)
    price = 1
    cost = 0.5
    noise = 0.0005
    e_msr = calculate_e_msr(n_fishers, q, r, K, price, cost)
    e_nash = calculate_e_nash(e_msr, n_fishers)
    print("e_msr: {}".format(e_msr))
    print("e_nash: {}".format(e_nash))
    num_feedback = 50
    num_steps = 1000

    # Creating Simulation_arrays object
    my_sim = Simulation_lattice(network_dims, delta, q, r, K, R_0, e_0, price, cost,
                        noise, num_feedback, num_steps)
    my_sim.simulate()
    fig = plt.figure()
    plt.suptitle("delta = {}, num_feedback = {}".format(delta, num_feedback))

    # Plotting resource levels vs. time
    ax1 = fig.add_subplot(2,2,1)
    for i in range(my_sim.n_fishers):
        ax1.plot(np.arange(num_steps), my_sim.R_data[i])
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Resource (K = {})".format(my_sim.K))
    ax1.set_title("Territory Resource Levels vs. Time")
    ax1.grid(True)

    # Plotting avg payoff vs. time
    U_avg = np.average(my_sim.U_data, axis=0)
    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(np.arange(num_steps), U_avg)
    ax2.set_xlabel("Time steps")
    ax2.set_ylabel("Average utility")
    ax2.set_title("Average Utility vs. Time")
    ax2.grid(True)

    # Plotting avg effort vs. time
    e_avg = np.average(my_sim.e_data, axis=0)
    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(np.arange(num_steps), e_avg)
    ax3.set_xlabel("Time steps")
    ax3.set_ylabel("Effort")
    ax3.set_title("Average Effort vs. Time")
    ax3.grid(True)

    # Plotting all efforts vs. time
    ax4 = fig.add_subplot(2,2,4)
    for i in range(my_sim.n_fishers):
        ax4.plot(my_sim.e_data[i])
    ax4.set_xlabel("Time steps")
    ax4.set_ylabel("Effort")
    ax4.set_title("Effort vs. Time")
    ax4.grid(True)
    fig.subplots_adjust(wspace=0.3, hspace=0.4)

    # Calculate R_avg (only for seeing asymptote)
    R_avg = np.average(my_sim.R_data, axis=0)
    R_val = np.average(R_avg[int(R_avg.size/2):])

    # Print some useful information
    print("Last time step avg utility: {}".format(U_avg[-1]))
    print("Last time step avg effort: {}".format(e_avg[-1]))
    print("Last time step avg resource: {}".format(R_val))
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()

if __name__ == "__main__":
    main()