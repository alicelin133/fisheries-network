import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

class Simulation(object):
    """A network of territories of fishers, where each territory harvests fish,
    and fish move between territories. The network has the following properties:

    Network attributes:
        G: networkx object, each node corresponds to a territory, each edge
        corresponds to a connection b/w territories that fish can leak through
        n_fishers: the number of territories in the network (1 fisher per
        territory)
        t: integer, the current time step *t* of the simulation
        delta: float, extent to which territories are ecologically
        connected
        q: float, "catchability" of fish resource
        r: float, growth factor of fish population 
        K: float, carrying capacity of fish population
        price: float, price received per unit of fish
        cost: float, cost per unit of effort invested in fishing
        noise: float, switching strategies occurs with +/- *noise*
    Node (territory) attributes:
        R: float, territory's resource level
        e: float, territory's effort level
        dR: float, for internal use in calculating leakage.
        pi: float, territory's payoff from current time step
    """

    def __init__(self, n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise):
        """Return a Simulation object with a complete graph on *n_fishers* nodes,
        a leakage factor of *delta*, a fish catchability factor of *q*, and
        individual nodes with *R_0[i]* resource level and *e_0[i]* effort level."""
        self.check_sim_attributes(n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise)
        self.G = nx.complete_graph(n_fishers)
        self.n_fishers = n_fishers
        self.t = 0 # intialize time step t to 0
        self.delta = delta
        self.q = q
        self.r = r
        self.K = K
        self.price = price
        self.cost = cost
        self.noise = noise
        for i in range(self.G.number_of_nodes()):
            self.G.node[i]['R'] = R_0[i]
            self.G.node[i]['e'] = e_0[i]
            self.G.node[i]['dR'] = 0.0
            self.G.node[i]['pi'] = 0.0

    def check_sim_attributes(self, n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise):
        """Checks that values of data attributes given in parameters to 
        constructor are in the correct format."""
        if delta < 0 or delta > 1:
            raise ValueError("delta must be in [0,1]")
        if q < 0 or q > 1:
            raise ValueError("q must be in [0,1]")
        if np.shape(R_0)[0] != n_fishers:
            raise ValueError("R_0 length must match n_fishers")
        if np.shape(e_0)[0] != n_fishers:
            raise ValueError("e_0 length must match n_fishers")
        for i in range(np.shape(e_0)[0]):
            if e_0[i] < 0 or e_0[i] > 1:
                raise ValueError("entry {} in e_0 must be in [0,1]".format(i))
        
    def simulate(self, maxstep):
        """Runs a discrete simulation of the fisheries network for *maxstep*
        time steps."""
        self.e_data = np.zeros((self.n_fishers, maxstep))
        self.R_data = np.zeros((self.n_fishers, maxstep))
        self.pi_data = np.zeros((self.n_fishers, maxstep))
        for t in range(maxstep):
            for nood in self.G.nodes(data=False):
                self.e_data[nood][t] = self.G.node[nood]['e']
                self.R_data[nood][t] = self.G.node[nood]['R']
                self.pi_data[nood][t] = self.G.node[nood]['pi']
            self.harvest()
            self.regrowth()
            # self.leakage()
            self.update_strategy()
            self.t += 1

    def harvest(self):
        """Updates resource R for each territory based on e, the territory's
        effort level, for a single time step. Also calculates payoff from
        harvest."""
        for nood in self.G.nodes(data=False):
            R = self.G.node[nood]['R']
            e = self.G.node[nood]['e']
            harvest = self.q * R * e
            # TODO: this set of conditionals isn't really necessary anymore
            # because q, e are required to be <= 1
            if harvest > R: # case 1: fisher wants more fish than he could take
                harvest = R
                self.G.node[nood]['R'] = 0
                print("Resource is {} for node {}".format(self.G.node[nood]['R'], nood))
            else: # case 2: fisher wants less fish than he could take
                self.G.node[nood]['R'] = R - harvest
            self.G.node[nood]['pi'] = self.price * harvest - self.cost * e

    def regrowth(self):
        """Updates resource R for each territory based on the fish population's
        logistic growth, occurs after the time step's harvest."""
        for nood in self.G.nodes(data=False):
            R = self.G.node[nood]['R']
            self.G.node[nood]['R'] += self.r * R * (1 - R/self.K)
            
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
            R_new = R * (1 - self.delta * deg / (deg + 1)) + dR
            self.G.node[nood]['R'] = R_new
    
    def update_strategy(self):
        """Selects two fishers randomly to compare payoff pi. The fisher with
        the lower payoff changes effort level to that of the higher-payoff
        fisher, but uniformly distributed around the effort level of the other
        fisher."""
        # TODO: update this crappy description >:(
        fisher1 = np.random.randint(0,self.n_fishers) # pick 2 fishers
        fisher2 = np.random.randint(0,self.n_fishers)
        if self.G.node[fisher1]['pi'] < self.G.node[fisher2]['pi']:
            fisher_lo = fisher1
            fisher_hi = fisher2
            pi_lo = self.G.node[fisher1]['pi']
            pi_hi = self.G.node[fisher2]['pi']
        else: # note that this case occurs when they have equal payoffs
            fisher_lo = fisher2
            fisher_hi = fisher1
            pi_lo = self.G.node[fisher2]['pi']
            pi_hi = self.G.node[fisher1]['pi']
        # Probability that lo-payoff fisher switches to hi-p fisher's strategy
        switch_prob = (pi_hi - pi_lo) / (abs(pi_hi) + abs(pi_lo))
        prob = np.random.random()
        if prob < switch_prob:
            diff = np.random.uniform(-1 * self.noise, self.noise)
            e_new = self.G.node[fisher_hi]['e'] + diff
            if e_new < 0:
                e_new = 0
            elif e_new > 1:
                e_new = 1
            self.G.node[fisher_lo]['e'] = e_new
        else:
            pass

def main():
    """Performs unit testing."""
    # Parameters: n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise
    n_fishers = 3
    delta = 1
    q = 0.5
    r = 0.05
    K = 200
    # R_0 = np.full(n_fishers,K/2)
    R_0 = np.array([50, 100, 150])
    e_0 = np.linspace(0.01,0.05,num=n_fishers)
    price = 1
    cost = 0.1
    noise = 0.0005
    num_steps = 1000
    my_sim2 = Simulation(n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise)
    my_sim2.leakage()
    # my_sim2.simulate(num_steps)
    # e_avg = np.average(my_sim2.e_data, axis=0)
    # for i in range(my_sim2.n_fishers):
    #     plt.plot(np.arange(num_steps), my_sim2.R_data[i])
    # print(my_sim2.R_data)
    # plt.show()

if __name__ == "__main__":
    main()
