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
        self.G = nx.complete_graph(n_fishers)
        # TODO: may not be necessary to have the n_fishers data attribute
        self.n_fishers = n_fishers
        self.t = 0 # intialize time step t to 0
        self.delta = delta
        self.q = q
        self.r = r
        self.K = K
        self.price = price
        self.cost = cost
        self.noise = noise
        # TODO: check length of R_0 and e_0 arrays, throw error if wrong
        for i in range(self.G.number_of_nodes()):
            self.G.node[i]['R'] = R_0[i]
            self.G.node[i]['e'] = e_0[i]
            self.G.node[i]['dR'] = 0.0
        
    def simulate(self, maxstep):
        """Runs a discrete simulation of the fisheries network for *maxstep*
        time steps."""
        for j in range(maxstep):
            harvest()
            regrowth()
            leakage()
            update_strategy()
            self.t += 1

    def harvest(self):
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
            else: # case 2: fisher wants less fish than he could take
                self.G.node[nood]['R'] = R - harvest
            self.G.node[nood]['pi'] = self.price * harvest - self.cost * e

    def regrowth(self):
        """Updates resource R for each territory based on the fish population's
        logistic growth, occurs after the time step's harvest."""
        # TODO: WRITE THIS
        # something like dR = r * R * (1 - R / K) and then R = R += dR
        for nood in self.G.nodes(data=False):
            R = self.G.node[nood]['R']
            self.G.node[nood]['R'] += self.r * R * (1 - R / self.K)
            
    def leakage(self):
        """Updates resource R for each territory based on resource leakage
        between territories. Each territory i gives (delta * R / self.G.degree(i))
        amount of resource to each of its adjacent territories."""
        for nood in self.G.nodes(data=False):
            R = self.G.node[nood]['R']
            for neighbor in self.G[nood]:
                self.G.node[neighbor]['dR'] += R * self.delta / (self.G.degree(nood) + 1)
        for nood in self.G.nodes(data=False):
            self.G.node[nood]['R'] = self.G.node[nood]['R'] * (1 - self.delta) + \
            self.delta * self.G.node[nood]['R'] / (self.G.degree(nood) + 1) + \
            self.G.node[nood]['dR']
    
    def update_strategy(self):
        """Selects two fishers randomly to compare payoff pi. The fisher with
        the lower payoff changes effort level to that of the higher-payoff
        fisher, but uniformly distributed around the effort level of the other
        fisher."""
        # TODO: update this crappy description >:(
        fisher1 = np.random.randint(0,n_fishers + 1) # pick 2 fishers
        fisher2 = np.random.randint(0,n_fishers + 1)
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
            self.G.node[fisher_lo]['e'] = e_new
            # TODO: should effort be within [0,1]? Seems to be just >0 atm
        else:
            pass

def main():
    """Performs unit testing."""
    R_0 = np.array([2.0,4.0,6.0])
    e_0 = np.array([0.1,0.1,0.1])
    my_sim = Simulation(3, 1, 1, 1, 10, R_0, e_0, 1, 0.1, 0.1)
    print("Node attributes before harvest: {}".format(my_sim.G.nodes(data=True)))
    my_sim.harvest()
    for nood in my_sim.G.nodes(data=False):
        print("Payoff for fisher {} is {}".format(nood, my_sim.G.node[nood]['pi']))
    print("Node attributes after harvest, before leakage: {}".format(my_sim.G.nodes(data=True)))
    my_sim.leakage()
    print("Node attributes after leakage: {}".format(my_sim.G.nodes(data=True)))
    # TODO: Run the simulation over time, my_sim.simulate(maxstep=100) or
    # something and figure out how to graph the results on matplotlib to see
    # what happens. It'll probably be absolutely awful but we'll see.

if __name__ == "__main__":
    main()
