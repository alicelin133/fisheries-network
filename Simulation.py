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
        t: integer representing the current time step t of the simulation
        delta: float representing extent to which territories are ecologically
        connected
        q: float representing "catchability" of fish resource
        price: price received per unit of fish
        cost: cost per unit of effort invested in fishing
    Node (territory) attributes:
        R: float representing territory's resource level
        e: float representing territory's effort level
        dR: float, for internal use in calculating leakage.
        pi: float representing territory's payoff from
        current time step
    """

    def __init__(self, n_fishers, delta, q, R_0, e_0, price, cost):
        """Return a Simulation object with a complete graph on *n_fishers* nodes,
        a leakage factor of *delta*, a fish catchability factor of *q*, and
        individual nodes with *R_0[i]* resource level and *e_0[i]* effort level."""
        self.G = nx.complete_graph(n_fishers)
        # TODO: may not be necessary to have the n_fishers data attribute
        self.n_fishers = n_fishers
        self.t = 0 # intialize time step t to 0
        self.delta = delta
        self.q = q
        self.price = price
        self.cost = cost
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
        pass

def main():
    """Performs unit testing."""
    R_0 = np.array([2.0,4.0,6.0])
    e_0 = np.array([1.0,2.0,3.0])
    my_sim = Simulation(3, 1, 1, R_0, e_0, 1, 0.1)
    print("Node attributes before harvest: {}".format(my_sim.G.nodes(data=True)))
    my_sim.harvest()
    for nood in my_sim.G.nodes(data=False):
        print("Payoff for fisher {} is {}".format(nood, self.G.node[nood]['pi']))
    print("Node attributes after harvest, before leakage: {}".format(my_sim.G.nodes(data=True)))
    my_sim.leakage()
    print("Node attributes after leakage: {}".format(my_sim.G.nodes(data=True)))

if __name__ == "__main__":
    main()
