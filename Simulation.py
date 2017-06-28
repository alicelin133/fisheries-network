class Simulation(object):
""" A network of territories of fishers, where each territory harvests fish,
and fish move between territories. Nodes represent territories, edges
represent connections that fish can move through.
The network has the following properties:
Network attributes:
    t: integer representing the current time step t of the simulation
    delta: float representing extent to which territories are ecologically
    connected
    q: float representing "catchability" of fish resource
Node (territory) attributes:
    R: float representing territory's resource level
    e: float representing territory's effort level
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

def __init__(self, n_fishers, delta, q, R_0, e_0):
    """Return a Simulation object with a complete graph on *n_fishers* nodes,
    a leakage factor of *delta*, a fish catchability factor of *q*, and
    individual nodes with *R_0[i]* resource level and *e_0[i]* effort level."""
    self.G = nx.complete_graph(n_fishers)
    # TODO: use assert to check length of R_0 and e_0 arrays
    self.G.graph['t'] = 0 # intialize time step t to 0
    self.G.graph['delta'] = delta
    self.G.graph['q'] = q
    for i in range(self.G.number_of_nodes()):
        self.G.node[i]['R'] = R_0[i]
        self.G.node[i]['e'] = e_0[i]
    
def simulate(self, maxstep):
    """Runs a discrete simulation of the fisheries network for *maxstep*
    time steps."""
    for j in range(maxstep):
        harvest()
        leakage()
        update_strategy()
        self.G.graph['t'] = j + 1

def harvest(self):
    """Updates resource R for each territory based on e, the territory's
    effort level, for a single time step."""
    pass

def leakage(self):
    """Updates resource R for each territory based on resource leakage
    between territories. Each territory i gives (delta * R / self.G.degree(i))
    amount of resource to each of its adjacent territories."""
    # TODO: figure out whether you want to use n or n-1 in the denominator
    # of what goes to the neighboring territories. probably n-1 because it
    # is easier to do that in code because it just uses the degree method
    pass