'''
min-model.py
Models a situation of n spatial units, each occupied by one fisher.
'''
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

def simulate(G, maxstep):
    for i in range(maxstep):
        G = one_step(G)
        G.graph['t'] += 1

def one_step(G):
    G = harvest(G)
    G = leakage(G)
    G = update_strategy(G)
    return G

def harvest(G):
    
    return G

def leakage(G):
    return G

def update_strategy(G):
    return G

if __name__ == "__main__":
    n = 10 # number of spatial units (territories), 1 fisher in each
    G = nx.complete_graph(n) # initializes network of fisheries
    G.graph['t'] = 0 # initializes time to 0
    delta = 0 # factor of leakage
    G.graph['delta'] = delta
    R_0 = 50 # initial biomass of fish per territory
    e_0 = 0.005 # initial effort level from one fisher
    for i in range(n):
        G.node[i]['R'] = R_0
        G.node[i]['e'] = e_0
    # print(G.nodes(data=True)) # checks node attributes
    maxstep = 10000
    simulate(G, maxstep) # start simulation


