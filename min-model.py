'''
min-model.py
Models a situation of n spatial units, each occupied by one fisher.
'''
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
import min-model-constants as const

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
    q = G.graph['q']
    for i in range(number_of_nodes(G)): # TODO: check if this function is correct
        R = G.node[i]['R']
        e = G.node[i]['e']
        R -= q * R * e
        G.node[i]['R'] = R
    return G

def leakage(G):
    delta = G.graph['delta']
    n = nx.number_of_nodes(G)
    sum = 0.0
    for i in range(n):
        sum += G.node[i]['R']
    for i in range(n):
        R = G.node[i]['R']
        R = (1 - delta) * R + sum / n
    return G

def update_strategy(G):
    return G

if __name__ == "__main__":
    G = nx.complete_graph(const.n) # initializes network of fisheries
    G.graph['t'] = 0 # initializes time to 0
    G.graph['delta'] = const.delta
    G.graph['q'] = const.q
    for i in range(const.n):
        G.node[i]['R'] = const.R_0
        G.node[i]['e'] = const.e_0
    # print(G.nodes(data=True)) # checks node attributes
    # simulate(G, maxstep) # start simulation
    print(leakage(G).nodes(data=True))


