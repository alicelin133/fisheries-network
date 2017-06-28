class Simulation():
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
import min_model_constants as const

def __init__(self):
    self.G = nx.complete_graph(const.n)