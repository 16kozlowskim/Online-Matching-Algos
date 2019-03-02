from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import math
import random as rand
from networkx.algorithms import bipartite
from itertools import groupby
from adversarial import perturbed_greedy

def generate_random_graph(streaming_set_size, static_set_size, seed):
    G = bipartite.random_graph(static_set_size, streaming_set_size, seed)
    streaming = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
    static = set(G) - streaming
    for node in G.nodes.data():
        if node[1]['bipartite'] == 1:
            node[1]['weight'] = rand.randint(1,100)

    return G, static, streaming


G, static, streaming = generate_random_graph(100,100,0.005)
print perturbed_greedy(G, static, streaming)
