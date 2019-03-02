from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random as rand
from networkx.algorithms import bipartite, matching
from itertools import groupby

def perturbed_greedy(G, G_static, G_stream):

    adj = {k: [v[1] for v in g] for k, g in groupby(sorted(G.edges), lambda e: e[0])}

    for edge in list(G.edges.data()):
        weight = G.nodes[edge[1]]['weight']
        edge[2]['weight'] = weight

    opt = matching.max_weight_matching(G, True)

    opt_weight = 0
    for edge in list(opt):
        node = max([edge[0], edge[1]])
        opt_weight += G.nodes[node]['weight']

    arrival = list(G_stream)
    rand.shuffle(arrival)
    matched = set()
    perturbation = {}

    for i in G_static:
        perturbation[i] = G.nodes[i]['weight']*(1-math.exp(rand.random()-1))

    for i in arrival:
        if i in adj:
            index_max = np.argmax(np.array([perturbation[j] for j in adj[i]]))
            if adj[i][index_max] not in matched:
                matched.add(adj[i][index_max])
                perturbation[adj[i][index_max]] = -1

    return sum([G.nodes[i]['weight'] for i in list(matched)])/float(opt_weight)
