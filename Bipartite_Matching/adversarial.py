from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random as rand
from networkx.algorithms import bipartite
from itertools import groupby


def greedy(G, G_stream, G_static):
    adj = {k: [v[1] for v in g] for k, g in groupby(sorted(G.edges), lambda e: e[0])}

    arrival = list(G_stream)
    rand.shuffle(arrival)
    matched = set()

    for i in arrival:
        if i in adj:
            for j in adj[i]:
                if j not in matched:
                    matched.add(j)
                    break

    return len(matched)

def random(G, G_stream, G_static):
    adj = {k: [v[1] for v in g] for k, g in groupby(sorted(G.edges), lambda e: e[0])}

    arrival = list(G_stream)
    rand.shuffle(arrival)
    matched = set()

    for i in arrival:
        if i in adj:
            rand.shuffle(adj[i])
            for j in adj[i]:
                if j not in matched:
                    matched.add(j)
                    break

    return len(matched)

def ranking(G, G_stream, G_static):
    adj = {k: [v[1] for v in g] for k, g in groupby(sorted(G.edges), lambda e: e[0])}
    B = nx.Graph()
    B.add_nodes_from(G_static)

    arrival = list(G_stream)
    rand.shuffle(arrival)

    static = list(G_static)
    rand.shuffle(static)
    ranked_static = {}
    for i in range(len(static)):
        ranked_static[static[i]] = i

    for i in arrival:
        if i in adj:
            rank = max([ranked_static[j] for j in adj[i]])
            if rank == -1:
                continue
            node = static[rank]
            ranked_static[node] = -1
            B.add_node(node)
            B.add_edge(i, node)

    return len(B.edges)
