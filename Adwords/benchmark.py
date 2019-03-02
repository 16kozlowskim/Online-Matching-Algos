from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import math
import random as rand
from networkx.algorithms import bipartite
from itertools import groupby
import cvxpy
from adversarial import MSVV


def generate_random_graph(static_set_size, streaming_set_size, seed):
    G = bipartite.random_graph(streaming_set_size, static_set_size, seed)
    streaming = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
    static = set(G) - streaming

    constraint_vector = []
    constraint_matrix = [[] for i in range(static_set_size)]
    budgets = {}

    for node in G.nodes.data():
        if node[1]['bipartite'] == 1:
            node[1]['budget'] = rand.random()*100
            budgets[node[0]] = node[1]['budget']
            constraint_vector.append(node[1]['budget'])

    for i, edge in enumerate(list(G.edges.data())):
        edge[2]['bid'] = rand.random()*20
        constraint_matrix[i % static_set_size].append(edge[2]['bid'])

    for i, row in enumerate(constraint_matrix):
        prefix_len = i * streaming_set_size
        postfix_len = static_set_size*streaming_set_size - streaming_set_size - prefix_len
        constraint_matrix[i] = [0]*prefix_len + row + [0]*postfix_len

    bids = constraint_matrix[:]

    for i in range(streaming_set_size):
        constraint_vector.append(1)
        constraint_matrix.append([0]*(static_set_size*streaming_set_size))
        for j in range(static_set_size):
            constraint_matrix[static_set_size + i][(j*streaming_set_size) + i] = 1

    var_vector_len = static_set_size*streaming_set_size

    return G, static, streaming, constraint_vector, constraint_matrix, var_vector_len, bids, static_set_size, budgets

def IP_solver(constraint_vector, constraint_matrix, var_vector_len, bids, G, one_vector_len):
    vars = cvxpy.Bool(var_vector_len)
    constraints = constraint_matrix * vars <= constraint_vector
    one_vector = np.array([1]*one_vector_len)

    obj = one_vector * (bids * vars)

    adwords = cvxpy.Problem(cvxpy.Maximize(obj), [constraints])

    adwords.solve(solver=cvxpy.GLPK_MI)

    opt_var =  np.squeeze(np.asarray(vars.value))
    opt = np.sum(bids.dot(opt_var))
    return opt

G, static, stream, constraint_vector, constraint_matrix, var_vector_len, bids, one_vector_len, budgets = generate_random_graph(4,10,1)
opt = IP_solver(np.array(constraint_vector), np.array(constraint_matrix), var_vector_len, np.array(bids), G, one_vector_len)
print opt
alg = MSVV(G, static, stream, budgets)
print alg

print alg/opt
