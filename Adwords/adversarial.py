from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random as rand
from networkx.algorithms import bipartite, matching
from itertools import groupby

def MSVV(G, G_static, G_stream, budgets):

    arrival = list(G_stream)

    spent = {}
    for i in budgets:
        spent[i] = 0.0

    bids = {}
    for i in list(G.edges.data()):
        if i[0] not in bids:
            bids[i[0]] = {}
        bids[i[0]][i[1]] = i[2]['bid']

    for i in bids:
        print i
        bid = {}
        for j in bids[i]:
            min_bid = min([bids[i][j], (1 - spent[j])*budgets[j]])
            if min_bid == (1 - spent[j])*budgets[j]:
                min_bid = 0
            bid[j] = min_bid * (1-math.exp(spent[j]-1))

        j = max(bid, key=bid.get)
        min_bid = min([bids[i][j], (1 - spent[j])*budgets[j]])
        if min_bid == (1 - spent[j])*budgets[j]:
            min_bid = 0
        spent[j] = 1 - ((1-spent[j])*budgets[j] - min_bid)/budgets[j]

    profit = 0
    for i in spent:
        profit+=budgets[i]*spent[i]
    return profit
