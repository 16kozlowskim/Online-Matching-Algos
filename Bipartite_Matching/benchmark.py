from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import math
import random as rand
from networkx.algorithms import bipartite
from itertools import groupby
from knownIID import suggested_matching, two_suggested_matchings
from adversarial import greedy, random, ranking


def generate_random_graph(streaming_set_size, static_set_size, seed):
    G = bipartite.random_graph(static_set_size, streaming_set_size, seed)
    static = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
    streaming = set(G) - static

    return G, static, streaming

def max_matching(G, bipartition):
    max_matching = nx.bipartite.maximum_matching(G, bipartition)
    return max_matching


def uniform_dist(streaming_set, streaming_count, static_count):

    distribution = {}

    ratio = int(float(static_count)/streaming_count)

    for i in streaming_set:
        distribution[i] = ratio

    return distribution

def gaussian_dist(streaming_set, std_factor, static_count):
    n = len(streaming_set)
    d = norm(loc=n/2.0, scale = n*std_factor)

    distribution = {}
    top = n/2.0
    bot = n/2.0

    dist_sum = 0

    for i in streaming_set:
        distribution[i] = d.cdf(top+0.5)-d.cdf(top)+d.cdf(bot)-d.cdf(bot-0.5)
        dist_sum += distribution[i]
        top += 0.5
        bot -= 0.5

    scale = static_count/dist_sum

    for i in distribution:
        distribution[i] = int(math.ceil(distribution[i]*scale))

    return distribution


def benchmark_naive(G, static, streaming, matching_algo):
    avg_cr = 0

    for i in range(10):
        


def benchmark(G, static, streaming, std_factor, mode, matching_algo):
    avg_cr = 0
    distribution = {}

    if mode == 'gaussian':
        distribution = gaussian_dist(streaming, std_factor, len(static))
    elif mode == 'uniform':
        distribution = uniform_dist(streaming, len(streaming), len(static))


    for i in range(10):
        alg, opt = matching_algo(G, streaming, static, distribution)
        avg_cr = avg_cr + ((float(alg)/float(opt))*(1.0/10))
    return avg_cr


def run_tests(mode, matching_algo):
    edges = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    std_factor = [1, 0.5, 0.25, 0.1, 0.05]
    streaming_count = 100
    static_count = 1000

    for i in edges:
        edge_res = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        for j in range(10):
            graph_res = []
            G, static, streaming = generate_random_graph(streaming_count, static_count, i)
            for k in std_factor:
                graph_res.append(benchmark(G, static, streaming, k, mode, matching_algo))
            edge_res += np.array(graph_res)/10

        if mode == 'uniform':
            print np.mean(edge_res)
        elif mode == 'gaussian':
            print edge_res

def real_data_graph():
    data = pd.read_csv('../Data/KAG_conversion_data.csv')
    categories = data.groupby(['age', 'gender', 'interest'], as_index=False)['Impressions'].agg('sum')

    factor = 652
    impression_count = categories['Impressions'].values
    impression_count = impression_count//factor

    G = nx.Graph()

    ad_id = 0
    impression_id = 0

    for i in range(categories.shape[0]):
        ads = data.loc[(data['age'] == categories.loc[i,'age'])
        & (data['gender'] == categories.loc[i,'gender'])
        & (data['interest'] == categories.loc[i, 'interest'])]

        impressions = []
        for j in range(impression_count[i]):
            G.add_node(impression_id, bipartite=1)
            impressions.append(impression_id)
            impression_id += 1

        for j in range(ads.shape[0]):
            G.add_node(ad_id, bipartite=0)
            G.add_edges_from([(ad_id, k) for k in impressions])
            ad_id += 1

    static = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
    streaming = set(G) - static

    return G, static, streaming


def real_data_test(matching_algo):

    G, static, streaming = real_data_graph()

    avg_cr = 0
    distribution = uniform_dist(streaming, 1, 1)

    for i in range(5):
        print i
        alg, opt = matching_algo(G, streaming, static, distribution)
        avg_cr = avg_cr + ((float(alg)/float(opt))*(1.0/5))
    print avg_cr



real_data_test(two_suggested_matchings)
print '___________________'
real_data_test(suggested_matching)
print '___________________'
run_tests('gaussian', suggested_matching)
print '___________________'
run_tests('gaussian', two_suggested_matchings)
print '___________________'
run_tests('uniform', suggested_matching)
print '___________________'
run_tests('uniform', two_suggested_matchings)
print '___________________'
'''
[u1624396@stone-11 Bipartite_Matching]$ python benchmark3.py
0.6385
0.6323
0.6321
0.6333
0.6323
0.6321
0.6322
0.6327
___________________
0.7476
0.7264
0.7265
0.7243
0.7249
0.7245
0.7250
0.7251139999999999

benchmark.py
0.752708204094
0.624761836272  0.629674701698

[u1624396@stone-11 Bipartite_Matching]$ python benchmark2.py
[0.63575586 0.63906205 0.65145147 0.64200386 0.63929039]
[0.63249696 0.63192422 0.63048603 0.63896395 0.63645269]
[0.63189662 0.63284681 0.63028604 0.63268915 0.63634926]
[0.63109 0.63057 0.63195 0.63333 0.63301]
[0.63223 0.63095 0.63215 0.63254 0.63334]
[0.633   0.63204 0.63352 0.63127 0.63159]
[0.63262 0.63224 0.63229 0.63132 0.63282]
[0.63266 0.62967 0.63174 0.63139 0.63248]
___________________
[0.74509345 0.74713943 0.75856022 0.77483748 0.78989249]
[0.72866104 0.72704996 0.72887404 0.73757908 0.74861454]
[0.72857    0.72742    0.73048    0.72784248 0.73236233]
[0.72646 0.72486 0.72785 0.72873 0.72789]
[0.72839 0.72756 0.72583 0.72638 0.72965]
[0.72492 0.72444 0.72483 0.72543 0.72748]
[0.72461 0.72529 0.72539 0.72642 0.7258 ]
[0.72353 0.72456 0.72505 0.72507 0.72554]
'''
