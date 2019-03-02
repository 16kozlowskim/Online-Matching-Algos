# Online Bipartite Matching Algos: Greedy, Random, Ranking, SM, TSM, TSM (non-adaptive)
# Online Vertex-Weighted Bipartite Matching: Perturbed Greedy
# Adwords: Greedy, Balance, MSVV
# The online Primal-Dual: ...
# Display Ads: ...
from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random as rand
from networkx.algorithms import bipartite
from itertools import groupby


def generate_random_graph(static_set_size, streaming_set_size, seed):
    G = bipartite.random_graph(streaming_set_size, static_set_size, seed)
    G_part_1 = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
    G_part_2 = set(G) - G_part_1

    return G, G_part_1, G_part_2

def max_matching(G, bipartition):
    max_matching = nx.bipartite.maximum_matching(G, bipartition)
    return max_matching

def greedy(G, G_stream, G_static):
    adj = {k: [v[1] for v in g] for k, g in groupby(sorted(G.edges), lambda e: e[0])}
    B = nx.Graph()
    B.add_nodes_from(G_static)

    arrival = list(G_stream)
    rand.shuffle(arrival)
    matched = set()

    for i in arrival:
        if i in adj:
            for j in adj[i]:
                if j not in matched:
                    matched.add(j)
                    B.add_node(j)
                    B.add_edge(i, j)
                    break

    return len(matched)

def random(G, G_stream, G_static):
    adj = {k: [v[1] for v in g] for k, g in groupby(sorted(G.edges), lambda e: e[0])}
    B = nx.Graph()
    B.add_nodes_from(G_static)

    arrival = list(G_stream)
    rand.shuffle(arrival)
    matched = set()

    for i in arrival:
        if i in adj:
            rand.shuffle(adj[i])
            for j in adj[i]:
                if j not in matched:
                    matched.add(j)
                    B.add_node(j)
                    B.add_edge(i, j)
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

def suggested_matching(G, G_stream, G_static, distribution):

    B = nx.DiGraph()
    B.add_nodes_from(G_static)
    B.add_nodes_from(G_stream)
    B.add_edges_from(G.edges, capacity = 1)
    B.add_nodes_from(['s','t'])
    for i in G_static:
        B.add_edge('s',i,capacity=1)
    for i in G_stream:
        B.add_edge(i,'t',capacity=distribution[i])

    flow_val, flow_dict = nx.maximum_flow(B, 's', 't')

    sum = 0
    for i in distribution:
        sum+=distribution[i]

    dist = {}
    for i in distribution:
        dist[i] = int(math.ceil(distribution[i] * (rand.random()*10)))

    stream = []
    for i in dist:
        for j in range(dist[i]):
            stream.append(i)

    arrival = []
    for i in range(sum):
        index = rand.randint(0, len(stream)-1)
        arrival.append(stream[index])

    realization = nx.Graph()
    realization.add_nodes_from(G_static)

    for i in range(len(arrival)):
        realization.add_node(i+len(G_static))

        for j in G.adj[arrival[i]].keys():
            realization.add_edge(i+len(G_static), j)

    opt = len(nx.bipartite.maximum_matching(realization, G_static))/2

    matched = set()

    for i in arrival:
        prob = rand.random()
        for j in range(len(G_static)):
            if i in flow_dict[j] and flow_dict[j][i] == 1:
                prob = prob - (1.0/distribution[i])
                if prob < 0:
                    if j not in matched:
                        matched.add(j)
                    break
    return len(matched), opt

def two_suggested_matchings(G, G_stream, G_static, distribution):

    B = nx.DiGraph()
    B.add_nodes_from(G_static)

    adj = {}
    streaming = []
    for i in list(G_stream):
        for j in range(distribution[i]):
            streaming.append(i)

    for i in range(len(streaming)):

        B.add_node(i+len(G_static))

        for j in G.adj[streaming[i]].keys():
            B.add_edge(j, i+len(G_static), capacity=1)
            if i+len(G_static) not in adj:
                adj[i+len(G_static)] = []
            adj[i+len(G_static)].append(j)

    B.add_nodes_from(['s','t'])
    for i in G_static:
        B.add_edge('s',i,capacity=2)
    for i in range(len(streaming)):
        B.add_edge(i+len(G_static),'t',capacity=2)

    flow_val, flow_dict = nx.maximum_flow(B, 's', 't')

    flow_edges = {}
    for node in G_static:
        for i in flow_dict[node]:
            if flow_dict[node][i] == 1:
                if node not in flow_edges:
                    flow_edges[node] = []
                flow_edges[node].append(i)
                if i not in flow_edges:
                    flow_edges[i] = []
                flow_edges[i].append(node)


    paths = []


    for node in B.nodes:
        if node in flow_edges and len(flow_edges[node])==1:
            prev = node
            current = flow_edges[node][0]
            arr = [prev, current]
            while True:
                ret = pathfinder(flow_edges, prev, current)
                if ret == -1:
                    break
                prev = current
                current = ret
                arr.append(ret)
            paths.append(arr)
            for i in paths[-1]:
                del flow_edges[i]


    cycles = []

    for node in B.nodes:
        if node in flow_edges:
            prev = node
            beg = node
            current = flow_edges[node][0]
            arr = [prev, current]
            while True:
                ret = cyclefinder(flow_edges, prev, current, beg)
                prev = current
                current = ret
                arr.append(ret)
                if ret == beg:
                    break
            cycles.append(arr)
            for i in cycles[-1][:-1]:
                del flow_edges[i]

    colors = {}

    static_size = len(G_static)

    for i in range(len(cycles)):
        for j in range(len(cycles[i])):
            if cycles[i][j] < static_size:
                continue
            colors[cycles[i][j]] = [cycles[i][j-1], cycles[i][j+1]]


    for path in paths:
        if path[0] < static_size and path[-1] >= static_size:
            for i in range(len(path)):
                if path[i] < static_size:
                    continue
                try:
                    colors[path[i]] = [path[i-1], path[i+1]]
                except IndexError:
                    colors[path[i]] = [path[i-1]]
        elif path[0] < static_size and path[-1] < static_size:
            for i in range(len(path)):
                if path[i] < static_size:
                    continue
                colors[path[i]] = [path[i-1], path[i+1]]
        elif path[0] >= static_size and path[-1] >= static_size:
            for i in range(len(path)):
                if path[i] < static_size:
                    continue
                elif i == 0:
                    colors[path[i]] = [path[i+1]]
                else:
                    try:
                        colors[path[i]] = [path[i-1], path[i+1]]
                    except IndexError:
                        colors[path[i]] = [path[i-1]]


    streaming =  list(B.nodes - G_static - set(['s', 't']))

    dist = {}

    for i in streaming:
        dist[i] = int(math.ceil(rand.random()*10))

    streaming_2 = []
    for i in dist:
        for j in range(dist[i]):
            streaming_2.append(i)

    arrival = []
    for i in range(len(streaming)):
        index = rand.randint(0, len(streaming_2)-1)
        arrival.append(streaming_2[index])


    realization = nx.Graph()
    realization.add_nodes_from(G_static)


    for i in range(len(arrival)):
        if arrival[i] not in adj:
            continue
        realization.add_node(i+len(G_static))
        for j in adj[arrival[i]]:
            realization.add_edge(i+len(G_static), j)

    opt = len(nx.bipartite.maximum_matching(realization, G_static))/2

    matched = set()

    count = {}

    for i in streaming:
        count[i] = 0

    for i in arrival:
        if i in colors:
            try:
                if count[i] == 0 and colors[i][0] not in matched:
                    matched.add(colors[i][0])
                    count[i] = 1
                elif count[i] == 1 and colors[i][1] not in matched:
                    matched.add(colors[i][1])
                    count[i] = 2
            except IndexError:
                continue

    return len(matched), opt



def cyclefinder(flow_edges, prev, current, beg):
    for i in flow_edges[current]:
        if i == beg and prev != beg:
            return beg
        if i != prev:
            return i

def pathfinder(flow_edges, prev, current):
    for i in flow_edges[current]:
        if i != prev:
            return i
    return -1


def generate_graph(streaming_count, static_count, edges):

    G, G_part_1, G_part_2 = generate_random_graph(streaming_count, static_count, edges)

    return G, G_part_1, G_part_2

def uniform_dist(streaming_set):

    distribution = {}

    for i in streaming_set:
        distribution[i] = 1

    return distribution

def gaussian_dist(streaming_set, std):
    d = norm(loc=0, scale = std)
    n = len(streaming_set)
    scale = float(round((d.cdf(n)-d.cdf(n-1))*100000))

    distribution = {}

    for index, i in enumerate(streaming_set):
        distribution[i] = int(round((d.cdf(index+1) - d.cdf(index))*(100000)/scale))

    return distribution

def benchmark_SM(std, G, G_part_1, G_part_2, scale):
    avg_cr = 0
    distribution = gaussian_dist(G_part_2, std)
    #distribution = uniform_dist(G_part_2)

    for i in distribution:
        distribution[i] = distribution[i]*scale
    for i in range(10):
        alg, opt = suggested_matching(G, G_part_2, G_part_1, distribution)
        avg_cr = avg_cr + ((float(alg)/float(opt))*(1.0/10))
    return avg_cr

def benchmark_TSM(std, G, G_part_1, G_part_2, scale):
    avg_cr = 0
    distribution = gaussian_dist(G_part_2, std)
    #distribution = uniform_dist(G_part_2)

    for i in distribution:
        distribution[i] = distribution[i]*scale
    for i in range(10):
        alg, opt = two_suggested_matchings(G, G_part_2, G_part_1, distribution)
        avg_cr = avg_cr + ((float(alg)/float(opt))*(1.0/10))
    return avg_cr

def static_count(streaming_count, std):
    G, G_part_1, G_part_2 = generate_graph(streaming_count, 100, 1)
    distribution = gaussian_dist(G_part_2, std)
    sum = 0
    for i in distribution.values():
        sum = sum + i
    return sum



def run_tests2():
    edges = [0.0001, 0.001, 0.005, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    streaming_count = 1000
    static_count = 1000
    for i in edges:
        print i
        edge_res = []
        for j in range(5):
            G, G_part_1, G_part_2 = generate_graph(streaming_count, static_count, i)
            edge_res.append(benchmark_TSM(0, G, G_part_1, G_part_2, 1))
        avg = 0
        for i in edge_res:
            avg += i/len(edge_res)
        print avg

def run_tests():
    edges = [0.0001, 0.001, 0.005, 0.01, 0.1, 0.3, 0.5]
    std = [300,260,220,180,140]
    streaming_count = 300
    static_count = 1000
    scale = [4, 3, 3, 2, 1]
    for i in edges:
        edge_res = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        for j in range(5):
            graph_res = []
            G, G_part_1, G_part_2 = generate_graph(streaming_count, static_count, i)
            print 'Graph '+str(j)+' with edges: '+str(i)
            for index, k in enumerate(std):
                #graph_res.append(benchmark_SM(k, G, G_part_1, G_part_2, scale[index]))
                graph_res.append(benchmark_TSM(k, G, G_part_1, G_part_2, scale[index]))
            edge_res += np.array(graph_res)/5
        print edge_res
run_tests()
    #print 'Cardinality of maximum matching: ' + str(len(max_matching)/2)
    #print 'Size of matching found by greedy: ' + str(greedy(G, G_part_1, G_part_2))
    #print 'Size of matching found by random: ' + str(random(G, G_part_1, G_part_2))
    #print 'Size of matching found by ranking: ' + str(ranking(G, G_part_1, G_part_2))
    #print 'Size of matching found by SM: ' + str(suggested_matching(G, G_part_2, G_part_1, distribution))
    #print 'Size of matching found by TSM: ' + str(two_suggested_matchings(G, G_part_2, G_part_1, distribution))
