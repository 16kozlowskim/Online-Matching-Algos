from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random as rand
from networkx.algorithms import bipartite
from itertools import groupby

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

    stream = []
    for i in distribution:
        for j in range(distribution[i]):
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

    arrival = []
    for i in range(len(streaming)):
        index = rand.randint(0, len(streaming)-1)
        arrival.append(streaming[index])


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
