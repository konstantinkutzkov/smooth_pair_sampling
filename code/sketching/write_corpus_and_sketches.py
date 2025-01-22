import numpy as np
import math
import sys
import os
import inspect
import argparse
from datasketch import HyperLogLog as HLL

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(parentdir)
sys.path.insert(0, parentdir)
import networkx as nx
#from clean_smooth_node_embeddings import graph_utilities
from graph_utilities import random_walk_nx, edges_to_nx_undirected_graph


def write_pairs(G, nr_walks, walk_length, window, path, prefix):
    f = open(path + prefix + 'pairs.txt', 'w')
    for walk_nr in range(nr_walks):
        if walk_nr % 50 == 0:
            print('walk', walk_nr)
        for i, u in enumerate(sorted(G.nodes())):
            seed = walk_nr*G.number_of_nodes() + i
            if nx.is_directed(G) and G.out_degree[u] == 0:
                continue
            walk_u = random_walk_nx(G, u, walk_length, seed=seed) #G.random_walk(u, walk_length, seed=seed)
            walk_pairs = skip_gram_pairs(walk_u, window)
            for pair in walk_pairs:
                f.write(str(pair[0]) + '\t' + str(pair[1]) + '\n')
    f.close()

def write_corpus(G, path, nr_walks, walk_length, prefix):
    f = open(path + prefix + f'corpus_{nr_walks}_{walk_length}_std.txt', 'w')
    for walk_nr in range(nr_walks):
        if walk_nr % 1 == 0:
            print('walk', walk_nr)
        for i, u in enumerate(sorted(G.nodes())):
            seed = walk_nr*G.number_of_nodes() + i
            if nx.is_directed(G) and G.out_degree[u] == 0:
                continue
            walk_u = random_walk_nx(G, u, walk_length, seed=seed)
            for node in walk_u:
                f.write(str(node) + ' ')
            f.write('\n')
    f.close()

def exact_count(G, nr_walks, walk_length, window):
    pairs = set()
    for walk_nr in range(nr_walks):
        if walk_nr % 1 == 0:
            print('walk', walk_nr)
        for i, u in enumerate(sorted(G.nodes())):
            seed = walk_nr*G.number_of_nodes() + i
            if nx.is_directed(G) and G.out_degree[u] == 0:
                continue
            walk_u = random_walk_nx(G, u, walk_length, seed=seed) #G.random_walk(u, walk_length, seed=seed)
            walk_pairs = skip_gram_pairs(walk_u, window)
            for pair in walk_pairs:
                pairs.add(pair)
    return len(pairs)

def estimate_nr_pairs_from_file(corpusfile, window):
    hll = HLL(16)
    f = open(corpusfile, 'r')
    for line in f:
        line_split = line.split()
        walk = [int(n) for n in line_split if len(n)>0]
        walk_pairs = skip_gram_pairs(walk, window)
        for pair in walk_pairs:
            hll.update(str(pair).encode('utf8'))
    f.close()
    return hll.count()

def estimate_nr_pairs(G, nr_walks, walk_length, window):
    hll = HLL(16)
    for walk_nr in range(nr_walks):
        if walk_nr % 1 == 0:
            print('walk', walk_nr)
        for i, u in enumerate(sorted(G.nodes())):
            seed = walk_nr*G.number_of_nodes() + i
            if nx.is_directed(G) and G.out_degree[u] == 0:
                continue
            walk_u = random_walk_nx(G, u, walk_length, seed=seed) #G.random_walk(u, walk_length, seed=seed)
            walk_pairs = skip_gram_pairs(walk_u, window)
            for pair in walk_pairs:
                hll.update(str(pair).encode('utf8'))
    return hll.count()

def skip_gram_pairs(rw_path, L):
    pairs = []
    for i in range(len(rw_path)):
        for j in range(1, L+1):
            if i+j < len(rw_path):
                pairs.append((rw_path[i], rw_path[i+j]))
            if i >= j:
                pairs.append((rw_path[i], rw_path[i-j]))    
    return pairs  

def sketch_corpus(G, nr_walks, walk_length, window, nr_pairs, budget, path, prefix=''):
    """ Returns a sketch of the pairs in the graph walks.
            G: the input graph
            nr_walks: how many walks per node
            walk_length: wength of the random walk.
            window: the window size for pair generation from the walk
            budget: the sketch size
    """
    sketch = {}
    cnt = 0 # the total number of pairs
    
    capacity = (nr_pairs*budget)//100
    for walk_nr in range(nr_walks):
        if walk_nr % 1 == 0:
            print('walk', walk_nr)
        for i, u in enumerate(sorted(G.nodes())):
            seed = walk_nr*G.number_of_nodes() + i
            if nx.is_directed(G) and G.out_degree[u] == 0:
                continue
            walk_u = random_walk_nx(G, u, walk_length, seed=seed) #G.random_walk(u, walk_length, seed=seed)
            walk_pairs = skip_gram_pairs(walk_u, window)
            for pair in walk_pairs:
                cnt += 1
                if pair in sketch:
                    sketch[pair] += 1
                elif len(sketch) < capacity:
                    sketch[pair] = 1
                else:
                    for p in list(sketch.keys()):
                        sketch[p] -= 1
                        if sketch[p] == 0:
                            del sketch[p]
    exact_cnts = {}
    for walk_nr in range(nr_walks):
        if walk_nr % 1 == 0:
            print('walk', walk_nr)
        for i, u in enumerate(sorted(G.nodes())):
            seed = walk_nr*G.number_of_nodes() + i
            if nx.is_directed(G) and G.out_degree[u] == 0:
                continue
            walk_u = random_walk_nx(G, u, walk_length, seed=seed) 
            walk_pairs = skip_gram_pairs(walk_u, window)
            for pair in walk_pairs:
                if pair in sketch:
                    exact_cnts.setdefault(pair, 0)
                    exact_cnts[pair] += 1

    name = '{}budget_{}_walks_{}_wl_{}_std.txt'.format(prefix, budget, nr_walks, walk_length)
    f = open(path + name, 'w')

    print('Path', path+name)

    print(cnt, sum(sketch.values()))
    f.write(str((cnt - sum(sketch.values()) )/capacity) + '\n')
    for pair, freq in exact_cnts.items():
        f.write(str(pair[0]) + ',' + str(pair[1]) + ':' + str(freq) + '\n')
    f.close()

if __name__ == "__main__":

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    root_dir = os.path.dirname(parentdir)
    sys.path.insert(0, parentdir)
   
    parser = argparse.ArgumentParser(description="Corpus generation and sketching.")
    parser.add_argument('--graph', nargs='?', default='Citeseer', help='Input graph name')
    parser.add_argument('--b', type=int, default='10', help='Sketch budget as percentage')
    parser.add_argument('--prefix', nargs='?', default='', help='Set to reduced_ for learning embeddings on the reduced graph for link prediction')

    args = parser.parse_args()

    graph = args.graph
    prefix = args.prefix
    budget = args.b
    G = edges_to_nx_undirected_graph(root_dir + f'/data/{graph}/{prefix}edges_std.txt')
    print('# edges', G.number_of_edges())
    nr_walks = 10
    walk_length = 80
    window = 10
    
    print('Estimate number of pairs')
    nr_pairs_est = estimate_nr_pairs(G, nr_walks=nr_walks, walk_length=walk_length, window=window)
    print("n to power", math.log(nr_pairs_est, G.number_of_nodes()))
    print('Write corpus')
    write_corpus(G, root_dir + f'/data/{graph}/', nr_walks=nr_walks, walk_length=walk_length, prefix=prefix)
    
    print('Sketch coprus for budget {}%'.format(budget))
    sketch_corpus(G, nr_walks=nr_walks, walk_length=walk_length, window=window,\
                   nr_pairs=nr_pairs_est, budget=budget, path=root_dir + '/data/' + graph + '/sketches/', prefix=prefix)
