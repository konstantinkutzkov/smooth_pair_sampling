from collections import defaultdict
from collections.abc import Iterable
import logging
import pandas as pd
import networkx as nx
import os
import time
import random
from typing import List
import json
import shutil

logger = logging.getLogger("smooth_deepwalk")


def skip_gram_pairs(rw_path, L):
    pairs = []
    for i in range(len(rw_path)):
        for j in range(1, L+1):
            if i+j < len(rw_path):
                pairs.append(( int(rw_path[i]), int(rw_path[i+j]) ))
            if i >= j:
                pairs.append(( int(rw_path[i]), int(rw_path[i-j]) ))    
    return pairs  

def random_walk_nx(G, start_node, walk_length, seed):
    walk = [start_node]
    random.seed(seed)
    while len(walk) < walk_length:
        cur = walk[-1]
        if len(G[cur]) > 0:
            nbrs = sorted(list(G.neighbors(cur)))
            idx = random.randint(0, len(nbrs)-1)
            walk.append(nbrs[idx]) #rand.choice(G[cur]))
        else: # restart as we reached a leaf
            walk.append(walk[0])
    return [str(node) for node in walk]


def pandas_graph_to_edgelist(datadir, graphname, fname):
    edges_df = pd.read_csv(os.path.join(datadir, graphname.lower() + ".cites"), \
                            sep='\t', header=None, names=["target", "source"])
    print(len(set(edges_df['target']).union(set(edges_df['source']))))
    f = open(fname, 'w')
    for x, y in zip(edges_df['target'], edges_df['source']):
        f.write(str(x)+' '+str(y)+'\n')
    f.close()


def pandas_graph_to_nx(datadir, graphname):
    G = nx.Graph()
    node2int = {}
    edges_df = pd.read_csv(os.path.join(datadir, graphname.lower() + ".cites"), \
                            sep='\t', header=None, names=["target", "source"])
    # print(len(set(edges_df['target']).union(set(edges_df['source']))))
    for x, y in zip(edges_df['target'], edges_df['source']):
        node2int.setdefault(x, len(node2int))
        node2int.setdefault(y, len(node2int))
        G.add_edge(node2int[x], node2int[y])
    return G, node2int


def write_edges_cora(datadir, graphname):
    edges_df = pd.read_csv(os.path.join(datadir, graphname.lower() + ".cites"), \
                            sep='\t', header=None, names=["target", "source"])
    f = open(datadir + 'edgelist.txt', 'w')
    for x, y in zip(edges_df['target'], edges_df['source']):
        x = "".join(str(x).split())
        y = "".join(str(y).split())
        f.write(str(x) + '\t' + str(y) + '\n')
    f.close()

def write_edges_pubmed(datadir, graphname):
    f = open(datadir + 'edgelist.txt', 'w')
    with open(os.path.join(datadir, graphname.lower() + ".cites"), 'r') as edgefile:
        for line in edgefile:
            # print(line)
            line_split = line.split('|')
            if len(line_split) > 1:
                l0 = line_split[0]
                l1 = line_split[1]
                x = l0.split(':')[1]
                y = l1.split(':')[1]
                x = "".join(x.split())
                y = "".join(y.split())
                f.write(str(x) + '\t' + str(y) + '\n')
    f.close()

def write_edges_gitweb(datadir):
    f = open(datadir + 'edgelist.txt', 'w')
    edges_df = pd.read_csv(os.path.join(datadir,  "edges.csv"))
    for x, y in zip(edges_df['id_1'], edges_df['id_2']):
        x = "".join(str(x).split())
        y = "".join(str(y).split())
        f.write(str(x) + '\t' + str(y) + '\n')
    f.close()


def write_edges_deezer(datadir):
    f = open(datadir + 'edgelist.txt', 'w')
    edges_df = pd.read_csv(os.path.join(datadir,  "edges.csv"))
    for x, y in zip(edges_df['node_1'], edges_df['node_2']):
        x = "".join(str(x).split())
        y = "".join(str(y).split())
        f.write(str(x) + '\t' + str(y) + '\n')
    f.close()

def write_edges_lastfm(datadir):
    f = open(datadir + 'edgelist.txt', 'w')
    edges_df = pd.read_csv(os.path.join(datadir,  "edges.csv"))
    for x, y in zip(edges_df['node_1'], edges_df['node_2']):
        x = "".join(str(x).split())
        y = "".join(str(y).split())
        f.write(str(x) + '\t' + str(y) + '\n')
    f.close()

def standardize_graph(edgefile, nodelabelsfile, path):
    """
        Standardize the graph such that nodes are represented by consecutive integers in [0, #nodes-1] 
        edges: list of (undirected) edges
        nodelabels: list of nodes and their respective label 
    """
    node2int = {}
    inp = open(edgefile, 'r')
    f = open(path + 'edges_std.txt', 'w')
    cnt = 0
    for line in inp: #edge in edges:
        line = line.strip('\n')
        edge = line.split('\t')
        x, y = edge[0], edge[1]
        node2int.setdefault(x, len(node2int))
        node2int.setdefault(y, len(node2int))
        f.write(str(node2int[x]) + '\t' + str(node2int[y]) +'\n')
        cnt += 1
    f.close()
    inp.close()
    
    print('# nodes {}, # edges {}'.format(len(node2int), cnt))
    f = open(path + 'labels_std.txt', 'w')
    inp = open(nodelabelsfile, 'r')
    label2int = {}
    for line in inp: #nodelabels:
        line = line.strip('\n')
        nl = line.split('\t')
        node, label = nl[0], nl[1]
        label2int.setdefault(label, len(label2int))
        f.write(str(node2int[node]) + '\t' + str(label2int[label])+'\n')
    f.close()
    f = open(path + 'nodemap.txt', 'w')
    for node, node_idx in node2int.items():
        f.write(str(node_idx) + '\t' + str(node) + '\n')
    f.close()
    f = open(path + 'labelmap.txt', 'w')
    for label, label_idx in label2int.items():
        f.write(str(label_idx) + '\t' + str(label) + '\n')
    f.close()
    inp.close()


def edges_to_nx_undirected_graph(edgefile):
    G = nx.Graph()
    f = open(edgefile, 'r')
    for line in f:
        line = line.strip('\n')
        edge = line.split('\t')
        x, y = int(edge[0]), int(edge[1])
        G.add_edge(x, y)
    f.close()
    return G


def get_node_distribution(path, edgefile, alpha=0.75):
    """
        write the (smoothed) node frequencies to a file
    """
    f = open(edgefile, 'r')
    nodefreq = {}
    for line in f:
        x, y = int(line.split('\t')[0]), int(line.split('\t')[1])
        nodefreq.setdefault(x, 0)
        nodefreq.setdefault(y, 0)
        nodefreq[y] += 1
        nodefreq[x] += 1
    f.close()
    name = 'node_degrees_std.txt'
    f = open(path + name, 'w')
    s = sum([v**alpha for v in nodefreq.values()])
    print(sum(nodefreq.values())//2)
    for node, freq in nodefreq.items():
        f.write(str(node) + '\t' + str(freq**alpha/s) + '\n')
    f.close()

if __name__ == "__main__":

    # # Cora
    #path = '../Desktop/Data/Graphs/Cora/'
    #write_edges_cora(path, 'Cora')
    #standardize_graph(path+'edgelist.txt', path+'labels.txt', '../Desktop/Data/Graphs/smooth/Graphs_standard/Cora/')
    #path = '../Desktop/Data/Graphs/smooth/Graphs_standard/Cora/'
    #get_node_distribution(path, path+'edges_std.txt')

    # Citeseer
    path = '../Desktop/Data/Graphs/Citeseer/'
    write_edges_cora(path, 'Citeseer')
    standardize_graph(path+'edgelist.txt', path+'labels.txt', '../Desktop/Data/Graphs/smooth/Graphs_standard/Citeseer/')
    path = '../Desktop/Data/Graphs/smooth/Graphs_standard/Citeseer/'
    get_node_distribution(path, path+'edges_std.txt')

    # # Pubmed
    # path = '../Desktop/Data/Graphs/Pubmed/'
    # write_edges_pubmed(path, 'Pubmed')
    # standardize_graph(path+'edgelist.txt', path+'labels.txt', '../Desktop/Data/Graphs/smooth/Graphs_standard/Pubmed/')
    # path = '/Desktop/Data/Graphs/smooth/Graphs_standard/Pubmed/'
    # get_node_distribution(path, path+'edges_std.txt')

    # # Deezer..
    # path = '../Desktop/Data/Graphs/Deezer/'
    # write_edges_deezer(path)
    # standardize_graph(path+'edgelist.txt', path+'labels.txt', '../Dsktop/Data/Graphs/smooth/Graphs_standard/Deezer/')
    # path = '../Desktop/Data/Graphs/smooth/Graphs_standard/Deezer/'
    # get_node_distribution(path, path+'edges_std.txt')


    # # GitWebML
    # path = '../Desktop/Data/Graphs/GitWebML/'
    # write_edges_gitweb(path)
    # standardize_graph(path+'edgelist.txt', path+'labels.txt', '/Data/Grap..hs/smooth/Graphs_standard/GitWebML/')
    # path = '../Desktop/Data/Graphs/smooth/Graphs_standard/GitWebML/'
    # get_node_distribution(path, path+'edges_std.txt')


    # Flickr
    # path = '///Desktop/Data/Graphs/Flickr/'
    # standardize_graph(path+'edgelist.txt', path+'labels.txt', '../Desktop/Data/Graphs/smooth/Graphs_standard/Flickr/')
    # path = '../Desktop/Data/Graphs/smooth/Graphs_standard/Flickr/'
    # get_node_distribution(path, path+'edges_std.txt')
