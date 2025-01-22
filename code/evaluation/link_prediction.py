import numpy as np
import time
import math
import argparse
import inspect
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc, balanced_accuracy_score as bacc, roc_auc_score, f1_score, dcg_score
from sklearn.neighbors import NearestNeighbors


def get_metrics_fast(embeddings, existing_edges, removed_edges, sample_ratio, seed, k):
    np.random.seed(seed)
    nr = embeddings.shape[0]
    nr_pairs = nr**2
    indices1 = np.random.choice(list(range(nr)), size=int(sample_ratio*nr_pairs), replace=True)
    indices2 = np.random.choice(list(range(nr)), size=int(sample_ratio*nr_pairs), replace=True)
    #print('sampling done')
    scores = []
    for i, j in zip(indices1, indices2):
        pair = (i,j)
        if i < j and pair not in existing_edges: 
            # adding the random value to avoid bias in the very unlikely event 
            # that different nodepairs have the very same dot product
            scores.append((np.dot(embeddings[i], embeddings[j]), np.random.random(), int(pair in removed_edges)))
    #print('added scores')
    removed_edges_list = list(sorted(removed_edges))
    pos_samples = np.random.choice(list(range(len(removed_edges_list))), 
                                   size=int(sample_ratio*len(removed_edges_list)), replace=False)
    # print('# pos samples', len(pos_samples))
    for idx in pos_samples:
        pos_pair = removed_edges_list[idx]
        # again adding a random value to avoid unlikely bias
        scores.append((np.dot(embeddings[pos_pair[0]], embeddings[pos_pair[1]]),  np.random.random(), 1))
    scores_k = sorted(scores, reverse=True)[:k]
    labels = [sc[2] for sc in scores_k]
    all_pos = sum([sc[2] for sc in scores])
    
    if sum(labels) == 0:
        # print('All labels are zero')
        return 0, 0, 0
    prec = sum(labels)/k
    rec = sum(labels)/all_pos
    # print(prec, rec)
    return (2*prec*rec)/(prec+rec), prec, rec
    
def get_edges(edges_path):
    edges = set()
    nodes = set()
    with open(edges_path, 'r') as f:
        for edge in f:
            edge_split = edge.split('\t')
            u = int(edge_split[0].strip())
            v = int(edge_split[1].strip('\n'))
            edges.add((u, v)) 
            # edges.add((v, u)) 
            nodes.add(u)
            nodes.add(v)
    return edges, list(nodes)

def read_embeddings(fname):
    embmap = {}
    f = open(fname, 'r')
    for line in f:
        node, vec = int(line.split(':')[0]), line.split(':')[1]
        vals = []
        for v in vec.split():
            vals.append(float(v))
        embmap[node] = vals/np.linalg.norm(vals)
    f.close()
    emb_array = np.zeros((len(embmap), len(embmap[0])))
    for node, emb in embmap.items():
        emb_array[node] = emb
    return emb_array

def read_samples(path, seed):
    samples = []
    f = open(path + 'sampled_pairs_'+str(seed)+'.txt', 'r')
    for line in f:
        u, v = int(line.split('\t')[0]), int(line.split('\t')[1].strip('\n'))
        samples.append((u,v))
    f.close()
    return samples

def write_results(filename, results):
    f = open(filename, 'w')
    for r in results:
        f.write(str(r)+'\n')
    f.close()

if __name__ == "__main__":

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    root_dir = os.path.dirname(parentdir)
    sys.path.insert(0, parentdir)

    parser = argparse.ArgumentParser(description="Corpus generation and sketching.")
    parser.add_argument('--graph', nargs='?', default='Citeseer', help='Input graph name')
    parser.add_argument('--b', type=int, default='10', help='Sketch budget as percentage')
    parser.add_argument('--beta', type=float, default='0.5', help='Smoothing exponent')
    parser.add_argument('--k', type=int, default='100', help='top k pairs to return')

    args = parser.parse_args()
    graph = args.graph
    beta = args.beta
    budget = args.b
    k = args.k

    existing_edges, nodes = get_edges(root_dir + f'/data/{graph}/reduced_edges_std.txt')
    removed_edges, _ = get_edges(root_dir + f'/data/{graph}/removed_edges_std.txt')
    # removed_edges_list = list(sorted(removed_edges))
    
    # budget = 1
    idx = 0
    emb_files = []
    names = []
    
    algo =  str(beta) + '_' + str(budget) 
    embfile = root_dir + f'/data/{graph}/embeddings/reduced_tf_embs_' + algo + '.txt'
    if not os.path.exists(embfile):
        print(embfile)
        print(f'No data for beta={beta}, budget={budget}')
        sys.exit(0)

    print('Graph {}, k: {}, beta: {}, budget: {}'.format(graph, k, beta, budget))
    embs = read_embeddings(embfile)
    f1s, precs, recalls = [], [], []
    for i in range(100):
        f1, prec, rec = get_metrics_fast(embs, existing_edges, removed_edges, sample_ratio=0.001, seed=i, k=k)
        f1s.append(f1)
        precs.append(prec)
        recalls.append(rec)
        if (i+1) % 10 == 0 and i < 100:
            print(i+1, 'mean F1', np.round(np.mean(f1s), 4))
    print('mean F1', np.round(np.mean(f1s), 4))
    print('mean precision', np.round(np.mean(precs), 4))
    print('mean recall', np.round(np.mean(recalls), 4))
    print('\n')
