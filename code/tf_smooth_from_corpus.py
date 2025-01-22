import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dot, Embedding, Flatten, Dense
import random
import keras
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import inspect
import argparse
#sys.path.insert(0, '../smooth_node_embeddings/')
from graph_utilities import skip_gram_pairs
import time


def load_sketch(filename):
    sketch = {}
    f = open(filename, 'r')
    for line in f:
        if len(line.split(':')) != 2:
            minfreq = float(line)
            continue
        x_y = line.split(':')[0]
        x, y = int(x_y.split(',')[0]), int(x_y.split(',')[1])
        val = float(line.split(':')[1])
        sketch[(x,y)] = val
    f.close()
    return sketch, minfreq

def load_frequencies(filename):
    nodefreq = {}
    f = open(filename, 'r')
    for line in f:
        node, freq = int(line.split('\t')[0]), float(line.split('\t')[1].strip('\n')) 
        nodefreq[node] = freq
    return nodefreq

def write_embeddings_to_file(embeddings, fname):
    f = open(fname, 'w')
    for node, emb in enumerate(embeddings):
        f.write(str(node) + ':')
        for v in emb:
            f.write(str(v) + ' ')
        f.write('\n')
    f.close() 


class FromCorpusGenerator(tf.keras.utils.Sequence):

    def __init__(self, filecorpus, sketch, minfreq, nodefreq, beta, kind='_max',
                    walk_length=80, windowsize=10, batchsize = 1000, nr_neg_samples=5):
        
        """Initialization
        :param nx.Graph G: the original graph
        :param dict sketch: the precomputed Frequent sketch
        :param int minfreq: the default frequency of pairs not in the sketch
        :param int windowsize: for the generation of pairs from random walks
        :param float beta: the power for #(w,c) for asmooth positive sampling
        :param str kind: what is the default frequency of pairs not in the sketch
        :param int nr_neg_samples: the number of negative pairs per positive pair
        """
        self.filecorpus = filecorpus
        self.sketch = sketch
        self.minfreq = minfreq
        self.nodefreq = nodefreq
        self.beta = beta
        assert kind == '' or kind == '_max'
        self.kind = kind
        self.walk_length = walk_length
        self.windowsize = windowsize
        self.batchsize = batchsize

        self.printit = True

        self.nr_neg_samples = nr_neg_samples

        self.epoch_cnt = 0
        self.index = 0

        self.sampled_pos = 0

        self.nodes = sorted(list(nodefreq.keys()))
        self.all_pairs = 0
        
        cnt = 0
        self.f = open(self.filecorpus, 'r')
        for line in self.f:
            assert len(line.split()) == self.walk_length
            cnt += len(skip_gram_pairs([1 for _ in range(self.walk_length)], self.windowsize))
        self.f.close()
        self.nr_pairs = cnt
        print('Total number of positive pairs', cnt)

        self.f = open(self.filecorpus, 'r')
        self.dummy = len(self.nodes)

        print(self.dummy, self.dummy in self.nodes)

        self.notfound = set()

    def __len__(self):
        return  int((self.nr_pairs/self.batchsize)) 

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.epoch_cnt += 1
        print('Pairs stats: # pos_pairs={}, # all_pairs={}, ratio={}'.format(self.sampled_pos, \
                                                self.all_pairs, self.sampled_pos/self.all_pairs))


    def __getitem__(self, idx):
        # enough pairs sampled, return a dummy training example
        if self.sampled_pos >= self.nr_pairs:
            if self.printit:
                print('we are here and # pairs is ', self.nr_pairs)
                self.printit = False
            self.f.close()
            return np.array([(self.dummy,self.dummy)]), np.array([1])
        samples = []
        labels = []
        sampled_nodes = []
        while len(labels) < self.batchsize:
            line = self.f.readline()
            # self.index += 1
            # if self.index % 2000 == 0:
            #     print('%sampled pairs', self.sampled_pos/self.nr_pairs)
            if not line:
                print('\nreading file again')
                print('%sampled pairs', self.sampled_pos/self.nr_pairs)
                self.f.close()
                self.f = open(self.filecorpus, 'r')
                line = self.f.readline()
            
            line_split = line.split()
            walk = [int(n) for n in line_split if len(n)>0]
            walk_pairs = skip_gram_pairs(walk, self.windowsize)
            for pair in walk_pairs:
                freq = 1
                if len(self.kind) > 1: # if kind is max assume minfreq as default freq of pair
                    freq = self.minfreq
                if pair in self.sketch and self.sketch[pair] > self.minfreq:
                    if len(self.kind) > 1:
                        freq = max(self.minfreq, self.sketch[pair])
                    else:
                        freq = self.sketch[pair]
                r = np.random.random()
                if freq**(self.beta-1) > r:
                    samples.append((int(pair[0]), int(pair[1])))
                    labels.append(1)
                    sampled_nodes.extend([int(pair[0]) for _ in range(self.nr_neg_samples)])
        # print('# samples in batch', len(labels))
        np.random.seed(self.sampled_pos)    
        r_negs = np.random.choice(list(self.nodefreq.keys()), len(sampled_nodes), list(self.nodefreq.values()))
        assert len(sampled_nodes) == len(r_negs)
        for node, neg in zip(sampled_nodes, r_negs):
            samples.append((node, neg))
            labels.append(0)
        self.sampled_pos += sum(labels)  
        self.all_pairs += len(labels)
        assert len(samples) == len(labels)

        return np.array(samples), np.array(labels)


class SmoothDeepWalk(Model):
    def __init__(self, nr_nodes, embedding_dim, init_embedding, context=False):
        super(SmoothDeepWalk, self).__init__()
        self.context = context
        self.target_embedding = Embedding(nr_nodes,
                                          embedding_dim,
                                          embeddings_initializer=tf.keras.initializers.Constant(init_embedding), 
                                          #"RandomNormal",
                                          input_length=1,
                                          name="target_embedding")
        self.context_embedding = Embedding(nr_nodes,
                                          embedding_dim,
                                          embeddings_initializer=tf.keras.initializers.Constant(init_embedding), 
                                          #"RandomNormal",
                                          input_length=1)
        self.dots = Dot(axes=1)
        self.flatten = Flatten()
        self.dense = Dense(1, activation='sigmoid')

    def call(self, pair):
        target, context = pair[:,0], pair[:,1]
        word_emb = self.target_embedding(target)
        if self.context:
            context_emb = self.context_embedding(context)
        else:
            context_emb = self.target_embedding(context)
        dots = self.dots([word_emb, context_emb])
        flat = self.flatten(dots)
        return self.dense(flat)


if __name__ == "__main__":

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_dir = os.path.dirname(currentdir)
    #sys.path.insert(0, parentdir)

    parser = argparse.ArgumentParser(description="Embedding generation for the different methods.")
    parser.add_argument('--graph', nargs='?', default='Citeseer', help='Input graph name')
    parser.add_argument('--beta', type=float, default='0.5', help='Smoothing exponent')
    parser.add_argument('--b', type=int, default='10', help='Sketch budget as percentage')
    parser.add_argument('--prefix', nargs='?', default='', help='Set to reduced_ for learning embeddings on the reduced graph for link prediction')
    parser.add_argument('--embedding_size', type=int, default=128, help='The dimensionality of the embeddings')
    parser.add_argument('--nr_walks', type=int, default=10, help='The number of walks per node')
    parser.add_argument('--walk_length', type=int, default=80, help='The length of the random walks')
    parser.add_argument('--windowsize', type=int, default=10, help='The window size for the skip-gram model')
    parser.add_argument('--batchsize', type=int, default=1000, help='The batch size of pairs')
    parser.add_argument('--neg_samples', type=int, default=5, help='Number of negative samples')
    
    args = parser.parse_args()
    graph = args.graph
    beta = args.beta
    budget = args.b
    emb_size = args.embedding_size # how many samples per node to generate 
    prefix = args.prefix
    nr_walks = args.nr_walks
    walk_length = args.walk_length
    windowsize = args.windowsize
    batchsize = args.batchsize
    nr_neg_samples = args.neg_samples

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    pathsketch =  root_dir + f'/data/{graph}/sketches/{prefix}budget_{budget}_walks_{nr_walks}_wl_{walk_length}_std.txt'
    sketch, minfreq = load_sketch(pathsketch)
    print('sketchsize', len(sketch), minfreq)
    nodefreq_path =  root_dir + f'/data/{graph}/{prefix}node_degrees_std.txt'
    nodefreq = load_frequencies(nodefreq_path)
    print(nodefreq_path)
    print('nodefreq', len(nodefreq), sum(nodefreq.values())) 
    corpusfile = root_dir + f'/data/{graph}/{prefix}corpus_{nr_walks}_{walk_length}_std.txt'
    print('Corpus:', corpusfile)

    gen = FromCorpusGenerator(corpusfile, 
                            sketch=sketch, 
                            minfreq=minfreq, 
                            nodefreq=nodefreq, 
                            beta=beta,  
                            batchsize=batchsize,
                            walk_length=walk_length,
                            windowsize=windowsize,
                            nr_neg_samples=nr_neg_samples) 

    init_embeddings = np.random.normal(0,1, size=(len(nodefreq)+1,emb_size))
    sdw_model = SmoothDeepWalk(len(nodefreq)+1, 
                            embedding_dim=emb_size, 
                            init_embedding=init_embeddings
                            )
    sdw_model.reset_states() 
    sdw_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    #loss= 'binary_crossentropy', 
                    metrics=['AUC'])
    print('Fitting generator for {} prefix={}, beta={}, budget={}'.format(graph, prefix, beta, budget))
    start = time.time()
    sdw_model.fit(gen, epochs=1, verbose=1)
    print('Elapsed time', time.time()-start)
    print('%sampled pos pairs', gen.sampled_pos/gen.nr_pairs)
    target_embeddings = sdw_model.get_layer('target_embedding').get_weights()[0]
    fname = root_dir + f'/data/{graph}/embeddings/{prefix}tf_embs_{beta}_{budget}.txt'
    write_embeddings_to_file(target_embeddings, fname)     
    sdw_model.reset_states() 
    del sdw_model
    tf.keras.backend.clear_session()
