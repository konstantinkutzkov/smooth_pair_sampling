import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dot, Embedding, Flatten, Dense
import random
import keras
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
from graph_utilities import skip_gram_pairs
import time
import os
from scipy.stats import norm
from collections import defaultdict


def write_embeddings_to_file(embeddings, fname):
    f = open(fname, 'w')
    for node, emb in enumerate(embeddings):
        f.write(str(node) + ':')
        for v in emb:
            f.write(str(v) + ' ')
        f.write('\n')
    f.close() 

def get_corpus_size(filecorpus):
    f = open(filecorpus, 'r')
    cnt = 0
    for line in f:
        cnt += 1
    f.close()
    return cnt

def get_nx_graph(datadir, graphname):
    f = open(datadir + graphname + '/edges_std.txt', 'r')
    G = nx.Graph()
    for line in f:#x, y in zip(edges_df['target'], edges_df['source']):
        line = line.strip('\n')
        edge = line.split('\t')
        x, y = int(edge[0]), int(edge[1])
        G.add_edge(x,y)
    f.close()
    return G

def load_frequencies(filename):
    nodefreq = {}
    f = open(filename, 'r')
    for line in f:
        node, freq = int(line.split('\t')[0]), float(line.split('\t')[1].strip('\n')) 
        nodefreq[node] = freq
    return nodefreq

# iterative version
def dfs(G, start_node, walks_num):
    stack=[]
    stack.append(start_node)
    seen=set()
    seen.add(start_node)
    walks = []
    adj_mask = set() # set(G.neighbors(start_node)) #adj_lists[start_node])
    while (len(stack)>0):
        vertex=stack.pop()
        nodes=G[vertex]
        for w in nodes:
            if w not in seen:
                stack.append(w)
                seen.add(w)
        if vertex in adj_mask:
            pass
        else:
            if vertex != start_node:
                walks.append(vertex)
        if len(walks) >= walks_num:
            break
    if len(walks) == 0:
        walks = [start_node]
    return walks

def intermediate(G, walks_num):
    candidate = defaultdict(list)
    for node in G.nodes():
        walk = dfs(G, node, walks_num)
        candidate[node].extend(walk)
    return candidate


def candidate_choose(G, walks_num):
    candidates = intermediate(G, walks_num)
    return candidates

def get_length(walks):
    length = 0
    for key in walks.keys():
        length += len(walks[key])
    return length

def negative_sampling(embeddings, candidates, start_given, q_1_dict, N_steps, node1, batch_size, alpha):
    distribution = [0.01] * 100
    # distribution = norm.pdf(np.arange(0,100,1), 50, 10)
    # distribution = norm.pdf(np.arange(0,100,1), 0, 50)
    # distribution = norm.pdf(np.arange(0,100,1), 100, 100)
    # distribution = norm.pdf(np.arange(0,100,1), 50, 100)
    distribution = [i/np.sum(distribution) for i in distribution]
    if start_given is None:
        start = np.random.choice(list(q_1_dict.keys()), batch_size)  # random init (user and item)
    else:
        start = start_given
    count = 0
    cur_state = start
    user_list = node1
    walks = defaultdict(list)
    generate_examples = list()
    while True:
        y_list = list()
        q_probs_list = list()
        q_probs_next_list = list()
        count += 1
        sample_num = np.random.random()
        if sample_num <= 0.5:
            y_list = np.random.choice(list(q_1_dict.keys()), len(cur_state), p=list(q_1_dict.values()))
            q_probs_list = [q_1_dict[i] for i in y_list]
            q_probs_next_list = [q_1_dict[i] for i in cur_state]
        else:
            for i in cur_state:
                if len(candidates[i]) == 0:
                    continue
                distr_i = [1/len(candidates[i])]*len(candidates[i])
                y = np.random.choice(candidates[i], 1, p=distr_i)[0]
                y_list.append(y)
                index = candidates[i].index(y)
                q_probs = distribution[index]
                q_probs_list.append(q_probs)
                node_list_next = candidates[y]
                if i in node_list_next:
                    index_next = node_list_next.index(i)
                    q_probs_next = distribution[index_next]
                else:
                    q_probs_next = q_1_dict[i]
                q_probs_next_list.append(q_probs_next) 
        u = np.random.rand()
        arr = np.sum(np.multiply(embeddings[user_list], embeddings[y_list]), axis=1)
        p_probs = np.sign(arr)*abs(arr)**alpha
        arr = np.sum(np.multiply(embeddings[user_list], embeddings[cur_state]), axis=1)
        p_probs_next = np.sign(arr)*abs(arr)**alpha
        A_a_list = np.multiply(np.array(p_probs), \
                    np.array(q_probs_next_list))/np.multiply(np.array(p_probs_next), np.array(q_probs_list))
        next_state = list()

        if count > N_steps:
            for i in list(range(len(cur_state))):
                walks[user_list[i]].append(cur_state[i])
            #cur_state = next_state
            #user_list = next_user
        else:
            for i in list(range(len(cur_state))):
                A_a = A_a_list[i]                        
                alpha = min(1, A_a)
                if u < alpha:
                    next_state.append(y_list[i])
                else:
                    next_state.append(cur_state[i])
            cur_state = next_state
        length = get_length(walks) 

        if length == batch_size:
            generate_examples = list()
            for user in node1:
                d = walks[user]
                if len(d) == 1:
                    generate_examples.append(d[0])    
                else:
                    generate_examples.append(d[0])
                    del walks[user][0]
            break
        else:
            continue  
    return generate_examples

def negative_sampling_old(embeddings, candidates, start_given, q_1_dict, N_steps, node1, batch_size, alpha):
    # q1_dict: the normalized node degrees dictionary, node: deg/sum(degrees)
    # candidates: dictionary node: dfs from node
    distribution = norm.pdf(np.arange(0,100,1), 50, 10) # q distribution in the paper
    # distribution = norm.pdf(np.arange(0,100,1), 0, 50)
    distribution = [i/np.sum(distribution) for i in distribution]
    # print('batch size', batch_size)
    if start_given is None:
        start = np.random.choice(list(q_1_dict.keys()), batch_size)  # random init (user and item)
    else:
        start = start_given
    count = 0
    cur_state = start
    user_list = node1
    # print('# user list', len(user_list))
    walks = defaultdict(list)
    generate_examples = list()
    while True:
        y_list = list()
        q_probs_list = list()
        q_probs_next_list = list()
        count += 1
        sample_num = np.random.random()
        if sample_num <= 0.5:
            y_list = np.random.choice(list(q_1_dict.keys()), len(cur_state), p=list(q_1_dict.values()))
            q_probs_list = [q_1_dict[i] for i in y_list]
            q_probs_next_list = [q_1_dict[i] for i in cur_state]
        else:
            for i in cur_state:
                if len(candidates[i]) == 100:
                    print('is 100')
                    y = np.random.choice(candidates[i], 1, p=distribution)[0]
                    y_list.append(y)
                    index = candidates[i].index(y)
                    q_probs = distribution[index]
                    q_probs_list.append(q_probs)
                    node_list_next = candidates[y]
                    if i in node_list_next: # if 
                        index_next = node_list_next.index(i)
                        q_probs_next = distribution[index_next]
                    else:
                        q_probs_next = q_1_dict[i]
                else:
                    y = np.random.choice(list(q_1_dict.keys()), 1, p=list(q_1_dict.values()))[0]
                    y_list.append(y)
                    q_probs_next = q_1_dict[i]
                    q_probs_list.append(q_1_dict[y])
                q_probs_next_list.append(q_probs_next) 
        u = np.random.rand()
        arr = np.sum(np.multiply(embeddings[user_list], embeddings[y_list]), axis=1)
        p_probs = np.sign(arr)*abs(arr)**alpha
        #(embeddings[user_list]*embeddings[y_list])**alpha
        # sess.run(model.p_probs, feed_dict={model.inputs1:user_list, model.inputs2:y_list , 
        #          model.batch_size: len(user_list)})

        arr = np.sum(np.multiply(embeddings[user_list], embeddings[cur_state]), axis=1)
        p_probs_next = np.sign(arr)*abs(arr)**alpha
        # p_probs_next = embeddings[user_list]*embeddings[cur_state]**alpha
        # p_probs_next = sess.run(model.p_probs, feed_dict={model.inputs1:user_list, .
        #                         model.inputs2:cur_state , model.batch_size: len(user_list)})

        # p_probs comes from embeddings, q_probs comes from uniform
        # q_prob_next should be q[x|y]
        #print('p probs', np.array(p_probs).shape)
        #print('q probs next', np.array(q_probs_next_list).shape)
        #S1 = np.sum(np.multiply(np.array(p_probs), np.array(q_probs_next_list)), axis=1) 
        #A_a_list = S1/np.multiply(np.array(p_probs_next), np.array(q_probs_list))
        A_a_list = np.multiply(np.array(p_probs), \
                    np.array(q_probs_next_list))/np.multiply(np.array(p_probs_next), np.array(q_probs_list))
        next_state = list()

        if count > N_steps:
            for i in list(range(len(cur_state))):
                walks[user_list[i]].append(cur_state[i])
            #cur_state = next_state
            #user_list = next_user
        else:
            for i in list(range(len(cur_state))):
                A_a = A_a_list[i]                        
                alpha = min(1, A_a)
                if u < alpha:
                    next_state.append(y_list[i])
                else:
                    next_state.append(cur_state[i])
            cur_state = next_state
        length = get_length(walks) 

        if length == batch_size:
            generate_examples = list()
            for user in node1:
                d = walks[user]
                if len(d) == 1:
                    generate_examples.append(d[0])    
                else:
                    generate_examples.append(d[0])
                    del walks[user][0]
            break
        else:
            continue  
    return generate_examples

class FromCorpusGenerator(tf.keras.utils.Sequence):

    def __init__(self, filecorpus, alpha, kind, current_embeddings, q_1_dict,
                    candidates, walk_length=80, windowsize=5, batchsize = 1000, nr_neg_samples=5):
        
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
        self.alpha = alpha
        assert kind == '' or kind == '_max'
        self.kind = kind
        self.walk_length = walk_length
        self.windowsize = windowsize
        self.batchsize = batchsize

        self.nr_neg_samples = nr_neg_samples

        self.epoch_cnt = 0
        self.index = 0
        
        self.current_embeddings = current_embeddings
        self.candidates = candidates
        self.q_1_dict = q_1_dict
        self.N_steps=10
        
        self.sampled_pos = 0

        self.nodes = sorted(list(q_1_dict.keys()))
        self.all_pairs = 0
        
        cnt = 0
        self.nr_lines = 0
        self.f = open(self.filecorpus, 'r')
        for line in self.f:
            assert len(line.split()) == self.walk_length
            cnt += len(skip_gram_pairs([1 for _ in range(self.walk_length)], self.windowsize))
        self.f.close()
        self.nr_pairs = cnt

        self.f = open(self.filecorpus, 'r')
        self.dummy = len(self.nodes)

        print(self.dummy, self.dummy in self.nodes)
        self.notfound = set()
        
    def __len__(self):
        return int((self.nr_pairs/self.batchsize)) 
    
    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.epoch_cnt += 1
        print('Pairs stats: # pos_pairs={}, # all_pairs={}, ratio={}'.format(self.sampled_pos, \
                                                self.all_pairs, self.sampled_pos/self.all_pairs))

    def __getitem__(self, idx):
        #print('Starting get item')
        # enough pairs sampled, return a dummy training example
        if self.sampled_pos >= self.nr_pairs:
            #print('we are here and # pairs is ', self.nr_pairs)
            self.f.close()
            return np.array([(self.dummy,self.dummy)]), np.array([1])
        samples = []
        labels = []
        sampled_nodes = []
        cnt_lines = 0
        # while cnt_lines < self.batchsize: #len(labels) < self.batchsize:
        while len(labels) < self.batchsize:
            line = self.f.readline()
            # print('\n reading line', line)
            if not line:
                print('\nreading file again')
                print('%sampled pairs', self.sampled_pos/self.nr_pairs)
                self.f.close()
                self.f = open(self.filecorpus, 'r')
                line = self.f.readline()
                print('\n reading line start')
                cnt_lines = 0
            cnt_lines += 1
            self.index +=1
            if self.index % 1000 == 0:
              print('% sampled pairs', self.sampled_pos/self.nr_pairs)
            # print('\nline', self.nr_lines + cnt_lines, ' : ', line) 
            line_split = line.split()
            walk = [int(n) for n in line_split if len(n)>0]
            walk_pairs = skip_gram_pairs(walk, self.windowsize)
            
            for pair in walk_pairs:
                samples.append((int(pair[0]), int(pair[1])))
                labels.append(1)
                sampled_nodes.extend([int(pair[0]) for _ in range(self.nr_neg_samples)])
                
        
        neg_samples = negative_sampling(self.current_embeddings, 
                            self.candidates, 
                            None, 
                            self.q_1_dict, 
                            self.N_steps, 
                            sampled_nodes, 
                            len(sampled_nodes), # self.batchsize,
                            alpha=self.alpha)
        for node, neg in zip(sampled_nodes, neg_samples):
            samples.append((node, neg))
            labels.append(0)
        
        self.nr_lines += cnt_lines 
        self.all_pairs += len(labels)
        self.sampled_pos += sum(labels)
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

    path = '../Desktop/Data/Graphs/smooth/Graphs_standard/'
    graphs = ['Cora', 'Citeseer', 'Pubmed', 'GitWebML', 'Flickr']
    graph = graphs[1]

    
    G = get_nx_graph(path, graph)
    candidates = candidate_choose(G, 10)
    print('graph read')
    
    prefixes = ['', 'reduced_']
    
    prefix = prefixes[0]
    
    #budget = 1
    alpha = 0.5
    nr_walks = 10
    kind = '_max'
    emb_size = 128
    batchsize = 1000
    context = True

    nodefreq = load_frequencies(path + '{}/node_degrees'.format(graph) + '_std.txt')
    print('nodefreq', len(nodefreq), sum(nodefreq.values())) 
    corpusfile = path + '{}/{}corpus_std.txt'.format(graph, prefix)

    corpus_size = get_corpus_size(corpusfile)

    print(f'Corpus:{corpusfile} of size {corpus_size}')
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    
    init_embeddings = np.random.normal(0,1, size=(len(nodefreq)+1,emb_size))
    gen = FromCorpusGenerator(corpusfile,  
                              alpha=alpha, 
                              kind=kind, 
                              q_1_dict=nodefreq,
                              candidates=candidates,
                              current_embeddings=init_embeddings,
                              batchsize=batchsize) 
    res = gen[0]
    print(np.mean(res[1]))

    sdw_model = SmoothDeepWalk(len(nodefreq)+1, 
                                embedding_dim=emb_size, 
                                init_embedding=init_embeddings
                                )
    sdw_model.reset_states() 
    sdw_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    #loss= 'binary_crossentropy', 
                    metrics=['AUC'])
    print('Fitting generator for {} prefix={}, kind={}, alpha={}, context={}'.format(graph, \
                                                prefix, kind, alpha, context))
    start = time.time()
    sdw_model.fit(gen, epochs=1, verbose=1)
    print('Elapsed time', time.time()-start)
    print('%sampled pos pairs', gen.sampled_pos/gen.nr_pairs)
    target_embeddings = sdw_model.get_layer('target_embedding').get_weights()[0]
    fname = f'../Desktop/Data/Graphs/competitors/{graph}/{prefix}neg_smooth.txt'
    write_embeddings_to_file(target_embeddings, fname)

    # nr_lines = gen.nr_lines
    # while nr_lines < corpus_size:
    #     print('NEW READ : nr lines in gen', nr_lines)
    #     sdw_model.fit(gen, epochs=1, verbose=1)
    #     target_embeddings = sdw_model.get_layer('target_embedding').get_weights()[0]
    #     print('target embeddings', np.mean(target_embeddings, axis=1))
    #     gen.current_embeddings = target_embeddings
    #     nr_lines = gen.nr_lines
    #     fname = \
    #     f'../Desktop/Data/Graphs/smooth/Graphs_standard/{graph}/embeddings/new_neg_smooth_{str(nr_lines)}.txt'
    #    write_embeddings_to_file(target_embeddings, fname)

                              
