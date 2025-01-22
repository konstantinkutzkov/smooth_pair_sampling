import os
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc, balanced_accuracy_score as bacc, roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

def one_hot(y, unique): 
    encoded = np.zeros((len(y), unique))
    for i, label_i in enumerate(y):
        encoded[i, label_i] = 1
    return encoded

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def get_freqs(y):
    freqmap = defaultdict(list)
    for i, y_i in enumerate(y):
        for l in y_i:
            freqmap[l].append(i)
    return freqmap

def stratified_class_sampling(X, y, seed, threshold=0.1):
    freqmap = get_freqs(y)
    samples = []
    np.random.seed(seed)
    for _, vals in freqmap.items():
        size = int(len(vals)*threshold)
        sample = np.random.choice(vals, size, replace=False)
        samples.extend(sample)
    train_indices = list(set(samples))
    test_indices = [i for i in range(len(y)) if i not in set(samples)]
    its = set(train_indices).intersection(set(test_indices))
    assert len(its) == 0
    return [X[idx] for idx in train_indices], [X[idx] for idx in test_indices], \
            [y[idx] for idx in train_indices], [y[idx] for idx in test_indices]


def get_X_y(fname, embs):
    f = open(fname, 'r')
    X, y = [], []
    labelset = set()
    for line in f:
        line = line.strip('\n')
        node = int(line.split('\t')[0])
        lb = line.split('\t')[1]
        lb = lb.split(',')
        X.append(embs[int(node)])
        y.append([int(l) for l in lb])
        labelset.update(set([int(l) for l in lb]))
    f.close()
    return X, y, labelset #one_hot(y, len(labelset))

def read_embeddings(fname):
    embmap = {}
    f = open(fname, 'r')
    for line in f:
        node, vec = int(line.split(':')[0]), line.split(':')[1]
        vals = []
        for v in vec.split():
            vals.append(float(v))
        embmap[node] = vals
    f.close()
    return embmap

def write_results(filename, results):
    f = open(filename, 'w')
    for r in results:
        f.write(str(r)+'\n')
    f.close()

path = '../Desktop/Data/Graphs/smooth/Graphs_standard/' 
graphs = ['Wikipedia', 'BlogCatalog']
graph = graphs[0]
labelfile = path + graph + '/labels_std.txt'
#labelmap = read_labels(labelfile)

for budget in [1]: #[10, 1, 5, 20]:
    for beta in [0.7]: #0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        print('\nbeta={}, budget={}'.format(beta, budget))
        normed = False
        methods = ['smooth', 'karate']
        kind = '_max'
        idx = 0
        method = methods[idx]
        context = True
        ctx = ''
        if context:
            ctx = '_context'
        embfile = \
        '../Desktop/Data/Graphs/smooth/Graphs_standard/{}/embeddings/new_tf_target_embs_{}_{}{}{}.txt'.format(graph,\
                str(beta), str(budget), kind, ctx)
        print(embfile)
        if not os.path.exists(embfile):
            print(f'No data for beta={beta}, budget={budget}')
            continue
        embs = read_embeddings(embfile)


        X, y, labelset = get_X_y(labelfile, embs)
        freqmap = get_freqs(y)
        #print(sorted(embs.keys()))
        for ts in [1]: #range(1, 2):
            train_size = ts/10
            
            f1_micros, f1_macros, f1_weights = [], [], []
            print('Graph {}, beta={}, budget={}, kind={}, normed={}, train_size={}'.format(graph,
                                                                            beta, budget, kind, normed, train_size))
            i = 0
            while len(f1_micros) < 40:
                i += 1
                if i % 10 == 0:
                    print(i, np.round(100*np.mean(f1_macros), 2))
                X_train, X_test, y_train, y_test = stratified_class_sampling(X, y, seed=i, threshold=train_size)
                y_train = one_hot(y_train, len(labelset))
                y_test = one_hot(y_test, len(labelset))
             
                clf = LogisticRegression(random_state=i, multi_class='ovr')
                model = OneVsRestClassifier(clf)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                f1_micros.append(f1_score(y_test, y_pred, average='micro', zero_division=1)) 
                f1_macros.append(f1_score(y_test, y_pred, average='macro'))  
            
            print('micro  ', 'macro ')
            print(np.round(100*np.mean(f1_micros), 2), np.round(100*np.mean(f1_macros), 2)) 

            path = '../Desktop/Data/Graphs/smooth/Graphs_standard/' + graph + '/results/clf_'
            name = graph + '_mlb_' + str(beta) + '_' + str(budget) + '_' + str(train_size)
            path += name #+ str(budget) + '_' + str(beta) 
            write_results(path + '_micros.txt', f1_micros)
            write_results(path + '_macros.txt', f1_macros)
            #write_results(path + '_weighted.txt', f1_weights)
