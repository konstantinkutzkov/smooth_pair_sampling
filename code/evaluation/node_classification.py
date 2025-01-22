import os
import inspect
import argparse
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc, balanced_accuracy_score as bacc, roc_auc_score, f1_score

from sklearn.neighbors import KNeighborsClassifier

def read_labels(fname):
    labelmap = {}
    f = open(fname, 'r')
    for line in f:
        line = line.strip('\n')
        lb = line.split('\t')[1]
        lb = int(lb.split(',')[0])
        node, label = int(line.split('\t')[0]), int(line.split('\t')[1])
        #node = int(line.split('\t')[0])
        labelmap[node] = label
    f.close()
    return labelmap
    # with open(fname) as data_file:
    #     labelmap = json.load(data_file)
    # return labelmap

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

def read_embeddings_dw(fname):
    embmap = {}
    f = open(fname, 'r')
    cnt = 0
    for line in f:
        cnt += 1
        if cnt == 1:
            continue
        node, vec = line.split()[0], line.split()[1:]
        vals = []
        for v in vec:
            vals.append(float(v))
        embmap[int(node)] = vals
    f.close()
    return embmap


def one_hot(y, unique): 
    encoded = np.zeros((len(y), unique))
    for i, label_i in enumerate(y):
        encoded[i, label_i] = 1
            
    return encoded

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

    args = parser.parse_args()
    graph = args.graph
    beta = args.beta
    budget = args.b

    labelfile = root_dir + f'/data/{graph}/labels_std.txt'
    labelmap = read_labels(labelfile)

    print('\nbeta={}, budget={}'.format(beta, budget))
    normed = False
    methods = ['smooth', 'karate']
    kind = '_max'
    idx = 0
    method = methods[idx]
    context = True
    ctx = ''
    
    embfile = root_dir + f'/data/{graph}/embeddings/tf_embs_{beta}_{budget}.txt'
    
    print(embfile)
    if not os.path.exists(embfile):
        print(f'No data for beta={beta}, budget={budget}')
        sys.exit(0)
    embs = read_embeddings(embfile)

    X, y = [], []
    for node, label in labelmap.items():
        if node not in embs:
            print('node',node)
            continue
        if normed:
            X.append(embs[node]/np.linalg.norm(embs[node]))
        else:
            X.append(embs[node])
        y.append(label)

    y = LabelEncoder().fit_transform(y)

    for ts in [1]:
        train_size = ts/10
       
        f1_micros, f1_macros, f1_weights = [], [], []
        #auc_micros, auc_macros = [], []
    
        print('Graph {},  beta={}, budget={}, train_size={}'.format(graph, beta, budget,  train_size))
        i = 0
        while len(f1_micros) < 100:
            i += 1
            if i % 50 == 0:
                print(len(f1_micros), np.round(100*np.mean(f1_macros), 2))
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=73*i)#, stratify=y)
            if len(np.unique(y_train)) != len(np.unique(y_test)):
                continue
           
            clf = LogisticRegression(random_state=i, max_iter=2000)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            #print(y_pred)
            f1_micros.append(f1_score(y_test, y_pred, average='micro')) 
            f1_macros.append(f1_score(y_test, y_pred, average='macro'))  
        print('macro-F1: ', np.round(100*np.mean(f1_macros), 2))
        print('micro-F1: ', np.round(100*np.mean(f1_micros), 2)) 
            
