import pandas as pd
import os 
import json


def pandas_graph_to_edgelist(edges_df, fname):
    
    print(len(set(edges_df['target']).union(set(edges_df['source']))))
    f = open(fname, 'w')
    for x, y in zip(edges_df['target'], edges_df['source']):
        f.write(str(x)+' '+str(y)+'\n')
    f.close()

path = '../Data/Graphs/' 
graphs = ['Cora', 'Citeseer', 'Pubmed', 'GitWebML', 'Deezer']

   
# Cora
graph = graphs[0]
content = pd.read_csv(os.path.join(path + graph, graph.lower() +".content"), sep='\t', header=None)
feature_names = ["w-{}".format(ii) for ii in range(content.shape[1]-2)]
column_names =  ['node'] + feature_names + ["label"]
nodedata = pd.read_csv(os.path.join(path + graph, graph.lower() +".content"), sep='\t', header=None, names=column_names)
edges_df = pd.read_csv(os.path.join(path + graph, graph.lower() + ".cites"), sep='\t', header=None, names=["target", "source"])
#pandas_graph_to_edgelist(edges_df, path + graph + '/edgelist.txt')

labelmap ={}
for node, label in zip(nodedata['node'], nodedata['label']):
    labelmap[node] = label

labelfile = path + graph + '/labels.json'
with open(labelfile, 'w', encoding='utf-8') as f:
    json.dump(labelmap, f, ensure_ascii=False, indent=4)


labelfile = path + graph + '/labels.txt'
with open(labelfile, 'w', encoding='utf-8') as f:
    for node, label in labelmap.items():
        f.write(str(node) + '\t' + str(label) + '\n')

# Citeseer
graph = graphs[1]
content = pd.read_csv(os.path.join(path + graph, graph.lower() +".content"), sep='\t', header=None)
feature_names = ["w-{}".format(ii) for ii in range(content.shape[1]-2)]
column_names =  ['node'] + feature_names + ["label"]
nodedata = pd.read_csv(os.path.join(path + graph, graph.lower() +".content"), sep='\t', header=None, names=column_names)

labelmap ={}
for node, label in zip(nodedata['node'], nodedata['label']):
    labelmap[node] = label

labelfile = path + graph + '/labels.json'
with open(labelfile, 'w', encoding='utf-8') as f:
    json.dump(labelmap, f, ensure_ascii=False, indent=4)


labelfile = path + graph + '/labels.txt'
with open(labelfile, 'w', encoding='utf-8') as f:
    for node, label in labelmap.items():
        f.write(str(node) + '\t' + str(label) + '\n')

# Pubmed
graph = graphs[2]

nodes = set()
with open(os.path.join(path+graph, graph.lower() + ".cites"), 'r') as edgefile:
    for line in edgefile:
        line_split = line.split('|')
        if len(line_split) > 1:
            l0 = line_split[0]
            l1 = line_split[1]
            u = l0.split(':')[1]
            v = l1.split(':')[1]
            nodes.add(str(u).strip())
            nodes.add(str(v).strip())

nodedata = {}
labelmap = {}
with open(os.path.join(path+graph, graph.lower() + ".content"), 'r') as contentfile:
    for line in contentfile:
        line_split = line.split()
        #print(line_split[:3])
        if len(line_split) < 3:
            continue
        if line_split[0] not in nodes:
            continue
        nodewords = {}
        for i in range(2, len(line_split)):
            w = line_split[i]
            w_split = w.split('=')
            if w_split[0] == 'summary':
                continue
            # this is the label
            nodewords[w.split('=')[0]] = float(w.split('=')[1])
        nodedata[line_split[0]] = (line_split[1], nodewords)
        labelmap[line_split[0]] = line_split[1]
        
labelfile = path + graph + '/labels.json'
with open(labelfile, 'w', encoding='utf-8') as f:
    json.dump(labelmap, f, ensure_ascii=False, indent=4)

labelfile = path + graph + '/labels.txt'
with open(labelfile, 'w', encoding='utf-8') as f:
    for node, label in labelmap.items():
        f.write(str(node) + '\t' + str(label) + '\n')


# GitWebML

# graph = graphs[4]

# nodedata = pd.read_csv(path + graph + '/target.csv', )

# labelmap ={}
# for node, label in zip(nodedata['id'], nodedata['ml_target']):
#     labelmap[node] = label

# labelfile = path + graph + '/labels.json'
# with open(labelfile, 'w', encoding='utf-8') as f:
#     json.dump(labelmap, f, ensure_ascii=False, indent=4)


labelfile = path + graph + '/labels.txt'
with open(labelfile, 'w', encoding='utf-8') as f:
    for node, label in labelmap.items():
        f.write(str(node) + '\t' + str(label) + '\n')


# Deezer 
graph = graphs[5]

nodedata = pd.read_csv(path + graph + '/target.csv', )

labelmap ={}
for node, label in zip(nodedata['id'], nodedata['target']):
    labelmap[node] = label

# labelfile = path + graph + '/labels.json'
# with open(labelfile, 'w', encoding='utf-8') as f:
#     json.dump(labelmap, f, ensure_ascii=False, indent=4)


labelfile = path + graph + '/labels.txt'
with open(labelfile, 'w', encoding='utf-8') as f:
    for node, label in labelmap.items():
        f.write(str(node) + '\t' + str(label) + '\n')
