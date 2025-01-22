import networkx as nx
import copy
import numpy as np

def sprasify_edges(path, threshold=0.2):
    
    G = nx.Graph()
    edgefile = path + 'edges_std.txt'
    
    f = open(edgefile, 'r')
    for line in f:
        x_y = line.split('\t')
        x, y = int(x_y[0]), int(x_y[1].strip('\n'))
        G.add_edge(x, y)
    f.close()

    removed_edges = []
    
    H = copy.deepcopy(G)
    while len(removed_edges) < threshold*G.number_of_edges():
        if len(removed_edges)%500 == 0:
            print(len(removed_edges), G.number_of_edges())
        i = np.random.randint(0, H.number_of_edges()-1) #get_rnd_int_in_range(stream, 0, H.number_of_edges()-1)
        edge = list(H.edges())[i]
        u = edge[0]
        v = edge[1]
        if H.degree[u] > 1 and H.degree[v] > 1:
            H.remove_edge(u, v)
            removed_edges.append((u, v))

    print(G.number_of_nodes(), H.number_of_nodes()) 
    suffix = 'reduced_edges_std.txt'
    reduced_edgefile = path + suffix
    f = open(reduced_edgefile, 'w')
    nns = set()
    for edge in H.edges():
        f.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
        nns.add(edge[0])
        nns.add(edge[1])
    
    # for node in G.nodes():
    #     if node not in nns:
    #         print('adding node', node)
    #         f.write(str(node) + '\t' + str(node) + '\n')
    # f.close()

    suffix = 'removed_edges_std.txt'
    removed_file = path + suffix
    f = open(removed_file, 'w')
    for edge in removed_edges:
        f.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
    f.close()


if __name__ == "__main__":

    # # Cora
    # print('Cora')
    # path = '../Desktop/Data/Graphs/smooth/Graphs_standard/Cora/'
    # sprasify_edges(path)

    # # Citeseer
    # print('Citeseer')
    # path = '../Desktop/Data/Graphs/smooth/Graphs_standard/Citeseer/'
    # sprasify_edges(path)

    # # Pubmed
    # print('Pubmed')
    # path = '../Desktop/Data/Graphs/smooth/Graphs_standard/Pubmed/'
    # sprasify_edges(path)
    
    # Flickr
    # print('Flickr')
    # path = '../Desktop/Data/Graphs/smooth/Graphs_standard/Flickr/'
    # sprasify_edges(path)

    print('Deezer')
    path = '../Desktop/Data/Graphs/smooth/Graphs_standard/Deezer/'
    sprasify_edges(path)
