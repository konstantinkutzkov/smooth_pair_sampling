# Code for paper _Learning Node Embeddings by Smooth Pair Sampling_
#### _Accepted for oral presentation at AISTATS 2025_

## Prerequisites

1. **Python packages** 
The code should work with a standard Python3 version. In addition, one needs some widely used libraries like pandas, numpy, networkx, scikit-learn, etc. The file requirements.txt gives a list of the package versions we used. In addition we provide an exhaustive list of all intalled packages in the virtual environment in freeze.txt. 
We recommend creating a virtual environment and installing the listed versions of the Python distribution and packages.

2. **Data sources** The provided package contains the 

 - [Cora graph](https://graphsandnetworks.com/the-cora-dataset/). 
 - [Citeseer graph](http://networkrepository.com/citeseer.php) 
 - [Pubmed graph](https://networkrepository.com/PubMed.php)
 - [Deezer graph](https://github.com/benedekrozemberczki/datasets/blob/master/README.md#deezer-social-networks) 

- [Git graph](https://github.com/benedekrozemberczki/datasets/blob/master/README.md#github-social-network)

- [Flickr graph](https://drive.google.com/drive/folders/1apP2Qn8r6G0jQXykZHyNT6Lz2pgzcQyL)

- Wikipedia and BlogCatalog have been downloaded from the links provided here: https://snap.stanford.edu/node2vec/#datasets (However, at the time of writing these instructuions, the link for BlogCatalog is broken.)

3. **Preprocessed graphs**
The above graphs come in different formats and we provide preprocessed versions in a common format. In particular, for each graph we provide the following files:
 - *edges_std.txt*: A list of edges of the graphs
 - *reduced_edges_std.txt*: A list of edges for the reduced after removing 20% of the edges, used for link prediction.
 - *removed_edges_std.txt*: The removed 20% edges from the original graph, used for link prediction.
 - *node_degrees.txt* and *reduced_node_degrees.txt*: The normalized node degrees, used for negative sampling.
 - *labels_std.txt*: The label(s) per node

## Running the code

1. **Generate and sketch the corpus**
 
       python code/sketching/write_corpus_and_sketches.py 

Usage:

       --graph [Citeseer]    Input graph name
       --b 10            Sketch budget as percentage
       --prefix []       Set to reduced_ for learning embeddings on the reduced graph for link prediction, otherwise leave empty

Note that we have generated the corpus and sketches for the Citeseer graph such that reviewers can directly train the embeddings.

2. **Train embeddings**


       python code/tf_smooth_from_corpus.py

Usage:

       --graph [Citeseer]          Input graph name
       --beta [0.5]            Smoothing exponent
       --b [10]                Sketch budget as percentage
       --prefix []             Set to reduced_ for learning embeddings on the reduced graph for link prediction
       --embedding_size [128]  The dimensionality of the embeddings
       --nr_walks [10]         The number of walks per node
       --walk_length [80]      The length of the random walks
       --windowsize [10]       The window size for the skip-gram model
       --batchsize [1000]      The batch size of pairs (note that we use 10000 for the larger graphs)
       --neg_samples [5]       Number of negative samples
    
Above in brackets we give the default arguments

3. **Node classification**

       python code/evaluation/node_classification.py

Usage:

       --graphname [Citeseer]   Input graph name
       --b [10]             Sketch budget as percentage
       --beta [0.5]         Smoothing exponent

4. **Link prediction**

       python code/evaluation/link_prediction.py

Usage

       --graphname [Citeseer]   Input graph name
       --b [10]             Sketch budget as percentage
       --beta [0.5]         Smoothing exponent
       --k [100]            top k pairs to return

5. **Additional code**
We also provide some additional code about how we preprocess the graphs and how we create the reduced graph for training node embeddings. This code is provided only for review and cannot be executed without modifications.
