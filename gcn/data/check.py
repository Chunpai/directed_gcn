import sys
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pickle as pkl


def load_data(dataset_str):
    for run in range(1):
        names = ['features', 'labels', 'directed.graph','undirected.graph']
        objects = []
        for i in range(len(names)):
            with open("{}/{}.{}".format(dataset, dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        features, labels, directed_graph, undirected_graph = tuple(objects)

        print features.shape
        g = nx.from_dict_of_lists(undirected_graph)
        centrality =  nx.eigenvector_centrality(dg)
        
        centr = np.array([centrality[i] for i in range(2708)])
        print centr
        # centr =  np.tile(centr,(2708,1))

        directed_adj = nx.adjacency_matrix(nx.from_dict_of_lists(directed_graph, create_using=nx.DiGraph()))
        directed_adj = sp.csr_matrix(np.array(directed_adj.todense()) * centr)
        undirected_adj = nx.adjacency_matrix(nx.from_dict_of_lists(undirected_graph))

        objects = []
        names = ['train.index', 'val.index', 'test.index']
        for j in range(len(names)):
            with open("{}/{}/{}.{}".format(dataset,run,dataset,names[j]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        idx_train, idx_val, idx_test = tuple(objects)
            
        # a = np.sum(labels[idx_train], axis=0)
        # print([round(e*1.0 / a.sum(),4) for e in a])

    #a = np.sum(labels, axis=0)
    #print("----------------------------")
    #print([round(e*1.0 / a.sum(),4) for e in a])
    

dataset = "citeseer"
load_data(dataset)

