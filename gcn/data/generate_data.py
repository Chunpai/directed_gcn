from collections import defaultdict as dd
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import random


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def generate(dataset):
    """read *.content file """
    print("Loading {}/{}.content file...".format(dataset, dataset))
    idx_feature_labels = np.genfromtxt("{}/{}.content".format(dataset, dataset),
                                       dtype=np.dtype(str))
    idx = np.array(idx_feature_labels[:,0], dtype=np.dtype(str))
    # features = idx_feature_labels[:, 1:-1]
    features = sp.csr_matrix(idx_feature_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_feature_labels[:,-1])

    # build undirected graph and directed graph
    # map paper_idx to new_idx which is range(0,num_of_nodes)
    print("Loading {}/{}.cites file...".format(dataset, dataset))
    idx_map = {j:i for i, j in enumerate(idx)}
    cites = np.genfromtxt("{}/{}.cites".format(dataset, dataset),
                          dtype=np.dtype(str))
    for c1,c2 in cites:
        if c1 not in idx_map:
            print c1
        if c2 not in idx_map:
            print c2
            
    edges = np.array(list(map(idx_map.get, cites.flatten())),dtype=np.int32).reshape(cites.shape)

    undirected_graph, directed_graph = dd(list), dd(list)
    for id1, id2 in edges:
        directed_graph[id2].append(id1)
        undirected_graph[id1].append(id2)
        undirected_graph[id2].append(id1)
    objects = [features,labels,directed_graph,undirected_graph]
    names = ["features","labels","directed.graph","undirected.graph"]
    for i in range(len(names)):
        with open("{}/{}.{}".format(dataset, dataset, names[i]),"wb") as f:
            pkl.dump(objects[i], f)

    size = labels.shape[0]
    for i in range(10):
        train_index, val_index, test_index = random_splits(size)
        print("train",train_index)
        print("val",val_index)
        print("test",test_index)
        # use pickle for data serialization
        objects = [train_index, val_index, test_index]
        names = ["train.index", "val.index", "test.index"]
        for j in range(len(names)):
            with open("{}/{}/{}.{}".format(dataset, i, dataset, names[j]),"wb") as f:
                pkl.dump(objects[j], f)


def random_splits(size):
    order = range(size)
    random.shuffle(order)
    train_index = order[:140]
    val_index = order[140:640]
    test_index = order[-1000:]
    return train_index, val_index, test_index



if __name__ == "__main__":
    dataset = "citeseer"
    generate(dataset)
