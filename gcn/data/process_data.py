from collections import defaultdict as dd
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import random

labels = {"Case_Based": 5,
          "Genetic_Algorithms": 2,
          "Neural_Networks": 3,
          "Probabilistic_Methods": 4,
          "Reinforcement_Learning": 1,
          "Rule_Learning": 6,
          "Theory": 0}


def one_hot_encoding(size, index):
    y = [0] * size
    y[index] = 1
    return y


def add_index(index, cnt, key):
    """
    feed in current cnt, which has not been used
    if key exists, then dont use it and return current cnt
    otherwise, use the current cnt, and add 1
    """
    if key in index: return cnt
    index[key] = cnt
    return cnt + 1


def process_data(filename, labels):
    allx, ally, tx, ty, x, y = [], [], [], [], [], []
    index, cnt = {}, 0
    _x = {_: [] for _ in labels}
    _y = {_: 0 for _ in labels}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:500]):
            inputs = line.strip().split()
            cnt = add_index(index, cnt, inputs[0])
            features = [float(_) for _ in inputs[1:-1]]
            one_hot_label = one_hot_encoding(len(labels), labels[inputs[-1]])
            allx.append(features)
            ally.append(one_hot_label)
            if i < 140:
                x.append(features)
                y.append(one_hot_label)
            _y[inputs[-1]] += 1
            _x[inputs[-1]].append(i)
        # for key in _x:
        #    idx = random.sample(_x[key],20)
        #    x += [allx[i] for i in idx]
        #    y += [ally[i] for i in idx]
        for i, line in enumerate(lines[500:1500]):
            inputs = line.strip().split()
            cnt = add_index(index, cnt, inputs[0])
            tx.append([float(_) for _ in inputs[1:-1]])
            ty.append(one_hot_encoding(len(labels), labels[inputs[-1]]))
            _y[inputs[-1]] += 1

        for i, line in enumerate(lines[1500:]):
            inputs = line.strip().split()
            cnt = add_index(index, cnt, inputs[0])
            features = [float(_) for _ in inputs[1:-1]]
            one_hot_label = one_hot_encoding(len(labels), labels[inputs[-1]])
            allx.append(features)
            ally.append(one_hot_label)
            _y[inputs[-1]] += 1
            _x[inputs[-1]].append(i)
    pkl.dump(sp.csr_matrix(np.array(x)), open("ind.cora.x", "wb"))
    pkl.dump(sp.csr_matrix(np.array(tx)), open("ind.cora.tx", "wb"))
    pkl.dump(sp.csr_matrix(np.array(allx)), open("ind.cora.allx", "wb"))
    pkl.dump(np.array(y), open("ind.cora.y", "wb"))
    pkl.dump(np.array(ty), open("ind.cora.ty", "wb"))
    pkl.dump(np.array(ally), open("ind.cora.ally", "wb"))
    return index




def read_cites(filename, index):
    cites, parents, graph = [], dd(list), dd(list)
    for i, line in enumerate(open(filename)):
        inputs = line.strip().split()
        cites.append((inputs[0], inputs[1]))
    for id1, id2 in cites:
        # cnt = add_index(index, cnt, id1)
        # cnt = add_index(index, cnt, id2)
        i, j = index[id1], index[id2]
        parents[j].append(i)
        graph[i].append(j)
        graph[j].append(i)
    pkl.dump(parents, open("ind.cora.parents", "wb"))
    pkl.dump(graph, open("ind.cora.graph", "wb"))
    return parents, graph


# index = process_data("cora.content",labels)
# read_cites("cora.cites",index)
# f = open("ind.cora.test.index","w")
# a = list(range(500,1500))
# for i in a:
#    f.write(str(i)+"\n")
# f.close()

read_content("cora")
