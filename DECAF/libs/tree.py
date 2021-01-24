import numpy as np
import time
import scipy.sparse as sp
from sklearn.preprocessing import normalize as scale
from functools import partial
import operator
import functools
from multiprocessing import Pool
import _pickle as cPickle


def _normalize(X, norm='l2'):
    X = scale(X, norm='l2')
    return X


def b_kmeans_dense(labels_features, index, metric='cosine', tol=1e-4, leakage=None):
    labels_features = _normalize(labels_features)
    n = labels_features.shape[0]
    if labels_features.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(
            low=0, high=labels_features.shape[0], size=(2))
    _centeroids = labels_features[cluster]
    _similarity = np.dot(labels_features, _centeroids.T)
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        clustered_lbs = np.array_split(
            np.argsort(_similarity[:, 1]-_similarity[:, 0]), 2)
        _centeroids = _normalize(np.vstack([
            np.mean(labels_features[x, :], axis=0) for x in clustered_lbs
        ]))
        _similarity = np.dot(labels_features, _centeroids.T)
        old_sim, new_sim = new_sim, np.sum(
            [np.sum(
                _similarity[indx, i]
            ) for i, indx in enumerate(clustered_lbs)])/n

    return list(map(lambda x: index[x], clustered_lbs))


def b_kmeans_sparse(labels_features, index, metric='cosine', tol=1e-4, leakage=None):
    if labels_features.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(
            low=0, high=labels_features.shape[0], size=(2))
    _centeroids = labels_features[cluster].todense()
    _similarity = _sdist(labels_features, _centeroids,
                         metric=metric, norm='l2')
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        clustered_lbs = np.array_split(
            np.argsort(_similarity[:, 1]-_similarity[:, 0]), 2)
        _centeroids = np.vstack([
            labels_features[x, :].mean(
                axis=0) for x in clustered_lbs
        ])
        _similarity = _sdist(labels_features, _centeroids,
                             metric=metric, norm='l2')
        old_sim, new_sim = new_sim, np.sum(
            [np.sum(
                _similarity[indx, i]
            ) for i, indx in enumerate(clustered_lbs)])

    if leakage is not None:
        _distance = 1-_similarity
        # Upper boundary under which labels will co-exists
        ex_r = [(1+leakage)*np.max(_distance[indx, i])
                for i, indx in enumerate(clustered_lbs)]
        """
        Check for labels in 2nd cluster who are ex_r_0 closer to 
        1st Cluster and append them in first cluster
        """
        clustered_lbs = list(
            map(lambda x: np.concatenate(
                [clustered_lbs[x[0]],
                 x[1][_distance[x[1], x[0]] <= ex_r[x[0]]]
                 ]),
                enumerate(clustered_lbs[::-1])
                )
        )
    return list(map(lambda x: index[x], clustered_lbs))


def _sdist(XA, XB, metric, norm=None):
    if norm is not None:
        XA = _normalize(XA, norm)
        XB = _normalize(XB, norm)
    if metric == 'cosine':
        score = XA.dot(XB.transpose())
    return score


def cluster_labels(labels, clusters, num_nodes, splitter):
    start = time.time()
    min_splits = min(8, num_nodes)
    while len(clusters) < min_splits:
        temp_cluster_list = functools.reduce(
            operator.iconcat,
            map(lambda x: splitter(labels[x], x),
                clusters), [])
        end = time.time()
        print("Total clusters {}".format(len(temp_cluster_list)),
              "Avg. Cluster size {}".format(
            np.mean(list(map(len, temp_cluster_list)))),
            "Total time {} sec".format(end-start))
        clusters = temp_cluster_list
        del temp_cluster_list
    with Pool(5) as p:
        while len(clusters) < num_nodes:
            temp_cluster_list = functools.reduce(
                operator.iconcat,
                p.starmap(
                    splitter,
                    map(lambda cluster: (labels[cluster], cluster),
                        clusters)
                ), [])
            end = time.time()
            print("Total clusters {}".format(len(temp_cluster_list)),
                  "Avg. Cluster size {}".format(
                      np.mean(list(map(len, temp_cluster_list)))),
                  "Total time {} sec".format(end-start))
            clusters = temp_cluster_list
            del temp_cluster_list
    return clusters


class hash_map_index:
    def __init__(self, clusters, label_to_idx, total_elements, total_valid_nodes, padding_idx=None):
        self.clusters = clusters
        self.padding_idx = padding_idx
        self.total_elements = total_elements
        self.size = total_valid_nodes
        self.weights = None
        if padding_idx is not None:
            self.weights = np.zeros((self.total_elements), np.float32)
            self.weights[label_to_idx == padding_idx] = -np.inf

        self.hash_map = label_to_idx

    def _get_hash(self):
        return self.hash_map

    def _get_weights(self):
        return self.weights


class build_tree:
    """
        build_tree class object for clustering labels
        Arguments
            ----------
            b_factors: list(int)
                np.log2 of number of nodes for each level
            M: int
                max leaf size
            method: string
                type of clustering strategy to use
                default=parabel
            force_shallow: bool 
                if true then only 1 level in used
                default=True              

    """
    def __init__(self, b_factors=[2], M=1, method='parabel',
                 force_shallow=True):
        self.b_factors = b_factors
        self.C = []
        self.method = method
        self.force_shallow = force_shallow
        self.height = 2

    def fit(self, lbl_repr, lbl_co=None):
        lbl_repr = _normalize(lbl_repr)
        self.num_labels = lbl_repr.shape[0]
        clusters = [np.asarray(np.arange(self.num_labels), dtype=np.int32)]
        self.hash_map_array = []
        if isinstance(lbl_repr, np.ndarray):
            print("Using dense kmeans++")
            b_kmeans = b_kmeans_dense
        else:
            b_kmeans = b_kmeans_sparse
        if self.method == 'leaky-Parabel':
            print("Using leaky-Parabel, upto d=4")
            upto = 2**(1)
            clusters = cluster_labels(lbl_repr, clusters, upto,
                                      partial(b_kmeans, leakage=0.0001))
        elif self.method == "NoCluster":
            self.height = 1
            print("No need to create splits")
            n_lb = self.num_labels
            self.hash_map_array.append(
                hash_map_index(None, clusters[0], n_lb, n_lb, n_lb))
            return

        self._parabel(lbl_repr, clusters, b_kmeans, self.force_shallow)

    def _parabel(self, labels, clusters, splitter=None, force_shallow=True):
        depth = 0
        while True:
            num_child_nodes = 2**self.b_factors[depth]
            print("Building tree at height %d with nodes: %d" %
                  (depth, num_child_nodes))
            if num_child_nodes >= self.num_labels:
                print("No need to train parabel")
                clusters = list(np.arange(self.num_labels).reshape(-1, 1))

            clusters = cluster_labels(labels,
                                      clusters,
                                      num_child_nodes,
                                      splitter)

            self.hash_map_array.append(
                hash_map_index(
                    clusters,
                    np.arange(num_child_nodes),
                    num_child_nodes,
                    num_child_nodes
                )
            )
            depth += 1
            if depth == len(self.b_factors):
                _M = max(list(map(lambda x: x.size, clusters)))
                self.C.append(_M)
                print("Preparing Leaf")
                break
            else:
                self.C.append(
                    2**(self.b_factors[depth] - self.b_factors[depth-1]))

        self.height = depth+1
        self.max_node_idx = num_child_nodes*self.C[-1]
        print("Building tree at height %d with max leafs: %d" %
              (self.height, self.max_node_idx))
        _labels_path_array = np.full(
            (self.max_node_idx), self.num_labels,
            dtype=np.int32)
        for idx, c in enumerate(clusters):
            index = np.arange(c.size) + idx*self.C[-1]
            _labels_path_array[index] = clusters[idx]
        self.hash_map_array.append(
            hash_map_index(None,
                           _labels_path_array,
                           self.max_node_idx,
                           self.num_labels,
                           self.num_labels))
        print("Sparsity of leaf is %0.2f" %
              ((1-(self.num_labels/self.max_node_idx))*100))

    def _get_cluster_depth(self, depth):
        return self.hash_map_array[depth].clusters

    def save(self, fname):
        cPickle.dump(self.__dict__, open(fname, 'wb'))

    def load(self, fname):
        self.__dict__ = cPickle.load(open(fname, 'rb'))
