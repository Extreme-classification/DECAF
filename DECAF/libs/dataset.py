import torch
import _pickle as pickle
import os
import sys
from scipy.sparse import lil_matrix
import numpy as np
import scipy.sparse as sp
from .dataset_base import DatasetBase
from .shortlist_handler import ShortlistHandlerSimple


def construct_dataset(data_dir, fname_features, fname_labels, data=None,
                      model_dir='', mode='train', size_shortlist=-1,
                      normalize_features=True, normalize_labels=True,
                      keep_invalid=False, num_centroids=1,
                      feature_type='sparse', num_clf_partitions=1,
                      feature_indices=None, label_indices=None,
                      shortlist_method='static', shorty=None,
                      classifier_type=None):

    if classifier_type in ["DECAF"]:
        return DatasetDECAF(
            data_dir, fname_features, fname_labels, data, model_dir, mode,
            feature_indices, label_indices, keep_invalid, normalize_features,
            normalize_labels, num_clf_partitions, size_shortlist,
            num_centroids, feature_type, shorty, "DECAF", shortlist_method)
    else:
        raise NotImplementedError(
            "Unknown dataset method: {}!".format(classifier_type))


class DatasetSparse(DatasetBase):
    """Dataset to load and use XML-Datasets with shortlist
    Parameters
    ---------
    data_dir: str
        data files are stored in this directory
    fname_features: str
        feature file (libsvm or pickle)
    fname_labels: str
        labels file (libsvm or pickle)    
    data: dict, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
        Keys: 'X', 'Y'
    model_dir: str, optional, default=''
        Dump data like valid labels here
    mode: str, optional, default='train'
        Mode of the dataset
    feature_indices: np.ndarray or None, optional, default=None
        Train with selected features only
    label_indices: np.ndarray or None, optional, default=None
        Train for selected labels only
    keep_invalid: bool, optional, default=False
        Don't touch data points or labels
    normalize_features: bool, optional, default=True
        Normalize data points to unit norm
    normalize_lables: bool, optional, default=False
        Normalize labels to convert in probabilities
        Useful in-case on non-binary labels
    num_clf_partitions: int, optional, default=1
        Partition classifier in multiple
        Support for multiple GPUs
    num_centroids: int, optional, default=1
        Multiple representations for labels
    feature_type: str, optional, default='sparse'
        sparse or dense features
    shortlist_type: str, optional, default='static'
        type of shortlist (static or dynamic)
    shorty: obj, optional, default=None
        Useful in-case of dynamic shortlist
    label_type: str, optional, default='dense'
        sparse (i.e. with shortlist) or dense (OVA) labels
    shortlist_in_memory: boolean, optional, default=True
        Keep shortlist in memory if True otherwise keep on disk
    """

    def __init__(self, data_dir, fname_features, fname_labels, data=None,
                 model_dir='', mode='train', feature_indices=None,
                 label_indices=None, keep_invalid=False,
                 normalize_features=True, normalize_labels=False,
                 num_clf_partitions=1, size_shortlist=-1, num_centroids=1,
                 feature_type='sparse', shortlist_method='static',
                 shorty=None, label_type='sparse', shortlist_in_memory=True):
        """
            Expects 'libsvm' format with header
            Args:
                data_file: str: File name for the set
        """
        super().__init__(data_dir, fname_features, fname_labels, data,
                         model_dir, mode, feature_indices, label_indices,
                         keep_invalid, normalize_features, normalize_labels,
                         feature_type, label_type=label_type)
        if self.labels is None:
            NotImplementedError(
                "No support for shortlist w/o any label, \
                    consider using dense dataset.")
        self.feature_type = feature_type
        self.num_centroids = num_centroids
        self.num_clf_partitions = num_clf_partitions
        self.shortlist_in_memory = shortlist_in_memory
        self.size_shortlist = size_shortlist
        self.multiple_cent_mapping = None
        self.shortlist_method = shortlist_method
        self.offset = 0
        if self.mode == 'train':
            self._remove_samples_wo_features_and_labels()

        if not keep_invalid:
            # Remove labels w/o any positive instance
            self._process_labels(model_dir)

        if shortlist_method == 'simple':
            self.offset += 1
            self.shortlist = ShortlistHandlerSimple(
                self.num_labels, model_dir, num_clf_partitions,
                mode, size_shortlist, num_centroids,
                shortlist_in_memory, self.multiple_cent_mapping)
        else:
            raise NotImplementedError(
                "Unknown shortlist method: {}!".format(shortlist_method))
        self.label_padding_index = self.num_labels

    def update_shortlist(self, shortlist, dist, fname='tmp', idx=-1):
        """Update label shortlist for each instance
        """
        pass

    def _process_labels(self, model_dir):
        """
            Process labels to handle labels without any training instance;
            Handle multiple centroids if required
        """
        data_obj = {}
        fname = os.path.join(model_dir, 'labels_params.pkl')
        if self.mode == 'train':
            self._process_labels_train(data_obj)
            pickle.dump(data_obj, open(fname, 'wb'))

        else:
            data_obj = pickle.load(open(fname, 'rb'))
            self._process_labels_predict(data_obj)

    def get_shortlist(self, index):
        """
            Get data with shortlist for given data index
        """
        pos_labels, _ = self.labels[index]
        return self.shortlist.get_shortlist(index, pos_labels)

    def __getitem__(self, index):
        """Get features and labels for index
        Arguments
        ---------
        index: int
            data for this index
        Returns
        -------
        features: np.ndarray or tuple
            for dense: np.ndarray
            for sparse: feature indices and their weights
        labels: tuple
            shortlist: label indices in the shortlist
            labels_mask: 1 for relevant; 0 otherwise
            dist: distance (used during prediction only)
        """
        x = self.features[index]
        y = self.get_shortlist(index)
        meta = {'num_labels': self.num_labels+self.offset,
                'true_num_labels': self.num_labels}
        return x, y, meta


class DatasetDECAF(DatasetSparse):
    """Dataset to load and use XML-Datasets with shortlist
    Parameters
    ---------
    data_dir: str
        data files are stored in this directory
    fname_features: str
        feature file (libsvm or pickle)
    fname_labels: str
        labels file (libsvm or pickle)    
    data: dict, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
        Keys: 'X', 'Y'
    model_dir: str, optional, default=''
        Dump data like valid labels here
    mode: str, optional, default='train'
        Mode of the dataset
    feature_indices: np.ndarray or None, optional, default=None
        Train with selected features only
    label_indices: np.ndarray or None, optional, default=None
        Train for selected labels only
    keep_invalid: bool, optional, default=False
        Don't touch data points or labels
    normalize_features: bool, optional, default=True
        Normalize data points to unit norm
    normalize_lables: bool, optional, default=False
        Normalize labels to convert in probabilities
        Useful in-case on non-binary labels
    num_clf_partitions: int, optional, default=1
        Partition classifier in multiple
        Support for multiple GPUs
    num_centroids: int, optional, default=1
        Multiple representations for labels
    feature_type: str, optional, default='sparse'
        sparse or dense features
    shortlist_type: str, optional, default='static'
        type of shortlist (static or dynamic)
    shorty: obj, optional, default=None
        Useful in-case of dynamic shortlist
    label_type: str, optional, default='dense'
        sparse (i.e. with shortlist) or dense (OVA) labels
    shortlist_in_memory: boolean, optional, default=True
        Keep shortlist in memory if True otherwise keep on disk
    """

    def __init__(self, data_dir, fname_features, fname_labels, data=None,
                 model_dir='', mode='train', feature_indices=None,
                 label_indices=None, keep_invalid=False,
                 normalize_features=True, normalize_labels=False,
                 num_clf_partitions=1, size_shortlist=-1, num_centroids=1,
                 feature_type='sparse', shorty=None, label_type='Parabel',
                 shortlist_in_memory=True):
        super().__init__(data_dir, fname_features, fname_labels, data,
                         model_dir, mode, feature_indices, label_indices,
                         keep_invalid, normalize_features, normalize_labels,
                         num_clf_partitions, size_shortlist, num_centroids,
                         feature_type, "simple", shorty,
                         label_type, shortlist_in_memory)
        self.place_holder = np.ones((1, 1), dtype=np.int32)
        self._mode = self.mode

    def _prep_for_depth(self, depth, clusters):
        self.depth = depth
        self.labels._prep_for_clusters(clusters)
        self.labels.depth = depth
        self.shortlist.setup(self.labels.num_labels)

    def __getitem__(self, index):
        """Get features and labels for index
        Arguments
            ---------
            index: int
                data for this index
        Returns
            -------
            features: np.ndarray or tuple
                for dense: np.ndarray
                for sparse: feature indices and their weights
            labels: tuple
                shortlist: label indices in the shortlist
                labels_mask: 1 for relevant; 0 otherwise
                dist: distance (used during prediction only)
        """
        x = self.features[index]
        if self.shortlist.shortlist is None:
            if self.mode == 'test':
                return x, self.place_holder
            return x, self.labels[index]
        else:
            return x, self.get_shortlist(index)

    def build_shortlist(self, net, shorty, batch=255):
        if shorty is not None:
            _, _, n_lbs = self.get_stats()
            n_int, o_lbs = shorty.shape
            mf = net.C[self.depth-1]
            Xs = np.arange(mf)
            rows = np.concatenate(
                list(map(lambda x: [x]*mf, np.arange(o_lbs))
                     ))
            cols = np.concatenate(
                list(map(lambda x: Xs + x*mf, np.arange(o_lbs))
                     ))
            if self.depth == net.height - 1:
                cols = net.leaf_hash[cols]
            to_beam = lil_matrix((o_lbs, n_lbs + 1))
            to_beam[rows, cols] = 1
            to_beam[:, -1] = 0
            sub_mats = []
            k = 0
            for start in np.arange(0, n_int, batch):
                end = min(n_int, start + batch)
                sub_mats.append(shorty[start:end].dot(to_beam))
                if k % 500 == 0:
                    print("Shortlist to beam [%d/%d]" % (start, n_int))
                k += 1
            shorty = sp.vstack(sub_mats).tocsr()
            self.shortlist.update_shortlist(shorty)
            del to_beam
