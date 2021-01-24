import numpy as np
import _pickle as pickle
import operator
import os
from scipy.sparse import load_npz, hstack, csr_matrix
from xclib.utils import sparse as sp
import pdb


class ShortlistHandlerBase(object):
    """
        Base class for ShortlistHandler
        Parameters
            ----------
            num_labels: int
                number of labels
            shortlist:
                shortlist object
            model_dir: str, optional, default=''
                save the data in model_dir
            num_clf_partitions: int, optional, default=''
                #classifier splits
            mode: str: optional, default=''
                mode i.e. train or test or val
            size_shortlist:int, optional, default=-1
                get shortlist of this size
            num_centroids: int, optional, default=1
                #centroids (useful when using multiple rep)
            label_mapping: None or dict: optional, default=None
                map labels as per this mapping
    """

    def __init__(self, num_labels, shortlist, model_dir='',
                 num_clf_partitions=1, mode='train',
                 size_shortlist=-1, num_centroids=1,
                 label_mapping=None):
        self.model_dir = model_dir
        self.num_centroids = num_centroids
        self.num_clf_partitions = num_clf_partitions
        self.size_shortlist = size_shortlist
        self.mode = mode
        self.num_labels = num_labels
        self.label_mapping = label_mapping
        self._create_partitioner()
        self.label_padding_index = self.num_labels

    def _create_shortlist(self, shortlist):
        """
            Create structure to hold shortlist
        """
        self.shortlist = shortlist

    def query(self, *args, **kwargs):
        return self.shortlist(*args, **kwargs)

    def _create_partitioner(self):
        """
            Create partiotionar to for splitted classifier
        """
        pass

    def _pad_seq(self, indices, dist):
        _pad_length = self.size_shortlist - len(indices)
        indices.extend([self.label_padding_index]*_pad_length)
        dist.extend([100]*_pad_length)

    def _remap_multiple_representations(self, indices, vals,
                                        _func=min, _limit=1e5):
        """
            Remap multiple centroids to original labels
        """
        pass

    def _adjust_shortlist(self, pos_labels, shortlist, dist, min_nneg=100):
        """
            Adjust shortlist for a instance
            Training: Add positive labels to the shortlist
            Inference: Return shortlist with label mask
        """
        if self.mode == 'train':
            if len(pos_labels) > self.size_shortlist:
                _ind = np.random.choice(
                    len(pos_labels), size=self.size_shortlist-min_nneg, replace=False)
                pos_labels = list(operator.itemgetter(*_ind)(pos_labels))
            neg_labels = list(
                filter(lambda x: x not in set(pos_labels), shortlist))
            diff = self.size_shortlist - len(pos_labels)
            labels_mask = [1]*len(pos_labels)
            dist = [2]*len(pos_labels) + dist[:diff]
            shortlist = pos_labels + neg_labels[:diff]
            labels_mask = labels_mask + [0]*diff
        else:
            labels_mask = [0]*self.size_shortlist
            pos_labels = set(pos_labels)
            for idx, item in enumerate(shortlist):
                if item in pos_labels:
                    labels_mask[idx] = 1
        return shortlist, labels_mask, dist

    def _get_sl_one(self, index, pos_labels):
        if self.shortlist.data_init:
            shortlist, dist = self.query(index)
            shortlist, labels_mask, dist = self._adjust_shortlist(
                pos_labels, shortlist, dist)
        else:
            shortlist = [0]*self.size_shortlist
            labels_mask = [0]*self.size_shortlist
            dist = [0]*self.size_shortlist
        return shortlist, labels_mask, dist

    def get_shortlist(self, index, pos_labels=None):
        """
            Get data with shortlist for given data index
        """
        return self._get_sl_one(index, pos_labels)


class ShortlistHandlerSimple(ShortlistHandlerBase):
    """ShortlistHandler with static shortlist
    - save/load/update/process shortlist
    - support for partitioned classifier
    - support for multiple representations for labels
    Parameters
    ----------
    num_labels: int
        number of labels
    model_dir: str, optional, default=''
        save the data in model_dir
    num_clf_partitions: int, optional, default=''
        #classifier splits
    mode: str: optional, default=''
        mode i.e. train or test or val
    size_shortlist:int, optional, default=-1
        get shortlist of this size
    num_centroids: int, optional, default=1
        #centroids (useful when using multiple rep)
    in_memory: bool: optional, default=True
        Keep the shortlist in memory or on-disk
    label_mapping: None or dict: optional, default=None
        map labels as per this mapping
    """

    def __init__(self, num_labels, model_dir='', num_clf_partitions=1,
                 mode='train', size_shortlist=-1, num_centroids=1,
                 in_memory=True, label_mapping=None):
        super().__init__(num_labels, None, model_dir, num_clf_partitions,
                         mode, size_shortlist, num_centroids, label_mapping)
        self.in_memory = in_memory
        self._create_shortlist()
        self.num_splits = None

    def query(self, index, split_id=-1):
        if split_id > -1:
            shorty = self.shortlist[split_id][index]
        else:
            shorty = self.shortlist[index]
        return shorty.indices, shorty.data

    def _get_sl_one(self, index, pos_labels, s_idx=-1):
        if self.shortlist is not None:
            shortlist, dist = self.query(index)
            labels_mask = np.zeros(shortlist.size)
            labels_mask[np.isin(shortlist, pos_labels)] = 1.0
        else:
            shortlist = [0]*self.size_shortlist
            labels_mask = [0]*self.size_shortlist
            dist = [0]*self.size_shortlist
        return shortlist, labels_mask, dist

    def _create_shortlist(self):
        """
            Create structure to hold shortlist
        """
        self.shortlist = None
        self.splits = None

    def update_shortlist(self, shortlist, splits=None, idx=-1):
        """
            Update label shortlist for each instance
        """
        if shortlist is not None:
            self.shortlist = shortlist
            n_inst, _ = shortlist.shape
            print("Average shortlist size", shortlist.nnz/n_inst)

    def setup(self, *args):
        pass

    def _post_process(self, splits=None):
        self.update_shortlist(self.shortlist, splits)
