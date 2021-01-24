import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import models.transform_layer as transform_layer
from libs.tree import build_tree as Tree
from .network import *
import pickle
import math
import os
import re


class DECAFLBase(nn.Module):
    """
        DECAF's wrapper model for all its variations
        Arguments
            ----------
            params: model parameters
    """
    def __init__(self, params):
        super(DECAFLBase, self).__init__()
        self.params = params
        self.fetch_labels(params)
        self.Xs = None
        if params.model_method == "parabel":
            self.tree = Tree(b_factors=params.b_factors,
                             method=params.cluster_method,
                             force_shallow=params.force_shallow)
        else:
            NotImplementedError("Unkown ANNS method")
        self.m_size = 0
        self.depth = -1
        self.freeze_embeddings = params.freeze_embeddings
        self.depth_node = nn.Sequential()

    def fetch_labels(self, params):
        """
            Loads label features
        """
        valid_labels = np.loadtxt(params.label_indices, dtype=np.int32)
        label_features = np.loadtxt(params.v_lbl_fts, dtype=np.int32)

        label_words = os.path.join(
            params.data_dir, params.dataset, params.label_words
        )

        label_words = du.read_sparse_file(label_words)[valid_labels]
        label_words = label_words.tocsc()[:, label_features].tolil()
        label_words = sp.hstack([label_words,
                                 sp.lil_matrix((valid_labels.size, 1))])
        self.lbl_repr_nnz = normalize(label_words.tocsr(), norm='l2').tolil()

    def build(self, lbl_repr, model_dir):
        if not os.path.exists(model_dir+"/clusters.pkl"):
            start = time.time()
            self.tree.fit(lbl_repr)
            print("ANNS trian time: {} sec".format(time.time() - start))
            self.save(model_dir)
        else:
            self.load(model_dir)
        self._setup()

    def _setup(self):
        self.C = self.tree.C
        self.height = self.tree.height
        self.leaf_hash = self.tree.hash_map_array[-1]._get_hash()
        self.leaf_weit = self.tree.hash_map_array[-1]._get_weights()

    def _layer(self, model, params):
        if model == "DECAFh":
            return DECAFh(params)
        elif model == "DECAFt":
            return DECAFt(params)
        elif model == "Base":
            return DECAFBase(params)
        else:
            print("{}:Kuch bhi".format(self.__class__.__name__))

    def _add_layer(self, depth):
        params = self.params
        hash_map_obj = self.tree.hash_map_array[depth]
        params.num_labels = hash_map_obj.size
        params.clusters = hash_map_obj.clusters
        params.label_padding_index = hash_map_obj.padding_idx
        params.lbl_repr_nnz = self.lbl_repr_nnz
        self.depth_node = self._layer(params.layers[depth], params)
        for params in self.depth_node.parameters():
            self.m_size += params.numel()

    def scores(self, batch, depth=0):
        return self.depth_node(batch)

    def forward(self, batch, depth=None):
        if depth is None:
            depth = self.depth
        scores = self.scores(batch, depth=depth)
        return scores

    def _predict(self, batch):
        return F.logsigmoid(self.depth_node._predict(batch))

    def _clusters(self, depth=None):
        return self.tree._get_cluster_depth(depth)

    def _prep_for_depth(self, depth, initialize=True, disable_grad=True):
        if initialize:
            self._initialize_from_next_layer(depth)
        self._add_layer(depth)
        if initialize:
            self.depth_node._init_(self.word_wts, self.word_wts)
        self.depth = depth
        self._call_back()

    def _build_predict(self, dataloader, depth):
        pass

    def eval(self, depth=None):
        self.depth_node._eval()

    def train(self):
        self.depth_node.to()
        self.depth_node._train()
        self._call_back()

    def _init_emb(self, weights):
        self.word_wts = weights

    def _init_clf(self, weights):
        pass

    def to(self):
        self.depth_node.to()

    def cpu(self):
        self.depth_node.cpu()

    def _call_back(self):
        pass

    def _initialize_from_next_layer(self, depth):
        pass

    @property
    def label_words(self):
        if self.params.layers[self.depth] in ["DECAFt", "DECAFpt"]:
            return self.depth_node.lbl_wrds
        return None

    def get_only_encoder(self):
        model = self._layer("Base", params=self.params)
        model._emb_init(self.word_wts)
        return model

    @property
    def model_size(self):
        return self.m_size*4/np.power(2, 20)

    def get_weights(self):
        clf_label_fts = {}
        for depth in range(self.height):
            clf_label_fts[str(depth)] = self[0].get_weights()
        return clf_label_fts

    def save(self, fname):
        fname = fname + "/clusters.pkl"
        self.tree.save(fname)

    def load(self, fname):
        fname = fname + "/clusters.pkl"
        self.tree.load(fname)

    @property
    def offset(self):
        return self.depth_node.offset


class DECAFpp(DECAFLBase):
    """
        DECAF's wrapper model for all its variations
        Arguments
            ----------
            params: model parameters
    """
    def __init__(self, params):
        super(DECAFpp, self).__init__(params)

    def _initialize_from_next_layer(self, depth):
        if depth > 0:
            print("INFO::DECAF::Setting from top layer")
            self.word_wts = self.depth_node.embed.cpu().state_dict()


class DECAFfe(DECAFLBase):
    """
        DECAF-lite's wrapper model for all its variations
        Arguments
            ----------
            params: model parameters
    """
    def __init__(self, params):
        # params.label_residual = True
        super(DECAFfe, self).__init__(params)

    def _call_back(self):
        if self.freeze_embeddings:
            for params in self.depth_node.embed.parameters():
                params.requires_grad = False

    @property
    def model_size(self):
        extra_params = (self.depth+1) * \
            np.sum([x.numel() for x in self.depth_node.embed.parameters()])
        return (self.m_size - extra_params)*4/np.power(2, 20)
