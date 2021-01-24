import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import models.custom_embeddings as custom_embeddings
import models.transform_layer as transform_layer
import models.linear_layer as linear_layer
from sklearn.preprocessing import normalize
import scipy.sparse as sp
from xclib.data import data_utils as du
import os
import time


class DECAFBase(nn.Module):
    """
        DECAFBase: Base class for DeepXML architecture
        Arguments:
            ----------
            params: model parameters
    """

    def __init__(self, params):
        self._set_params(params)
        super(DECAFBase, self).__init__()
        self.vocabulary_dims = params.vocabulary_dims+1
        self.embedding_dims = params.embedding_dims
        self.num_labels = params.num_labels
        # TODO update this
        self.embed = self._construct_transform(self.trans_config).transform[0]
        self.embed.device = self.device
        # Keep embeddings on first device
        self.device_embeddings = torch.device(self.device)

    def _set_params(self, params):
        self.labelfts_transform = None
        params.trans_config_lbl = None
        transform_config_dict = transform_layer.fetch_json(
            params.trans_method, params)
        self.trans_config = transform_config_dict['transform_coarse']
        self.device = "cuda:0"
        # self.device = "cpu"
        self.fixed_lbs = None
        self.trans_config['tfidf_embed']['sparse'] = False
        self.fixed_fts = params.freeze_embeddings

    def _construct_embedding(self):
        return custom_embeddings.CustomEmbedding(
            num_embeddings=self.vocabulary_dims,
            embedding_dim=self.embedding_dims,
            scale_grad_by_freq=False,
            sparse=self.sparse_emb)

    def _construct_classifier(self):
        pass

    def _construct_transform(self, trans_config):
        if trans_config is not None:
            return transform_layer.Transform(
                transform_layer.get_functions(trans_config),
                device=self.device)
        return None

    @property
    def representation_dims(self):
        return self.embedding_dims

    def encode(self, batch_data, **kwargs):
        """encode documents
        Parameters:
        -----------
        batch_data: dict
            batch_data['X']: torch.Tensor
                dense feature vector or
                feature indices in case of sparse features
            batch_data['X_w]: torch.Tensor
                feature weights in case of sparse features

        Returns
        -------
        out: torch.Tensor
            encoding of a document
        """
        embed = self.embed(batch_data)
        return embed

    def forward(self, batch_data):
        """Forward pass
        Parameters:
        -----------
        batch_data: dict
            batch_data['X']: torch.Tensor
                dense feature vector or
                feature indices in case of sparse features
            batch_data['X_w]: torch.Tensor
                feature weights in case of sparse features
        Returns
        -------
        out: logits for each label
        """
        return self.classifier(self.encode(batch_data))

    def init_model(self, state_dict):
        """Initialize embeddings from existing ones
        Parameters:
        -----------
        word_embeddings: numpy array
            existing embeddings
        """
        pass

    def _emb_init(self, state_dict):
        self.embed._init_(state_dict)

    def to(self):
        """Send layers to respective devices
        """
        self.embed.to()

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def model_size(self):
        return self.num_trainable_params * 4 / math.pow(2, 20)


class DECAFh(DECAFBase):
    """DeepXMLh: Head network for DeepXML architecture
    Allows additional transform layer
    """

    def __init__(self, params):
        super(DECAFh, self).__init__(params)
        self.transform_fine = self._construct_transform(self.trans_config_fine)
        self._sparse_matrix(params)
        self.use_classifier_wts = params.use_classifier_wts
        self.clf_from_fts, self.clf_from_wts = self._construct_classifier(
            params)

    def _load_fts(self, params):
        lbl_wrds = params.lbl_repr_nnz
        if params.clusters is not None:
            print("Merging labels")
            _new_sparse_mat = sp.lil_matrix((len(params.clusters),
                                             lbl_wrds.shape[0]),
                                            dtype=np.float32)
            rows = np.concatenate(list(
                map(lambda x: np.tile(x[0], x[1].size),
                    enumerate(params.clusters))
            ))
            data = np.concatenate(list(
                map(lambda x: np.tile(1.0/min(1, x[1].size), x[1].size),
                    enumerate(params.clusters))
            ))
            _new_sparse_mat[rows, np.concatenate(params.clusters)] = data
            lbl_wrds = normalize(_new_sparse_mat.dot(lbl_wrds), 'l2')
        return lbl_wrds.tocoo()

    def _set_params(self, params):
        super(DECAFh, self)._set_params(params)
        transform_config_dict = transform_layer.fetch_json(
            params.trans_method, params)
        self.trans_config_fine = transform_config_dict['transform_fine']
        self.trans_config_label = transform_config_dict['transform_label']
        self.label_padding_idx = None
        self.sparse_fts = False
        self.sparse_clf = False
        self.fixed_lbs = None
        self.trans_config['tfidf_embed']['sparse'] = False
        self.offset = 0
        self.train_clf = False
        self.prepredict = None
        self.use_classifier_wts = params.use_classifier_wts
        self.fixed_fts = params.freeze_embeddings

    def _construct_classifier(self, params):
        clf0, clf1 = None, None
        if params.label_features is not None:
            clf0 = linear_layer.LabelFts(
                input_size=self.representation_dims,
                label_features=params.label_features,
                sparse=self.sparse_fts and not self.use_classifier_wts,
                use_external_weights=True,
                fixed_features=self.fixed_fts,
                device=self.device,
                transform=self._construct_transform(self.trans_config_label)
            )
        clf1 = linear_layer.CombineUV(
            input_size=self.representation_dims,
            output_size=self.num_labels + self.offset,
            use_classifier_wts=params.use_classifier_wts,
            sparse=self.sparse_clf,
            padding_idx=self.label_padding_idx,
            device=self.device
        )
        assert ((params.label_features is not None) or
                params.use_classifier_wts), "One needs to be true"
        return clf0, clf1

    def _bs_mat(self, lbl_wrds):
        values = torch.FloatTensor(lbl_wrds.data)
        index = torch.LongTensor(
            np.vstack([lbl_wrds.row, lbl_wrds.col])
        )
        mat = torch.sparse_coo_tensor(
            index, values, torch.Size(lbl_wrds.shape)
        )
        mat._coalesced_(True)
        return mat

    def _sparse_matrix(self, params):
        self.lbl_wrds = self._bs_mat(self._load_fts(params))

    def encode(self, batch_data, return_coarse=False):
        if batch_data['is_sparse']:
            encoding = super().encode(batch_data)
        else:
            encoding = batch_data['X'].to(self.device)
        return encoding if return_coarse else self.transform_fine(encoding)

    def forward(self, batch_data):
        input = self.encode(batch_data)
        if self.fixed_fts:
            lbs_repr = self.fixed_lbs
        else:
            lbs_repr = self.lbl_wrds
        label_fts = self.clf_from_fts(lbs_repr, weight=self.embed.weight)
        return self.clf_from_wts(input, label_fts)

    def _predict(self, batch_data):
        input = self.encode(batch_data)
        return self.clf_from_wts(input, None)

    def _init_(self, state_dict_emb, state_dict_clf):
        self._emb_init(state_dict_emb)
        self.clf_from_fts._init_(state_dict_clf)
        self.clf_from_wts._init_(self.lbl_wrds, state_dict_clf)

    def get_weights(self):
        """Get classifier weights
        """
        clf0 = self.clf_from_fts.get_weights()
        clf1 = self.clf_from_wts.get_weights()

        return {'words': clf0, 'classifier': clf1}

    def _pred_init(self):
        if self.fixed_fts:
            lbs_repr = self.fixed_lbs
        else:
            lbs_repr = self.lbl_wrds
        self.clf_from_wts._setup(self.clf_from_fts(lbs_repr,
                                                   weight=self.embed.weight))

    def _eval(self):
        self.eval()
        self._pred_init()

    def _train(self):
        self.clf_from_wts._clean()
        self.train()

    def _get_v(self):
        return self.lbl_wrds.cpu().mm(self.embed.weight.cpu())

    def to(self):
        """Send layers to respective devices
        """
        if self.fixed_fts and self.fixed_lbs is None:
            self.cpu()
        self.transform_fine.to()
        self.clf_from_fts.to()
        if self.clf_from_wts.prepredict is None:
            self.clf_from_wts.to()
            if self.fixed_fts:
                self.fixed_lbs = self._get_v()
                self.fixed_lbs = self.fixed_lbs.to(self.device)
            else:
                self.lbl_wrds = self.lbl_wrds.to(self.device)
        else:
            self.clf_from_wts._pred_to()
        self.embed.to()

    def cpu(self):
        self.embed = self.embed.cpu()
        self.transform_fine = self.transform_fine.cpu()
        self.lbl_wrds = self.lbl_wrds.cpu()
        self.clf_from_wts = self.clf_from_wts.cpu()
        if self.clf_from_wts.prepredict is not None:
            self.clf_from_wts._pred_cpu()
        self.clf_from_fts = self.clf_from_fts.cpu()
        if self.fixed_fts:
            if self.fixed_lbs is None:
                self.fixed_lbs = self._get_v()
            self.fixed_lbs = self.fixed_lbs.cpu()


class DECAFt(DECAFh):
    """DeepXMLh: Head network for DeepXML architecture
    Allows additional transform layer
    """

    def __init__(self, params):
        super(DECAFt, self).__init__(params)

    def _set_params(self, params):
        super(DECAFt, self)._set_params(params)
        self.label_padding_idx = params.num_labels
        self.sparse_fts = False
        self.sparse_clf = True and self.use_classifier_wts
        self.sparse_emb = False
        self.device = "cuda:0"
        self.offset = 0
        # torch.set_num_threads(8)
        if self.label_padding_idx is not None:
            self.offset = 1

    def _sparse_matrix(self, params):
        self.lbl_wrds = self._load_fts(params).tocsr()
        if self.label_padding_idx is not None:
            self.lbl_wrds = sp.vstack([
                self.lbl_wrds,
                sp.csr_matrix((1, self.lbl_wrds.shape[1]),
                              dtype=np.float32)]).tocsr()

    def _just_create_matrix(self, lbl_wrd):
        mat = torch.sparse_coo_tensor(lbl_wrd[0], lbl_wrd[1], lbl_wrd[2])
        mat._coalesced_(True)
        return mat

    def forward(self, batch_data):
        """Forward pass
        Parameters:
        -----------
        batch_data: dict
            batch_data['X']: torch.Tensor
                dense feature vector or
                feature indices in case of sparse features
            batch_data['X_w]: torch.Tensor
                feature weights in case of sparse features
        Returns
        -------
        out: logits for each label
        """
        input = self.encode(batch_data)
        v_lbl_idx = batch_data['v_lbl_idx'].to(self.device)
        Y_lbl_rmp = batch_data['Y_lbl_rmp'].to(self.device)
        if self.fixed_fts:
            lbl_repr = F.embedding(v_lbl_idx, self.fixed_lbs, sparse=False,
                                   padding_idx=None).squeeze()
            label_fts = self.clf_from_fts(lbl_repr, weight=self.embed.weight)
        else:
            v_lbl_fts = batch_data['v_lbl_fts'].to(self.device)
            v_lbl_wrd = self._just_create_matrix(
                batch_data['v_lbl_wrd']).to(self.device)
            label_fts = self.clf_from_fts(
                v_lbl_wrd, v_lbl_fts, self.embed.weight)
        return self.clf_from_wts(input, label_fts, Y_lbl_rmp, v_lbl_idx)

    def _predict(self, batch_data):
        input = self.encode(batch_data)
        return self.clf_from_wts(input, None, batch_data['Y_s'].to(self.device))

    def _get_v(self):
        sp_mat = self._bs_mat(self.lbl_wrds.tocoo()).cpu()
        _weight = self.embed.weight.cpu()
        return sp_mat.mm(_weight)

    def cpu(self):
        self.embed = self.embed.cpu()
        self.transform_fine = self.transform_fine.cpu()
        self.clf_from_wts = self.clf_from_wts.cpu()
        if self.clf_from_wts.prepredict is not None:
            self.clf_from_wts._pred_cpu()
        self.clf_from_fts = self.clf_from_fts.cpu()
        if self.fixed_fts:
            if self.fixed_lbs is None:
                self.fixed_lbs = self._get_v()
            self.fixed_lbs = self.fixed_lbs.cpu()

    def to(self):
        """Send layers to respective devices
        """
        if self.fixed_fts and self.fixed_lbs is None:
            self.cpu()
        self.transform_fine.to()
        self.clf_from_fts.to()
        if self.clf_from_wts.prepredict is None:
            self.clf_from_wts.to()
            if self.fixed_fts:
                self.fixed_lbs = self._get_v()
                self.fixed_lbs = self.fixed_lbs.to(self.device)
        else:
            self.clf_from_wts._pred_to()
        self.embed.to()

    def _pred_init(self):
        if self.fixed_fts:
            lbs_repr = self.fixed_lbs
        else:
            lbs_repr = self._bs_mat(self.lbl_wrds.tocoo())
        self.clf_from_wts._setup(self.clf_from_fts(lbs_repr,
                                                   weight=self.embed.weight))

    def _init_(self, state_dict_emb, state_dict_clf):
        self._emb_init(state_dict_emb)
        self.clf_from_fts._init_(state_dict_clf)
        lbl_repr = self._bs_mat(self.lbl_wrds.tocoo())
        self.clf_from_wts._init_(lbl_repr, state_dict_clf)
    