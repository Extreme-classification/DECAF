import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import models.transform_layer as transform_layer
import math


class LabelFts(nn.Module):
    """
        Class for encoding label features
        Arguments
            ----------
            input_size: int
                embeddings dimension for the label representation
            label_features: int
                number of token in the label text
            padding_idx: int (default=None)
            device: string (default="cuda:0")
            sparse: bool (default=False)
            fixed_features: bool (default=False)
            use_external_weights: bool (default=False)
            transform: nn.Module
                transformation over label features
        Returns:
            nn.Module
                Network block to encode label features
    """
    def __init__(self, input_size, label_features, device="cuda:0",
                 sparse=False, fixed_features=False,
                 use_external_weights=False, transform=None):
        super(LabelFts, self).__init__()
        self.device = device  # Useful in case of multiple GPUs
        self.input_size = input_size
        self.label_features = label_features
        self.use_external_weights = use_external_weights
        self.fixed_features = fixed_features
        if not self.use_external_weights:
            self.weight = Parameter(torch.Tensor(self.label_features,
                                                 self.input_size))
            self.sparse = True
        else:
            self.sparse = sparse
        self.Rpp = transform
        self.reset_parameters()

    def _get_clf(self, labels, features_shortlist=None,
                 weights=None):
        if features_shortlist is not None:
            weights = F.embedding(features_shortlist,
                                  weights,
                                  sparse=self.sparse,
                                  padding_idx=None).squeeze()
        lbl_clf = labels.to(weights.device).mm(weights)
        return lbl_clf

    def forward(self, labels, features_shortlist=None, weight=None):
        if self.fixed_features:
            return self.Rpp(labels)
        else:
            if not self.use_external_weights:
                weight = self.weight
            return self.Rpp(self._get_clf(
                labels, features_shortlist, weight
            ))

    def to(self, device=None):
        if device is None:
            super().to(self.device)
        else:
            super().to(device)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_size)
        if not self.use_external_weights:
            self.weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = "{name}({input_size}, {label_features}, {sparse}, {device}, {fixed_features}"
        if not self.use_external_weights:
            s += ", weight={}".format(self.weight.detach().cpu().numpy().shape)
        s += "\n%s" % (self.Rpp.__repr__())
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)
    
    def _init_(self, state_dict):
        """
            Initilizes model parameters
        """
        if not self.use_external_weights:
            keys = list(state_dict.keys())
            key = [ x for x in keys if x.split(".")[-1] in ['weight']][0]
            weight = state_dict[key]
            self.weight.data.copy_(weight)


class CombineUV(nn.Module):
    """
        Combines all classifier components for DECAF
        Arguments
            ----------
            input_size: int
                Classifier's embeddinging dimension
            output_size: int
                Number of classifiers to train
            bias: bool (default=True)
                Flag to use bias
            use_classifier_wts: bool (default=False)
                If True learns a refinement vector
            padding_idx: int (default=None)
                Label padding index
            device: string (default="cuda:0")
                GPU id to keep the parameters
            sparse: bool (default=False)
                Type of optimizer to use for training
    """

    def __init__(self, input_size, output_size, bias=True,
                 use_classifier_wts=False, padding_idx=None,
                 device="cuda:0", sparse=False):
        super(CombineUV, self).__init__()
        self._device = device  # Useful in case of multiple GPUs
        self.input_size = input_size
        self.output_size = output_size
        self.use_classifier_wts = use_classifier_wts
        self.sparse = sparse
        if self.sparse:
            self.device = "cpu"
        self.padding_idx = padding_idx

        if self.use_classifier_wts:
            if bias:
                if self.sparse:
                    self.bias = Parameter(torch.Tensor(self.output_size, 1))
                else:
                    self.bias = Parameter(torch.Tensor(self.output_size))
            else:
                self.register_parameter('bias', None)
            self.weight = Parameter(torch.Tensor(self.output_size,
                                                 self.input_size))
            self.alpha = Parameter(torch.Tensor(1, self.input_size))
            self.beta = Parameter(torch.Tensor(1, self.input_size))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)
        self.prebias = None
        self.prepredict = None
        self.reset_parameters()

    def _get_clf(self, weights, shortlist=None, sparse=True,
                 padding_idx=None):
        if shortlist is not None and weights is not None:
            return F.embedding(shortlist, weights, sparse=sparse,
                               padding_idx=padding_idx, max_norm=None,
                               norm_type=2., scale_grad_by_freq=False)
        return weights

    def _get_rebuild(self, lbl_clf, shortlist=None):
        if self.prepredict is None:
            bias = self._get_clf(self.bias, shortlist,
                                 self.sparse, self.padding_idx)
            lbl_clf = lbl_clf.to(self.device)
            _lbl_clf = self._get_clf(self.weight, shortlist,
                                     self.sparse, self.padding_idx)
            if self.use_classifier_wts:
                lbl_clf = self.alpha.sigmoid()*_lbl_clf.squeeze() + \
                    self.beta.sigmoid()*lbl_clf
            return lbl_clf, bias
        else:
            return self.prepredict, self.prebias

    def forward(self, input, labels, shortlist=None, shortlist_first=None):
        input = input.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if shortlist_first is not None:
            shortlist_first = shortlist_first.to(self.device)
        lbl_clf, bias = self._get_rebuild(labels, shortlist_first)
        if shortlist is not None:
            shortlist = shortlist.to(self.device)
            input = input.to(self.device)
            lbl_clf = self._get_clf(lbl_clf, shortlist, sparse=False)
            bias = self._get_clf(bias, shortlist, sparse=False)
            out = torch.matmul(input.unsqueeze(1), lbl_clf.permute(0, 2, 1))
            if bias is not None:
                out = out.squeeze() + bias.squeeze()
            return out.squeeze()
        else:
            return F.linear(input, lbl_clf, bias)

    def _setup(self, lbl_clf):
        self.device = "cpu"
        if self.use_classifier_wts:
            norm_u = np.mean(torch.norm(self.weight, dim=1).detach().cpu().numpy())
            print("Alpha=", torch.mean(self.alpha.detach().sigmoid()).cpu().numpy(),
                  "Beta=", torch.mean(self.beta.detach().sigmoid()).cpu().numpy())
            norm_v = np.mean(torch.norm(lbl_clf, dim=1).detach().cpu().numpy())
            lbl_clf, _bias = self._get_rebuild(lbl_clf)
            norm_w = np.mean(torch.norm(lbl_clf, dim=1).detach().cpu().numpy())
            print("||u||=%0.2f, ||v||=%0.2f, ||w||=%0.2f"%(norm_u, norm_v, norm_w))
            self.prebias = _bias+0
        self.prepredict = lbl_clf

    def _pred_to(self):
        self.device = self._device
        self.prepredict = self.prepredict.to(self.device)
        if self.use_classifier_wts:
            self.prebias = self.prebias.to(self.device)

    def _pred_cpu(self):
        self.device = "cpu"
        self.prepredict = self.prepredict.cpu()
        if self.use_classifier_wts:
            self.prebias = self.prebias.cpu()

    def _clean(self):
        if self.prepredict is None:
            print("No need to clean")
            return
        print("Cleaing for GPU")
        self.prepredict = self.prepredict.cpu()
        if self.use_classifier_wts:
            self.prebias = self.prebias.cpu()
        self.prebias, self.prepredict = None, None

    def to(self):
        """
            Transfer to device
        """
        self.device = self._device
        super().to(self._device)

    def reset_parameters(self):
        """
            Initialize vectors
        """
        stdv = 1. / math.sqrt(self.input_size)

        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
            self.alpha.data.fill_(0)
            self.beta.data.fill_(0)

        if self.bias is not None:
            self.bias.data.fill_(0)

    def get_weights(self):
        """
            Get weights as dictionary
        """
        parameters = {}
        if self.bias is not None:
            parameters['bias'] = self.bias.detach().cpu().numpy()
        if self.use_classifier_wts:
            parameters['weight'] = self.weight.detach().cpu().numpy()
            parameters['alpha'] = self.alpha.detach().cpu().numpy()
            parameters['beta'] = self.beta.detach().cpu().numpy()
        return parameters

    def __repr__(self):
        s = "{name}({input_size}, {output_size}, {_device}, {use_classifier_wts}, sparse={sparse}"
        if self.use_classifier_wts:
            if self.bias is not None:
                s += ', bias=True'
            s += ", Alpha={}".format(self.alpha.detach().cpu().numpy().shape)
            s += ", Beta={}".format(self.beta.detach().cpu().numpy().shape)
            s += ", weight={}".format(self.weight.detach().cpu().numpy().shape)
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
    
    def _init_(self, lbl_fts_mat, state_dict):
        if self.use_classifier_wts:
            print("INFO::Initializing the classifier")
            keys = list(state_dict.keys())
            key = [ x for x in keys if x.split(".")[-1] in ['weight']][0]
            weight = state_dict[key]
            self.weight.data.copy_(lbl_fts_mat.mm(weight))
