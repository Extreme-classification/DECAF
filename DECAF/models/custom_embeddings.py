import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class CustomEmbedding(torch.nn.Module):
    """
        Memory efficient way to compute weighted EmbeddingBag
    """

    def __init__(self, num_embeddings, embedding_dim,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, device="cuda:0"):
        """
            Args:
                num_embeddings: int: vocalubary size
                embedding_dim: int: dimension for embeddings
                padding_idx: int: index for <PAD>; embedding is not updated
                max_norm: 
                norm_type: int: default: 2
                scale_grad_by_freq: boolean: True/False
                sparse: boolean: sparse or dense gradients
        """
        super(CustomEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = num_embeddings
        if self.padding_idx is not None:
            self.num_embeddings = num_embeddings+1
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(self.num_embeddings, embedding_dim))
        self.sparse = sparse
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        """
            Reset weights
        """
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def to(self):
        super().to(self.device)

    def forward(self, batch_data):
        """
            Forward pass for embedding layer
            Arguments
                ----------
                batch_data: dict
                    {'X': torch.LongTensor (BxN),
                     'X_w': torch.FloatTensor (BxN)}
                    'X': Feature indices
                    'X_w': Feature weights
            Returns:
                ----------
                torch.Tensor
                    embedding for each sample B x embedding_dims
        """
        features = batch_data['X'].to(self.device)
        weights = batch_data['X_w'].to(self.device)
        out = F.embedding(
            features, self.weight,
            self.padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse)
        out = weights.unsqueeze(1).bmm(out).squeeze()
        return out

    def get_weights(self):
        return self.weight.detach().cpu().numpy()

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}, {device}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
    
    def _init_(self, state_dict):
        keys = list(state_dict.keys())
        key = [ x for x in keys if x.split(".")[-1] in ['weight']][0]
        weight = state_dict[key]
        self.weight.data.copy_(weight)
