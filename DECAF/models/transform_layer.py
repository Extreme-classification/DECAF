import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import sys
import models.custom_embeddings as tdif
from libs.utils import fetch_json



class Rpp(nn.Module):
    """Implementation of a Residual block
    Parameters:
    ----------
    input_size: int
            input dimension
    output_size: int
            output dimension
    dropout: float
            dropout probability
    init: str, default='eye'
            initialization strategy
    """

    def __init__(self, input_size, output_size, dropout,
                 init='eye', nonLin="r", transform=None,
                 device='cuda:0'):
        super(Rpp, self).__init__()
        self.name = "SR"
        self.input_size = input_size
        self.output_size = output_size
        self.init = init
        self.dropout = dropout
        self.nonLin = nonLin
        self.sparse = False
        self.padding_size = self.output_size - self.input_size
        self.device = device
        elements = []
        self.linear_pos = 0

        if self.nonLin == 'r':
            elements.append(nn.ReLU())
            self.linear_pos += 1
        elif self.nonLin == 'lr':
            elements.append(nn.LeakyReLU())
            self.linear_pos += 1
        else:
            pass

        if dropout > 0.0:
            elements.append(nn.Dropout(p=dropout))
            self.linear_pos += 1

        self.transform = transform
        if self.transform is not None:
            elements.append(nn.utils.spectral_norm(
                nn.Linear(self.input_size, self.input_size)))
        self.nonLin = nn.Sequential(*elements)

        self.a = Parameter(torch.Tensor(1, self.input_size))
        self.b = Parameter(torch.Tensor(1, self.input_size))

        self.initialize(self.init)

    def forward(self, embed):
        """Forward pass for Residual
        Parameters:
        ----------
        embed: torch.Tensor
                dense document embedding

        Returns
        -------
        out: torch.Tensor
                dense document embeddings transformed via residual block
        """
        embed = self.a.sigmoid()*self.nonLin(embed) + self.b.sigmoid()*embed
        return embed

    def initialize(self, init_type):
        """Initialize units
        Parameters:
        -----------
        init_type: str
                Initialize hidden layer with 'random' or 'eye'
        """
        if self.transform is not None:
            if init_type == 'random':
                nn.init.xavier_uniform_(
                    self.nonLin[self.linear_pos].weight,
                    gain=nn.init.calculate_gain('relu'))
            else:
                print("Using eye to initialize!")
                nn.init.eye_(self.nonLin[self.linear_pos].weight)
            if self.nonLin[self.linear_pos].bias is not None:
                nn.init.constant_(self.nonLin[self.linear_pos].bias, 0.0)
        self.a.data.fill_(0)
        self.b.data.fill_(0)

    def extra_repr(self):
        s = "(a): Parameter(dim={})".format(self.a.shape)
        s += "\n(b): Parameter(dim={})".format(self.b.shape)
        return s

    @property
    def stats(self):
        name = self.name
        weight = 0
        bias = 0
        a = torch.mean(self.a.sigmoid()).detach().cpu().numpy()
        b = torch.mean(self.b.sigmoid()).detach().cpu().numpy()
        if self.transform is not None:
            weight = torch.mean(torch.abs(
                torch.diagonal(self.nonLin[self.linear_pos].weight, 0))
            ).detach().cpu().numpy()
            if self.nonLin[self.linear_pos].bias is not None:
                bias = torch.mean(torch.abs(self.nonLin[self.linear_pos].bias))
        s = "{}(diag={:0.2f}, bias={:0.2f}, a={:0.2f}, b={:0.2f})".format(
            name, weight, bias, a, b)
        return s
    
    def to(self):
        print(self.device)
        super().to(self.device)

elements = {
    'dropout': nn.Dropout,
    'batchnorm1d': nn.BatchNorm1d,
    'linear': nn.Linear,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'Rpp': Rpp,
    'tfidf_embed': tdif.CustomEmbedding
}


class Transform(nn.Module):
    """
        Transform document representation!
        Arguments:
            ----------
            transform_string: lisr
                list of nn.Module objects
            device: string (default="cuda:0")
                gpu id to keep module objects
    """

    def __init__(self, modules, device="cuda:0"):
        super(Transform, self).__init__()
        self.device = device
        self.transform = nn.Sequential(*modules)

    def forward(self, embed):
        return self.transform(embed)

    def to(self):
        super().to(self.device)


def get_functions(obj, params=None):
    return list(map(lambda x: elements[x](**obj[x]), obj['order']))
