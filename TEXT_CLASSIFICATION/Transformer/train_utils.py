# train_utils.py

import torch
from torch import nn
from torch.autograd import Variable
import copy
import math
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from functools import partial
from util.constants import Pooling

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])




class Embeddings(nn.Module):
    # TODO: double-check the initialization
    '''
    Usual Embedding layer with weights multiplied by sqrt(d_model)
    '''
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe[:, 1::2] = torch.cos(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))#torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # self.pe = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        # print(x.size())
        # print(x.size(1))
        # print(self.pe[:, :x.size(1)].size())
        # exit()
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=True)
        return self.dropout(x)


class PoolingFunction:

    @classmethod
    def get_pool_func(self, pool, dim=1):
        """
        Return the corresponding torch activation function given its name

        Note that we have to warp return functions as class methods instead of lambda functions (e.g., max_pooling),
        otherwise there would be error saving models by torch.save
        """
        assert isinstance(pool, Pooling)
        if pool == Pooling.last_dim:  # Used in complex-order paper
            return partial(self.last_dim, dim=dim)
        elif pool == Pooling.sum:
            return partial(torch.sum, dim=dim)
        elif pool == Pooling.max:
            return partial(self.max_pooling, dim=dim)
        elif pool == Pooling.mean:
            return partial(torch.mean, dim=dim)
        else:
            raise NotImplementedError(f"Haven't yet implemented models with {pool.name} pooling.")

    @staticmethod
    def last_dim(x, dim=1):
        if dim == 1:
            return x[:,-1,:]
        else:
            return x[-1,:]

    @staticmethod
    def max_pooling(x, dim=1):
        return torch.max(x, dim=dim)[0]