# Model.py
import math

import torch
import torch.nn as nn
from copy import deepcopy
from train_utils import Embeddings, PoolingFunction
from attention import MultiHeadedAttention
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
import numpy as np
from torch.autograd import Variable

from util.constants import TC_OutputSize, Constants, PE_Type
from utils import *
from common.torch_util import TorchUtil as tu


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len, original_mode=False, small_pe=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.original_mode = original_mode
        self.small_pe = small_pe
        if self.original_mode:
            pe = torch.randn(Constants.ORIGINAL_MAX_PE_LEN, d_model)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        else:
            self.pe = nn.Embedding(max_len, d_model)
            self.positions = tu.move(torch.arange(max_len))
        
    def forward(self, x):
        if self.original_mode:
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        else:
            assert int(x.size(1)) == len(self.positions)
            pe = self.pe(self.positions)
            if not self.small_pe:
                pe *= math.sqrt(self.d_model)
            x = torch.add(x, pe)
        return self.dropout(x)

class Transformer_PE_real(nn.Module):
    def __init__(self, config, src_vocab, max_len=5000):
        super(Transformer_PE_real, self).__init__()
        
        h, N, dropout = config.model_cfg.trans_num_heads, config.model_cfg.trans_num_layers, config.model_cfg.trans_dropout
        d_model, d_ff = config.model_cfg.trans_dim_model, config.model_cfg.trans_dim_ff
        
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.PE_type = config.model_cfg.trans_pe_type
        self.encoder = Encoder(EncoderLayer(config.model_cfg.trans_dim_model, deepcopy(attn), deepcopy(ff), dropout), N)

        if self.PE_type == PE_Type.none:
            self.src_embed = nn.Sequential(Embeddings(config.model_cfg.trans_dim_model, src_vocab))
        elif self.PE_type == PE_Type.ape:
            position = PositionalEncoding(d_model, dropout, max_len=max_len,
                                          original_mode=config.original_mode,
                                          small_pe=config.model_cfg.trans_small_pe)
            self.src_embed = nn.Sequential(Embeddings(config.model_cfg.trans_dim_model, src_vocab), deepcopy(position))
        else:
            assert isinstance(self.PE_type, PE_Type)
            raise NotImplementedError(f"Haven't implemented model with {self.PE_type.name} (PE type).")

        self.fc = nn.Linear(
            config.model_cfg.trans_dim_model,
            int(TC_OutputSize[config.experiment_data])
        )

        self.pool = PoolingFunction.get_pool_func(config.model_cfg.trans_pooling)
        self.softmax = nn.Softmax(dim=1)

        self.max_epochs = config.num_epochs
        self.original_mode = config.original_mode

    def forward(self, x):
        embedded_sents = self.src_embed(x.permute(1,0))
        encoded_sents = self.encoder(embedded_sents)

        assert encoded_sents.ndim == 3
        final_feature_map = self.pool(encoded_sents)
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)
    
    # def add_optimizer(self, optimizer):
    #     self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def reduce_lr(self, optimizer):
        print("Reducing LR")
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / 2
                
    def run_epoch(self, train_iterator, epoch, optimizer):
        train_losses = []
        val_accuracies = []
        # losses = []
        
        # if (epoch == int(self.max_epochs/3)) or (epoch == int(2*self.max_epochs/3)):
        #     self.reduce_lr(optimizer)

        if not self.original_mode:
            self.train()

        for i, batch in enumerate(train_iterator):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text[0].cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text[0]
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            train_losses.append(loss.data.cpu().numpy())
            optimizer.step()
            if self.original_mode:
                self.train()
                
        return train_losses, val_accuracies