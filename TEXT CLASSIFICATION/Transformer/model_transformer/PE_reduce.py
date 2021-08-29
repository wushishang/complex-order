# Model.py
import math

import torch
import torch.nn as nn
from copy import deepcopy
from train_utils import Embeddings
from attention import MultiHeadedAttention
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
import numpy as np
from utils import *
from torch.autograd import Variable

from utils import get_pe_variance


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        self.d_model = d_model
        self.count = 0
        
    def forward(self, x):
        # TODO: track PE
        if self.count % 100 == 0:
            print(get_pe_variance(self.pe.weight))
        self.count += 1
        pe = self.pe(torch.arange(x.size(1))) * math.sqrt(self.d_model)
        x = torch.add(x, pe)
        return self.dropout(x)

class Transformer_PE_reduce(nn.Module):
    def __init__(self, config, src_vocab):
        super(Transformer_PE_reduce, self).__init__()
        self.config = config
        
        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff
        
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        
        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        self.src_embed = nn.Sequential(Embeddings(config.d_model, src_vocab), deepcopy(position)) 

        self.fc = nn.Linear(
            self.config.d_model,
            self.config.output_size
        )
        
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sents = self.src_embed(x.permute(1,0)) 
        encoded_sents = self.encoder(embedded_sents)

        # TODO: try other pooling func, e.g., mean/sum
        # final_feature_map = encoded_sents[:,-1,:]
        final_feature_map = torch.sum(encoded_sents, dim=1)
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
                
    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        
        if (epoch == int(self.config.max_epochs/3)) or (epoch == int(2*self.config.max_epochs/3)):
            self.reduce_lr()
            
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text[0].cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text[0]
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            self.train()                
                
        return train_losses, val_accuracies