# Model.py

import torch
import torch.nn as nn
from copy import deepcopy
from train_utils import Embeddings, PoolingFunction
from attention import MultiHeadedAttention
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
import numpy as np

from util.constants import TC_OutputSize
from utils import *


class Transformer_wo(nn.Module):
    def __init__(self, config, src_vocab, max_len=5000):
        super(Transformer_wo, self).__init__()

        h, N, dropout = config.model_cfg.trans_num_heads, config.model_cfg.trans_num_layers, config.model_cfg.trans_dropout
        d_model, d_ff = config.model_cfg.trans_dim_model, config.model_cfg.trans_dim_ff
        
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.encoder = Encoder(EncoderLayer(config.model_cfg.trans_dim_model, deepcopy(attn), deepcopy(ff), dropout), N)
        self.src_embed = nn.Sequential(Embeddings(config.model_cfg.trans_dim_model, src_vocab))

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
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
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

        if (epoch == int(self.max_epochs / 3)) or (epoch == int(2 * self.max_epochs / 3)):
            self.reduce_lr(optimizer)

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