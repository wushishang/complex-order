# Model.py
import math

import torch
import torch.nn as nn
from copy import deepcopy

from model_transformer.model_utils import last_dims_tuple
from regularizers import JpRegularizer
from train_utils import Embeddings, PoolingFunction
from attention import MultiHeadedAttention
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
import numpy as np
from torch.autograd import Variable

from util.constants import TC_OutputSize, Constants, PE_Type, RegRepresentation, Regularization
from utils import *
from common.torch_util import TorchUtil as tu
from my_common.my_helper import is_tensor


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len, original_mode=False, small_pe=False, dropout_input=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.original_mode = original_mode
        self.small_pe = small_pe
        self.dropout_input = dropout_input
        if self.original_mode:
            pe = torch.randn(Constants.ORIGINAL_MAX_PE_LEN, d_model)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        else:
            self.pe = nn.Embedding(max_len, d_model)
            self.positions = tu.move(torch.arange(max_len))
        
    def forward(self, x, pe=None):
        if self.original_mode:
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
            return self.dropout(x), pe
        else:
            if pe is None:  # normal forward pass (training/evaluating pass); regularization pass if pe is provided
                assert int(x.size(1)) == len(self.positions)
                pe = self.pe(self.positions)
                if not self.small_pe:
                    pe = pe * math.sqrt(self.d_model)
                    x = torch.add(x, pe)
            else:
                x = torch.add(x, pe)

            if self.dropout_input:
                return self.dropout(x), pe
            else:
                return x, pe

class Transformer_PE_real(nn.Module):
    def __init__(self, config, src_vocab, max_len=5000):
        super(Transformer_PE_real, self).__init__()

        self.mask_padding = config.model_cfg.trans_mask_padding
        
        h, N, dropout = config.model_cfg.trans_num_heads, config.model_cfg.trans_num_layers, config.model_cfg.trans_dropout
        d_model, d_ff = config.model_cfg.trans_dim_model, config.model_cfg.trans_dim_ff
        
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.PE_type = config.model_cfg.trans_pe_type
        self.encoder = Encoder(EncoderLayer(config.model_cfg.trans_dim_model, deepcopy(attn), deepcopy(ff), dropout), N)

        if self.PE_type == PE_Type.ape:
            position = PositionalEncoding(d_model, dropout, max_len=max_len,
                                          original_mode=config.original_mode,
                                          small_pe=config.model_cfg.trans_small_pe,
                                          dropout_input=config.model_cfg.trans_dropout_input)
            self.input_embed = deepcopy(position)
        elif self.PE_type == PE_Type.none:
            pass
        else:
            assert isinstance(self.PE_type, PE_Type)
            raise NotImplementedError(f"Haven't implemented model with {self.PE_type.name} (PE type).")

        self.token_embed = Embeddings(config.model_cfg.trans_dim_model, src_vocab)

        self.fc = nn.Linear(
            config.model_cfg.trans_dim_model,
            int(TC_OutputSize[config.experiment_data])
        )

        self.pool = PoolingFunction.get_pool_func(config.model_cfg.trans_pooling, dim=0 if self.mask_padding else 1)
        self.softmax = nn.Softmax(dim=1)

        self.max_epochs = config.num_epochs
        self.original_mode = config.original_mode

    def forward(self, x, pe=None, sen_len=None):
        if self.mask_padding:
            assert sen_len is not None
            mask = self.generate_padding_mask(sen_len, x.shape[0])
        else:
            mask = None

        if x.dim() == 2:  # forward pass
            x = self.token_embed(x.permute(1, 0))
        else:
            assert x.dim() == 3  # regularization pass

        if self.PE_type == PE_Type.ape:
            embedded_sents, pe = self.input_embed(x, pe)
        else:
            assert pe is None
            embedded_sents = x
        encoded_sents = self.encoder(embedded_sents, mask)

        assert encoded_sents.ndim == 3
        if self.mask_padding:
            fml = []
            for idx, ss in enumerate(encoded_sents):
                sen_l = sen_len[idx]
                fm = self.pool(ss[:sen_l])
                fml.append(fm)
            final_feature_map = torch.stack(fml)
        else:
            final_feature_map = self.pool(encoded_sents)
        final_out = self.fc(final_feature_map)
        return final_out, final_feature_map, embedded_sents, pe

    def add_regularizer(self, regularizer: JpRegularizer):
        self.regularizer = regularizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def penalty(self, pred, latent, loss, return_latent_norm=False, **input_kwargs):
        """
        Birkhoff regularization and latent norm embedding
        kwargs can pass adjmats, sets, and possibly vertex features.
        """
        if loss.ndim > 1:
            if loss.ndim == 2:  # node classification
                loss = loss.mean(dim=last_dims_tuple(loss))
            else:
                raise RuntimeError(f"Unexpected loss dim: {loss.ndim}")
        overall_loss = loss
        breg_penalty = torch.zeros_like(loss)  # Even when not regularizing, we pass variable to reductions later
        # latent_penalty = torch.zeros_like(loss)
        # stats = dict()

        if self.training:
            assert pred.dim() in (2,3) and latent.dim() in (2, 3), \
                "Unexpected number of dimensions in representation passed to regularizer."
            # ------------------------------------
            # Penalize permutation sensitivity
            # ------------------------------------
            if self.regularizer.regularization != Regularization.none:
                if self.regularizer.representations == RegRepresentation.pred:
                    representation = pred
                elif self.regularizer.representations == RegRepresentation.latent:
                    representation = latent
                elif self.regularizer.representations == RegRepresentation.positional_embedding:
                    representation = input_kwargs['positional_encodings']
                else:
                    raise NotImplementedError(f"Haven't implemented regularization for {self.regularizer.representations.name}.")
                # compute penalty and multiply by regularization strength
                breg_penalty = self.regularizer(representation,
                                                self,
                                                disable_gradient=False,
                                                **input_kwargs
                                                )

                if self.regularizer.representations != RegRepresentation.pe and breg_penalty.shape != loss.shape:
                    raise RuntimeError("Differing shapes in penalty and loss")

                overall_loss = overall_loss + breg_penalty

            # # ------------------------------------
            # # Penalize large values of the latent
            # # ------------------------------------
            # if return_latent_norm or self.scaling.penalize_embedding:
            #     latent_norm = torch.norm(latent, dim=last_dims_tuple(latent))
            #     if return_latent_norm:
            #         stats.update({'latent_norm': latent_norm})
            #     if self.scaling.penalize_embedding:
            #         # strength * || latent ||
            #         latent_penalty = torch.mul(latent_norm, self.scaling.embedding_penalty)
            #         # Add to overall loss
            #         overall_loss = overall_loss + latent_penalty

            # # ------------------------------------
            # # Add weight decay to model components
            # # ------------------------------------
            # if self.weight_decay.active:
            #     # We only consider weight decay for positional embedding for now
            #     weight_decay_penalty = self.weight_decay.penalty(input_kwargs['positional_encodings'])
            #     overall_loss = overall_loss + weight_decay_penalty  # broadcasting to batch dim
            #     stats.update({'weight_decay_penalty': weight_decay_penalty})

        return overall_loss, breg_penalty # latent_penalty, stats

    def mean_reduction(self, overall_loss, breg_penalty):
        """
        Appropriate reduction of loss and penalty (or lack thereof):
        """
        return overall_loss.mean(), breg_penalty.mean()

    def predict(self, x, pe=None, sen_len=None):
        return self.__call__(x, pe, sen_len)[0]
                
    def run_epoch(self, train_iterator, optimizer, track_latent_norm=False):
        overall_losses = []
        train_losses = []
        breg_penalties = []
        # latent_penalties = []
        # stats = {}
        
        # if (epoch == int(self.max_epochs/3)) or (epoch == int(2*self.max_epochs/3)):
        #     self.reduce_lr(optimizer)

        if not self.original_mode:
            self.train()

        for i, batch in enumerate(train_iterator):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text[0].cuda()
                sen_len = batch.text[1].cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text[0]
                sen_len = batch.text[1]
                y = (batch.label - 1).type(torch.LongTensor)
            prediction, latent, embedded_sents, positional_encodings = self.__call__(x, sen_len=sen_len)
            train_loss = self.loss_op(self.softmax(prediction), y)
            overall_loss, breg_penalty = self.penalty(prediction, latent, train_loss,  # track_latent_norm,
                                                      sets=embedded_sents, positional_encodings=positional_encodings)
            overall_loss, breg_penalty = self.mean_reduction(overall_loss, breg_penalty)
            overall_loss.backward()
            overall_losses.append(overall_loss.item())
            breg_penalties.append(breg_penalty.item())
            # latent_penalties.append(latent_penalty.item())
            train_losses.append(overall_losses[-1] - breg_penalties[-1])
            optimizer.step()

            if self.original_mode:
                self.train()

        return overall_losses, train_losses, breg_penalties  #, latent_penalties, stats
        # return train_losses, val_accuracies


    def generate_padding_mask(self, sen_len, max_len):
        # Generate mask for padding with broadcasting trick. Assume post padding.
        # Ref: http://juditacs.github.io/2018/12/27/masked-attention.html
        assert is_tensor(sen_len) and sen_len.dim()==1 and len(sen_len) > 0
        assert is_positive_int(max_len)

        mask = torch.arange(max_len)[None, :] < sen_len[:, None]
        mask = mask.type(torch.LongTensor)
        mask = tu.move(mask.repeat_interleave(max_len, dim=0))
        return mask.view(len(sen_len), max_len, max_len)
