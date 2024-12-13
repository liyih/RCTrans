import copy
import numpy as np
import torch
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", cross_only=False):
        super().__init__()
        self.cross_only=cross_only
        if not cross_only:
            self.attn = MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
       
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)


    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, query_pos_embed, key, key_pos_embed, attn_mask=None):
        """
        :param query: B C Pq
        :param query_pos: N B C
        :return:
        """
        # NxCxP to PxNxC
        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)
        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.attn(q, k, value=v)[0]
            query = query + self.dropout(query2)
            query = self.norm(query)

        q = self.with_pos_embed(query, query_pos_embed)
        k = v = self.with_pos_embed(key, key_pos_embed)

        query2 = self.cross_attn(q, k, value=v)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # PxNxC to NxCxP
        query = query.permute(1, 2, 0)
        return query


