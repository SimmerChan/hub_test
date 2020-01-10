# encoding=utf-8

"""
@author: SimmerChan

@email: huangshilei@corp.netease.com

@file: utils.py

@desc:
"""
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(0.5)

        max_len = 500
        dim = 512

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, start_loc=0):
        x = x + self.pe[:, start_loc: start_loc + x.size(1)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask