# encoding=utf-8

"""
@author: SimmerChan

@email: huangshilei@corp.netease.com

@file: transformer_encoder.py

@desc:

基于transformer的编码器

"""
import torch.nn as nn
import math
import torch
from model.transformer_modules import TransformerEncoder, PositionalEncoding


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransformerEncoderModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, vocab_size, hidden_dim=512, ff_dim=2048, head_num=8, layer_num=6, dropout=0.1, padding_idx=0):
        super(TransformerEncoderModel, self).__init__()

        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding()
        self.encoder = TransformerEncoder(layer_num, hidden_dim, head_num, ff_dim, dropout)
        self.hidden_dim = hidden_dim
        self.generator = nn.Linear(hidden_dim, vocab_size)

        self.generator.weight = self.embedding.weight

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        mask = src.data.eq(self.padding_idx).unsqueeze(1)  # (batch_size, 1, src_len)
        src = self.embedding(src) * math.sqrt(self.hidden_dim)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask)
        output = self.generator(output)
        return output