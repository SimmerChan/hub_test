# encoding=utf-8

"""
@author: SimmerChan

@email: huangshilei@corp.netease.com

@file: kw2poem.py

@desc:
"""

import torch.nn as nn
from model.lstm_decoder import LSTMDecoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.masked_cross_entropy_loss import sequence_mask
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Kw2Poem(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512):
        super(Kw2Poem, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        self.decoder = LSTMDecoder(vocab_size=vocab_size, embedding=self.embedding)

    def forward(self, kws, poem, kw_lens):

        kws_mask = sequence_mask(kw_lens.view(-1, 1).to(kws.device))

        kws_embedding = self.dropout(self.embedding(kws))

        packed_embedding = pack_padded_sequence(kws_embedding, kw_lens, batch_first=True)

        kws_rep, kws_hidden = self.encoder(packed_embedding)

        kws_rep, _ = pad_packed_sequence(kws_rep, batch_first=True)

        pred = self.greedy_decoding(kws_rep, kws_hidden, poem, kws_mask)

        return pred

    def greedy_decoding(self, kws_rep, kws_hidden, poem, mask=None):
        bs, max_len = poem.size()

        predictions = torch.zeros(bs, max_len, self.vocab_size).to(device)

        for t in range(max_len):
            pred, kws_hidden = self.decoder(poem[:, t].unsqueeze(1), kws_rep, kws_hidden, mask=mask)
            predictions[:, t, :] = pred

        return predictions
