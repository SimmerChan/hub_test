# encoding=utf-8

"""
@author: SimmerChan

@email: huangshilei@corp.netease.com

@file: img2poem.py

@desc:
"""

import torch.nn as nn
from torchvision import models
import torch
from model.feature_extractor import FeatureExtractor
from utils.masked_cross_entropy_loss import sequence_mask
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.multi_modal_decoder import MultiModalDecoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class Img2Poem(nn.Module):
    def __init__(self, ck_path, vocab_size, word2idx, using_kws, embedding_dim=512, hidden_dim=512, max_len=35, pad_idx=0):
        super(Img2Poem, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.using_kws = using_kws

        # TODO Define image encoder

        model = models.resnet101(pretrained=False)

        ck = torch.load(ck_path)
        category2idx = ck['category2idx']
        idx2category = dict(zip(category2idx.values(), category2idx.keys()))
        self.img_encoder_dim = 2048

        model.fc = nn.Linear(self.img_encoder_dim, len(category2idx))
        model.load_state_dict(ck['model'])

        self.feature_extractor = FeatureExtractor(model, word2idx=word2idx, idx2category=idx2category, pad_idx=pad_idx)
        self.fine_tune_encoder(flag=False)

        if not self.using_kws:
            self.init_h = nn.Linear(self.img_encoder_dim, self.hidden_dim)
            self.init_c = nn.Linear(self.img_encoder_dim, self.hidden_dim)

        # TODO Define embedding_layer for keywords encoder and poem decoder
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(0.3)

        # TODO Define kws encoder
        if self.using_kws:
            self.encoder = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)

        # TODO Define decoder

        self.decoder = MultiModalDecoder(vocab_size=vocab_size, embedding=self.embedding,
                                         img_encoder_dim=self.img_encoder_dim,
                                         kws_encoder_dim=self.hidden_dim,
                                         decoder_dim=self.hidden_dim,
                                         using_kws=self.using_kws)

    def forward(self, img, poem, kws=None, kw_lens=None):

        if self.using_kws:
            img_rep, kws_rep, kws_hidden, kws_mask, kws_names = self.encoding(img, kws, kw_lens)
            pred = self.greedy_decoding(img_rep, poem, kws_rep=kws_rep, init_hidden=kws_hidden, mask=kws_mask)
        else:
            img_rep = self.encoding(img, kws, kw_lens)
            img_hidden = self.init_hidden_state(img_rep)
            pred = self.greedy_decoding(img_rep, poem, init_hidden=img_hidden)
        return pred

    def encoding(self, img, kws, kw_lens, kws_names=None):
        img_rep, kws_tuple = self.feature_extractor(img, self.using_kws)

        if not self.training and self.using_kws:
            kws, kw_lens, kws_names = kws_tuple

        if self.using_kws:

            kws_mask = sequence_mask(kw_lens.view(-1, 1).to(kws.device))

            kws_embedding = self.dropout(self.embedding(kws))

            packed_embedding = pack_padded_sequence(kws_embedding, kw_lens, batch_first=True, enforce_sorted=self.training)

            kws_rep, kws_hidden = self.encoder(packed_embedding)

            kws_rep, _ = pad_packed_sequence(kws_rep, batch_first=True)

            return img_rep, kws_rep, kws_hidden, kws_mask, kws_names
        else:
            return img_rep

    def greedy_decoding(self, img_rep, poem, kws_rep=None, init_hidden=None, mask=None):
        bs, max_len = poem.size()

        predictions = torch.zeros(bs, max_len, self.vocab_size).to(device)
        decoder_hidden = init_hidden

        for t in range(max_len):
            pred, decoder_hidden = self.decoder(poem[:, t].unsqueeze(1), img_rep, kws_rep, decoder_hidden, mask=mask)
            predictions[:, t, :] = pred

        return predictions

    def fine_tune_encoder(self, flag=False):
        for p in self.feature_extractor.parameters():
            p.requires_grad = flag

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out).unsqueeze(0)  # (1, batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out).unsqueeze(0)
        return h, c



