# encoding=utf-8

"""
@author: SimmerChan

@email: huangshilei@corp.netease.com

@file: multi_modal_decoder.py

@desc:
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_hidden):
        super(Attention, self).__init__()
        self.attn_layer = nn.Linear(encoder_dim + decoder_hidden, decoder_hidden)
        self.v = nn.Parameter(torch.randn(decoder_hidden), requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        """

        :param encoder_outputs: (bs, seq_len, encoder_dim)
        :param decoder_hidden: (1, bs, hidden_dim)
        :return:
        """
        bs = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.repeat(seq_len, 1, 1).permute(1, 0, 2)  # (bs, seq_len, hidden_dim)

        energy = torch.tanh(self.attn_layer(torch.cat((encoder_outputs, decoder_hidden), dim=2)))  # (bs, seq_len, hidden_dim)
        v = self.v.repeat(bs, 1).unsqueeze(2)  # (bs, hidden_dim, 1)

        weights = torch.bmm(energy, v)  # (bs, seq_len, 1)
        if mask is not None:
            weights.data.masked_fill_(~mask.unsqueeze(-1).data, -float('inf'))
        weights = weights.softmax(dim=1)

        weighted_encoder_outputs = (weights * encoder_outputs).sum(dim=1, keepdim=True)  # (batch_size, 1, encoder_dim)

        return weighted_encoder_outputs


class MultiModalDecoder(nn.Module):
    def __init__(self, vocab_size, embedding, img_encoder_dim, kws_encoder_dim, decoder_dim, using_kws, pad_idx=0):
        super(MultiModalDecoder, self).__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.img_encoder_dim = img_encoder_dim
        self.kws_encoder_dim = kws_encoder_dim
        self.decoder_dim = decoder_dim
        self.embedding = embedding
        self.dropout = nn.Dropout(0.5)
        self.img_attention = Attention(self.img_encoder_dim, self.decoder_dim)
        self.using_kws = using_kws
        if self.using_kws:
            self.kws_attention = Attention(self.kws_encoder_dim, self.decoder_dim)
            self.decoder = nn.LSTM(img_encoder_dim + kws_encoder_dim + self.embedding.embedding_dim, self.decoder_dim, batch_first=True)
        else:
            self.decoder = nn.LSTM(img_encoder_dim + self.embedding.embedding_dim, self.decoder_dim, batch_first=True)

        self.generator = nn.Linear(decoder_dim + self.embedding.embedding_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.generator.bias.data.fill_(0)
        self.generator.weight.data.uniform_(-0.1, 0.1)

    def forward(self, tgt_char, img_rep, kws_rep, hidden, mask=None):
        """

        :param tgt_char: (bs, 1)
        :param img_rep: (bs, pixels, channel(img_encoder_dim))
        :param kws_rep: (bs, kws_max_len, kws_encoder_dim)
        :param hidden: (1, bs, decoder_dim)
        :param mask: (bs, kws_max_Len)
        :return:
        """
        h, c = hidden

        weighted_img_rep = self.img_attention(img_rep, h)  # (bs, 1, img_encoder_dim)
        current_input = self.dropout(self.embedding(tgt_char))

        if self.using_kws:
            weighted_kws_rep = self.kws_attention(kws_rep, h, mask)  # (bs, 1, kws_encoder_dim)
            decoder_input = torch.cat([current_input, weighted_kws_rep, weighted_img_rep], dim=-1)  # (bs, 1, kws_encoder_dim + img_encoder_dim + embedding_dim)
        else:
            decoder_input = torch.cat([current_input, weighted_img_rep], dim=-1)  # (bs, 1, img_encoder_dim + embedding_dim)

        output, (h, c) = self.decoder(decoder_input, (h, c))  # output (bs, 1, decoder_dim)
        concat_output = torch.cat((output, current_input), dim=-1)  # (bs, 1, decoder_dim + embedding_dim)
        pred = self.generator(self.dropout(concat_output)).squeeze(1)  # (bs, vocab_size)

        return pred, (h, c)

    def beam_search(self, start_idx, img_rep, kws_rep=None, init_hidden=None, beam_size=5, end_idx=4, decode_len=35):
        """

        :param start_idx: indicates generate 5 or 7 quatrain.
        :param img_rep: (bs, pixels, channel(img_encoder_dim))
        :param kws_rep: (bs, kws_max_len, encoder_dim)
        :param init_hidden: (1, bs, encoder_dim)
        :param beam_size:
        :param end_idx:
        :param decode_len:
        :return:
        """

        batch_size = img_rep.size(0)
        img_rep = img_rep.repeat(beam_size, 1, 1)

        if kws_rep is not None:
            kws_rep = kws_rep.repeat(beam_size, 1, 1)

        h, c = init_hidden
        h = h.repeat(1, beam_size, 1)
        c = c.repeat(1, beam_size, 1)
        init_hidden = (h, c)
        decoder_hidden = init_hidden

        current_input = torch.full((batch_size * beam_size, 1), fill_value=start_idx).long().to(device)
        prevs = torch.zeros((beam_size * batch_size, 1)).fill_(start_idx).long().to(device)  # (beam_size * batch_size, 1)

        beam_scores = torch.zeros(batch_size, beam_size, device=device)
        beam_lens = torch.ones(batch_size, beam_size, dtype=torch.long, device=device)
        is_end = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

        for i in range(decode_len):
            logits, decoder_hidden = self(current_input, img_rep, kws_rep, decoder_hidden, mask=None)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.view(batch_size, beam_size, -1)  # (bs, beam_size, vocab_size)

            beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1))

            if i == 0:
                beam_scores = beam_scores[:, 0, :]
                beam_scores, idxs = beam_scores.topk(beam_size, dim=-1)
                beam_idxs = torch.zeros((batch_size, beam_size)).long().to(device)

            else:
                beam_scores = beam_scores.view(batch_size, -1)  # (bs, beam_size * vocab_size)

                # TODO sampling
                # _, idxs = beam_scores.topk(beam_size, dim=-1)  # (bs, beam_size)
                idxs = torch.multinomial(input=beam_scores.exp(), num_samples=beam_size, replacement=True, out=idxs)

                beam_scores = torch.gather(beam_scores, -1, idxs)  # (bs, beam_size)
                beam_idxs = (idxs.float() / self.vocab_size).long()

            current_input = torch.fmod(idxs, self.vocab_size)
            is_end = torch.gather(is_end, 1, beam_idxs)
            beam_lens = torch.gather(beam_lens, 1, beam_idxs)

            current_input[is_end] = self.pad_idx  # 结束的beam下一个输入为pad
            beam_lens[~is_end] += 1
            is_end[current_input == end_idx] = 1

            current_input = current_input.view(batch_size * beam_size, 1)  # 出错 current_input = beam_idxs.view(current_bs * beam_size, 1)
            prevs = prevs.view(batch_size, beam_size, -1)
            prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))

            prevs = prevs.view(batch_size * beam_size, -1)
            prevs = torch.cat([prevs, current_input], dim=1)

            if all(is_end.view(-1)):
                break

        predicts = []
        result = prevs.view(batch_size, beam_size, -1)

        bests = beam_scores.argmax(dim=-1)

        for i in range(batch_size):
            best_len = beam_lens[i, bests[i]]
            best_seq = result[i, bests[i], 1:best_len - 1]
            predicts.append(best_seq.tolist())

        return predicts