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
from model.feature_extractor import MultiLabelFeatureExtractor
from model.transformer_modules import TransformerEncoder, TransformerDecoder
from model.utils import PositionalEmbedding
import math
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class Img2PoemTransformer(nn.Module):
    def __init__(self, ck, img_class_index2kws, vocab_size, word2idx, using_kws, gen_kws, embedding_dim=512, hidden_dim=512, ff_dim=2048, layer_num=6, head_num=8, max_len=35, pad_idx=0, dropout=0.1, transformer_ck_path=None):
        super(Img2PoemTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.using_kws = using_kws
        self.gen_kws = gen_kws
        self.idx2word = dict(zip(word2idx.values(), word2idx.keys()))

        self.img_class_index2kws = img_class_index2kws

        # TODO Define image encoder

        model = models.resnet101(pretrained=False)

        self.img_encoder_dim = 2048

        model.fc = nn.Linear(self.img_encoder_dim, ck['class_num'])
        model.load_state_dict(ck['model'])

        self.feature_extractor = MultiLabelFeatureExtractor(model, word2idx=word2idx, img_class_index2kws=img_class_index2kws, pad_idx=pad_idx)
        self.fine_tune_encoder(flag=False)
        self.map_layer = nn.Linear(self.img_encoder_dim, self.hidden_dim)

        # TODO Define embedding_layer for keywords encoder and poem decoder
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.position_embedding = PositionalEmbedding()

        self.dropout = nn.Dropout(dropout)

        # TODO Define kws encoder
        if self.using_kws:
            self.encoder = TransformerEncoder(layer_num, hidden_dim, head_num, ff_dim, dropout)

        # TODO Define decoder
        self.decoder = TransformerDecoder(layer_num, hidden_dim, head_num, ff_dim, dropout)

        if transformer_ck_path is not None:
            ck_point = torch.load(transformer_ck_path)
            raw_paras = ck_point['model']
            self.load_pretrained_transformer_weights(raw_paras)
            del raw_paras

        # TODO Define generator
        self.generator = nn.Linear(hidden_dim, vocab_size)
        self.generator.weight = self.embedding.weight

    def load_pretrained_transformer_weights(self, raw_paras):
        transformer_paras = {}
        embedding_paras = {}

        for k, v in raw_paras.items():
            if k.startswith('encoder'):
                _, name = k.split('.', maxsplit=1)
                transformer_paras[name] = v
            if k.startswith('embedding'):
                _, name = k.split('.', maxsplit=1)
                embedding_paras[name] = v

        if self.using_kws:
            encoder_paras_tmp = {}
            encoder_paras = dict(self.encoder.named_parameters())
            for k in transformer_paras.keys():
                if k in encoder_paras:
                    encoder_paras_tmp[k] = transformer_paras[k]

            encoder_paras.update(encoder_paras_tmp)
            self.encoder.load_state_dict(encoder_paras)
            print('Load pretrained poem LM for kws encoder.')

        self.embedding.load_state_dict(embedding_paras)
        print('Load pretrained poem LM for embedding layer.')

        decoder_paras_tmp = {}
        decoder_paras = dict(self.decoder.named_parameters())
        for k in transformer_paras.keys():
            if k in decoder_paras:
                decoder_paras_tmp[k] = transformer_paras[k]

        decoder_paras.update(decoder_paras_tmp)
        self.decoder.load_state_dict(decoder_paras)
        print('Load pretrained poem LM for decoder.')

    def forward(self, img, poem, kws=None):

        img_rep, kws_tuple = self.img_encoding(img)
        poem_rep = self.poem_encoding(poem)
        poem_mask = poem.data.eq(self.pad_idx).unsqueeze(1)  # (batch_size, 1, poem_len)

        kws_rep = None
        src_mask = None
        if self.using_kws:
            start_loc = img_rep.size(1)
            kws_rep, kws_mask, kws_names, kws_prob = self.kws_encoding(kws=kws, kws_tuple=kws_tuple, start_loc=start_loc)
            img_mask = torch.zeros((img_rep.size(0), 1, img_rep.size(1))).bool().to(device)
            src_mask = torch.cat([img_mask, kws_mask], dim=-1)

        pred = self.decoding(img_rep, poem_rep, poem_mask=poem_mask, kws_rep=kws_rep, src_mask=src_mask)

        return pred

    def img_encoding(self, img):
        img_rep, kws_tuple = self.feature_extractor(img, self.using_kws, self.gen_kws)
        img_rep = self.map_layer(img_rep)  # (batch_size, h x w, hidden_dim)
        return img_rep, kws_tuple

    def poem_encoding(self, poem):
        poem_embedding = self.dropout(self.embedding(poem)) * math.sqrt(self.hidden_dim)
        poem_rep = self.position_embedding(poem_embedding)
        return poem_rep

    def kws_encoding(self, kws=None, kws_tuple=None, start_loc=None):
        kws_names = None
        kws_prob = None

        if self.gen_kws:
            kws, kws_names, kws_prob = kws_tuple

        kws_mask = kws.data.eq(self.pad_idx).unsqueeze(1)

        kws_embedding = self.dropout(self.embedding(kws)) * math.sqrt(self.hidden_dim)
        kws_embedding = self.position_embedding(kws_embedding, start_loc=start_loc)

        kws_rep = self.encoder(kws_embedding)  # (batch_size, kws_len, hidden_dim)

        return kws_rep, kws_mask, kws_names, kws_prob

    def decoding(self, img_rep, poem_rep, poem_mask, kws_rep=None, src_mask=None):
        """

        :param img_rep: (batch_size, h X w, img_dim)
        :param poem_rep: (batch_size, poem_len, hidden_dim)
        :param poem_mask: (batch_size, 1, poem_len)
        :param kws_rep: (batch_size, kws_len, hidden_dim)
        :param src_mask: (batch_size, 1, kws_len + h X w)
        :return:
        """

        if kws_rep is not None:
            memory = torch.cat([img_rep, kws_rep], dim=1)  # (batch_size, h X w + kws_len, hidden_dim)
        else:
            memory = img_rep

        out = self.decoder(tgt_rep=poem_rep, memory=memory, src_pad_mask=src_mask, tgt_pad_mask=poem_mask)
        predictions = self.generator(out)  # (batch_size, poem_len, vocab_size)

        return predictions

    def fine_tune_encoder(self, flag=False):
        for p in self.feature_extractor.parameters():
            p.requires_grad = flag

    def beam_search(self, img, start_idx=3, beam_size=5, end_idx=4, decode_len=35, using_penalize=False):
        """
        :param img: (bs, channel, h, w)
        :param start_idx: indicates generate 5 or 7 quatrain.
        :param beam_size:
        :param end_idx:
        :param decode_len:
        :param using_penalize: penalize duplicate n-grams
        :return:
        """
        img_rep, kws_tuple = self.img_encoding(img)

        batch_size = img_rep.size(0)
        img_rep = img_rep.repeat(beam_size, 1, 1)

        kws_rep = None
        src_mask = None
        kws_names = None
        kws_prob = None
        if self.using_kws:
            start_loc = img_rep.size(1)
            kws_rep, kws_mask, kws_names, kws_prob = self.kws_encoding(kws_tuple=kws_tuple, start_loc=start_loc)
            kws_rep = kws_rep.repeat(beam_size, 1, 1)
            kws_mask = kws_mask.repeat(beam_size, 1, 1)
            img_mask = torch.zeros((img_rep.size(0), 1, img_rep.size(1))).bool().to(device)
            src_mask = torch.cat([img_mask, kws_mask], dim=-1)

        prevs = torch.zeros((beam_size * batch_size, 1)).fill_(start_idx).long().to(device)  # (beam_size * batch_size, 1)

        beam_scores = torch.zeros(batch_size, beam_size, device=device)
        beam_lens = torch.ones(batch_size, beam_size, dtype=torch.long, device=device)
        is_end = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

        for i in range(decode_len):
            poem_rep = self.poem_encoding(prevs)
            poem_mask = prevs.data.eq(self.pad_idx).unsqueeze(1)  # (batch_size, 1, poem_len)
            logits = self.decoding(img_rep, poem_rep, poem_mask, kws_rep, src_mask)
            logits = logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.view(batch_size, beam_size, -1)  # (bs, beam_size, vocab_size)

            if using_penalize:
                beam_scores = self.penalize_duplicate_grams(prevs, beam_scores)

            beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1))

            if i == 0:
                beam_scores = beam_scores[:, 0, :]
                beam_scores, idxs = beam_scores.topk(beam_size, dim=-1)
                beam_idxs = torch.zeros((batch_size, beam_size)).long().to(device)

            else:
                beam_scores = beam_scores.view(batch_size, -1)  # (bs, beam_size * vocab_size)

                _, idxs = beam_scores.topk(beam_size, dim=-1)  # (bs, beam_size)

                beam_scores = torch.gather(beam_scores, -1, idxs)  # (bs, beam_size)
                beam_idxs = (idxs.float() / self.vocab_size).long()

            current_input = torch.fmod(idxs, self.vocab_size)
            is_end = torch.gather(is_end, 1, beam_idxs)
            beam_lens = torch.gather(beam_lens, 1, beam_idxs)
            current_input[is_end] = self.pad_idx  # 结束的beam下一个输入为pad
            beam_lens[~is_end] += 1
            is_end[current_input == end_idx] = 1

            current_input = current_input.view(batch_size * beam_size, 1)
            prevs = prevs.view(batch_size, beam_size, -1)
            prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))

            prevs = prevs.view(batch_size * beam_size, -1)
            prevs = torch.cat([prevs, current_input], dim=1)

            if all(is_end.view(-1)):
                break

        all_predicts = []
        result = prevs.view(batch_size, beam_size, -1)

        bests = beam_scores.argmax(dim=-1)

        for i in range(batch_size):
            tmp_predicts = []
            for j in range(beam_size):
                length = beam_lens[i, j]
                seq = result[i, j, 1:length - 1]
                tmp_predicts.append(seq)
            all_predicts.append(tmp_predicts)

        best_predicts = []
        for i in range(batch_size):
            idxs = all_predicts[i][bests[i].item()].tolist()
            best_predicts.append(''.join([self.idx2word[idx] for idx in idxs]))

        # for preds in all_predicts:
        #     for pred in preds:
        #         print(''.join([self.idx2word[idx] for idx in pred.tolist()]))

        return best_predicts, kws_names, kws_prob

    @staticmethod
    def penalize_duplicate_grams(seqs, scores):
        """

        :param seqs: (bs * beam_size, length)
        :param scores: (bs, beam_size)
        :return:
        """
        num = seqs.size(0)
        bs = scores.size(0)
        scores = scores.view(-1, 1)  # (bs * beam_size, 1)
        for i in range(num):
            seq = seqs[i].tolist()
            all_2_grams_num = len(seq) - 1
            distinct_2_grams_num = len(set([str(seq[i]) + str(seq[i+1]) for i in range(all_2_grams_num)]))
            diff_num = all_2_grams_num - distinct_2_grams_num
            scores[i] += scores[i] * 0.04 * diff_num + scores[i] * 0.01 * (len(seq) - len(set(seq)))

        return scores.view(bs, -1)

    def top_search(self, img, start_idx=3, beam_size=5, end_idx=4, decode_len=35, top_k=100, top_p=None, using_penalize=False):
        """
        :param top_p:
        :param top_k:
        :param img: (bs, channel, h, w)
        :param start_idx: indicates generate 5 or 7 quatrain.
        :param beam_size:
        :param end_idx:
        :param decode_len:
        :param using_penalize: penalize duplicate n-grams
        :return:
        """
        img_rep, kws_tuple = self.img_encoding(img)

        batch_size = img_rep.size(0)
        img_rep = img_rep.repeat(beam_size, 1, 1)

        kws_rep = None
        src_mask = None
        kws_names = None
        kws_prob = None
        if self.using_kws:
            start_loc = img_rep.size(1)
            kws_rep, kws_mask, kws_names, kws_prob = self.kws_encoding(kws_tuple=kws_tuple, start_loc=start_loc)
            kws_rep = kws_rep.repeat(beam_size, 1, 1)
            kws_mask = kws_mask.repeat(beam_size, 1, 1)
            img_mask = torch.zeros((img_rep.size(0), 1, img_rep.size(1))).bool().to(device)
            src_mask = torch.cat([img_mask, kws_mask], dim=-1)

        prevs = torch.zeros((beam_size * batch_size, 1)).fill_(start_idx).long().to(device)  # (beam_size * batch_size, 1)

        beam_scores = torch.zeros(batch_size, beam_size, device=device)
        beam_lens = torch.ones(batch_size, beam_size, dtype=torch.long, device=device)
        is_end = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

        for i in range(decode_len):
            poem_rep = self.poem_encoding(prevs)
            poem_mask = prevs.data.eq(self.pad_idx).unsqueeze(1)  # (batch_size, 1, poem_len)
            logits = self.decoding(img_rep, poem_rep, poem_mask, kws_rep, src_mask)
            logits = logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.view(batch_size, beam_size, -1)  # (bs, beam_size, vocab_size)

            if top_p is None:
                log_probs, top_indices = log_probs.topk(top_k)  # (bs, beam_size, sampling_topk)
                probs = log_probs.exp_()  # (bs, beam_size, sampling_topk)
            else:
                probs = log_probs.exp_()
                sorted_probs, sorted_indices = probs.sort(descending=True)
                cumsum_probs = sorted_probs.cumsum(dim=2)
                mask = cumsum_probs.lt(top_p)
                max_dim = mask.sum()
                probs = sorted_probs[:, :, :max_dim+1]
                top_indices = sorted_indices[:, :, :max_dim+1]

            if using_penalize:
                beam_scores = self.penalize_duplicate_grams(prevs, beam_scores)

            indices_buf = torch.multinomial(probs.view(batch_size * beam_size, -1), 1, replacement=True).view(batch_size, beam_size)
            scores = torch.gather(probs, dim=2, index=indices_buf.unsqueeze(-1)).squeeze(2)
            scores_log = scores.log_().view(batch_size, -1)  # (bs, beam_size)
            idxs = torch.gather(top_indices.expand(batch_size, beam_size, -1), dim=2, index=indices_buf.unsqueeze(-1)).squeeze(2)  # (bs, beam_size)
            beam_scores += scores_log

            current_input = idxs
            current_input[is_end] = self.pad_idx
            beam_lens[~is_end] += 1
            is_end[current_input == end_idx] = 1
            current_input = current_input.view(batch_size * beam_size, 1)
            prevs = prevs.view(batch_size * beam_size, -1)
            prevs = torch.cat([prevs, current_input], dim=1)

            if all(is_end.view(-1)):
                break

        all_predicts = []
        result = prevs.view(batch_size, beam_size, -1)

        bests = beam_scores.argmax(dim=-1)

        for i in range(batch_size):
            tmp_predicts = []
            for j in range(beam_size):
                length = beam_lens[i, j]
                seq = result[i, j, 1:length - 1]
                tmp_predicts.append(seq)
            all_predicts.append(tmp_predicts)

        best_predicts = []
        for i in range(batch_size):
            idxs = all_predicts[i][bests[i].item()].tolist()
            best_predicts.append(''.join([self.idx2word[idx] for idx in idxs]))

        # for preds in all_predicts:
        #     for pred in preds:
        #         print(''.join([self.idx2word[idx] for idx in pred.tolist()]))

        return best_predicts, kws_names, kws_prob

    def hybrid_search(self, img, start_idx=3, beam_size=5, end_idx=4, decode_len=35, top_k=100, top_p=None, using_penalize=False):
        """
        :param top_p:
        :param top_k:
        :param img: (bs, channel, h, w)
        :param start_idx: indicates generate 5 or 7 quatrain.
        :param beam_size:
        :param end_idx:
        :param decode_len:
        :param using_penalize: penalize duplicate n-grams
        :return:
        """
        img_rep, kws_tuple = self.img_encoding(img)

        batch_size = img_rep.size(0)
        img_rep = img_rep.repeat(beam_size, 1, 1)

        kws_rep = None
        src_mask = None
        kws_names = None
        kws_prob = None
        if self.using_kws:
            start_loc = img_rep.size(1)
            kws_rep, kws_mask, kws_names, kws_prob = self.kws_encoding(kws_tuple=kws_tuple, start_loc=start_loc)
            kws_rep = kws_rep.repeat(beam_size, 1, 1)
            kws_mask = kws_mask.repeat(beam_size, 1, 1)
            img_mask = torch.zeros((img_rep.size(0), 1, img_rep.size(1))).bool().to(device)
            src_mask = torch.cat([img_mask, kws_mask], dim=-1)

        prevs = torch.zeros((beam_size * batch_size, 1)).fill_(start_idx).long().to(device)  # (beam_size * batch_size, 1)

        beam_scores = torch.zeros(batch_size, beam_size, device=device)
        beam_lens = torch.ones(batch_size, beam_size, dtype=torch.long, device=device)
        is_end = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

        # TODO 前两句用sampling的方式解码增加多样性
        for i in range((decode_len - 1) // 2):
            poem_rep = self.poem_encoding(prevs)
            poem_mask = prevs.data.eq(self.pad_idx).unsqueeze(1)  # (batch_size, 1, poem_len)
            logits = self.decoding(img_rep, poem_rep, poem_mask, kws_rep, src_mask)
            logits = logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.view(batch_size, beam_size, -1)  # (bs, beam_size, vocab_size)

            if top_p is None:
                log_probs, top_indices = log_probs.topk(top_k)  # (bs, beam_size, sampling_topk)
                probs = log_probs.exp_()  # (bs, beam_size, sampling_topk)
            else:
                probs = log_probs.exp_()
                sorted_probs, sorted_indices = probs.sort(descending=True)
                cumsum_probs = sorted_probs.cumsum(dim=2)
                mask = cumsum_probs.lt(top_p)
                max_dim = mask.sum()
                probs = sorted_probs[:, :, :max_dim + 1]
                top_indices = sorted_indices[:, :, :max_dim + 1]

            if using_penalize:
                beam_scores = self.penalize_duplicate_grams(prevs, beam_scores)

            indices_buf = torch.multinomial(probs.view(batch_size * beam_size, -1), 1, replacement=True).view(
                batch_size, beam_size)
            scores = torch.gather(probs, dim=2, index=indices_buf.unsqueeze(-1)).squeeze(2)
            scores_log = scores.log_().view(batch_size, -1)  # (bs, beam_size)
            idxs = torch.gather(top_indices.expand(batch_size, beam_size, -1), dim=2,
                                index=indices_buf.unsqueeze(-1)).squeeze(2)  # (bs, beam_size)
            beam_scores += scores_log

            current_input = idxs
            current_input[is_end] = self.pad_idx
            beam_lens[~is_end] += 1
            is_end[current_input == end_idx] = 1
            current_input = current_input.view(batch_size * beam_size, 1)
            prevs = prevs.view(batch_size * beam_size, -1)
            prevs = torch.cat([prevs, current_input], dim=1)

        # TODO 后两句用beam search的方式解码增加相关性
        for i in range((decode_len - 1) // 2 + 1):
            poem_rep = self.poem_encoding(prevs)
            poem_mask = prevs.data.eq(self.pad_idx).unsqueeze(1)  # (batch_size, 1, poem_len)
            logits = self.decoding(img_rep, poem_rep, poem_mask, kws_rep, src_mask)
            logits = logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.view(batch_size, beam_size, -1)  # (bs, beam_size, vocab_size)

            if using_penalize:
                beam_scores = self.penalize_duplicate_grams(prevs, beam_scores)

            beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1))

            beam_scores = beam_scores.view(batch_size, -1)  # (bs, beam_size * vocab_size)
            _, idxs = beam_scores.topk(beam_size, dim=-1)  # (bs, beam_size)

            beam_scores = torch.gather(beam_scores, -1, idxs)  # (bs, beam_size)
            beam_idxs = (idxs.float() / self.vocab_size).long()

            current_input = torch.fmod(idxs, self.vocab_size)
            is_end = torch.gather(is_end, 1, beam_idxs)
            beam_lens = torch.gather(beam_lens, 1, beam_idxs)
            current_input[is_end] = self.pad_idx  # 结束的beam下一个输入为pad
            beam_lens[~is_end] += 1
            is_end[current_input == end_idx] = 1

            current_input = current_input.view(batch_size * beam_size, 1)
            prevs = prevs.view(batch_size, beam_size, -1)
            prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))

            prevs = prevs.view(batch_size * beam_size, -1)
            prevs = torch.cat([prevs, current_input], dim=1)

            if all(is_end.view(-1)):
                break

        all_predicts = []
        result = prevs.view(batch_size, beam_size, -1)

        bests = beam_scores.argmax(dim=-1)

        for i in range(batch_size):
            tmp_predicts = []
            for j in range(beam_size):
                length = beam_lens[i, j]
                seq = result[i, j, 1:length - 1]
                tmp_predicts.append(seq)
            all_predicts.append(tmp_predicts)

        best_predicts = []
        for i in range(batch_size):
            idxs = all_predicts[i][bests[i].item()].tolist()
            best_predicts.append(''.join([self.idx2word[idx] for idx in idxs]))

        return best_predicts, kws_names, kws_prob


