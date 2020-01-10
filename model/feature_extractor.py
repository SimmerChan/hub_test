# encoding=utf-8

"""
@author: SimmerChan

@email: huangshilei@corp.netease.com

@file: feature_extractor.py

@desc:
"""
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatureExtractor(nn.Module):
    def __init__(self, model, word2idx, idx2category, pad_idx):
        super(FeatureExtractor, self).__init__()
        modules = list(model.children())
        self.resnet_feature = nn.Sequential(*modules[:-2])
        self.fc_pool = modules[-2]
        self.fc = modules[-1]
        self.idx2category = idx2category
        self.word2idx = word2idx
        self.pad_idx = pad_idx

    def forward(self, x, using_kws):
        feature = self.resnet_feature(x)  # (batch_size, 2048, h, w)
        kws_tuple = None

        if not self.training and using_kws:
            out = self.fc_pool(feature).squeeze(-1)
            out = out.squeeze(-1)
            logit = self.fc(out)  # (batch_size, class_num)
            pred_class_idx = logit.argmax(dim=-1)  # (bs)
            kw_char_idx, kw_lens, kws_names = self.get_kw_char_idx(pred_class_idx)
            kws_tuple = (kw_char_idx, kw_lens, kws_names)

        bs = feature.size(0)
        channel = feature.size(1)
        feature = feature.view(bs, channel, -1).permute(0, 2, 1)  # (batch_size, h X w, 2048)

        return feature, kws_tuple

    def get_kw_char_idx(self, class_idxs):
        class_seqs = list()
        class_names = list()
        seq_lens = list()
        max_char = 0
        device = class_idxs.device
        for i in range(class_idxs.size(0)):
            class_idx = class_idxs[i]
            class_name = self.idx2category[class_idx.item()]
            class_seq = [char for char in ''.join(class_name.split('_'))]
            seq_lens.append(len(class_seq))
            class_names.append(class_name)
            if len(class_seq) > max_char:
                max_char = len(class_seq)
            class_seqs.append(torch.Tensor([self.word2idx[char] for char in class_seq]).view(-1, 1).long().to(device))
        kw_char_idx = pad_sequence(class_seqs, batch_first=True, padding_value=self.pad_idx).squeeze(-1)

        return kw_char_idx, torch.Tensor(seq_lens).long(), class_names


class MultiLabelFeatureExtractor(nn.Module):
    def __init__(self, model, word2idx, img_class_index2kws, pad_idx):
        super(MultiLabelFeatureExtractor, self).__init__()
        modules = list(model.children())
        self.resnet_feature = nn.Sequential(*modules[:-2])
        self.fc_pool = modules[-2]
        self.fc = modules[-1]
        self.img_class_index2kws = img_class_index2kws
        self.word2idx = word2idx
        self.pad_idx = pad_idx
        self.top_n_img_pred = 2

    def forward(self, x, using_kws, gen_kws):
        feature = self.resnet_feature(x)  # (batch_size, 2048, h, w)
        kws_tuple = None

        if gen_kws and using_kws:
            out = self.fc_pool(feature).squeeze(-1)
            out = out.squeeze(-1)
            logit = self.fc(out)  # (batch_size, class_num)
            sigmoid_logit = logit.sigmoid()  # (bs, class_num)
            s_prob, s_idx = torch.sort(sigmoid_logit, dim=-1, descending=True)
            top_s_idx = s_idx[:, :self.top_n_img_pred]  # (bs, self.top_n_img_pred)
            top_s_prob = s_prob[:, :self.top_n_img_pred]  # (bs, self.top_n_img_pred)
            kw_char_idx, kws_names, kw_prob = self.get_kw_char_idx(top_s_idx, top_s_prob)
            kws_tuple = (kw_char_idx, kws_names, kw_prob)

        bs = feature.size(0)
        channel = feature.size(1)
        feature = feature.view(bs, channel, -1).permute(0, 2, 1)  # (batch_size, h X w, 2048)

        return feature, kws_tuple

    def get_kw_char_idx(self, top_s_idx, top_s_prob):
        top_s_idx = top_s_idx.tolist()
        batch_kw_seqs = list()
        batch_kw_names = list()
        batch_kw_probs = list()
        max_char = 0
        for batch_index in range(len(top_s_idx)):
            kw_names = set()
            kw_probs = list()

            # TODO select top_n_img_pred kws
            counter = 0
            while len(kw_names) < self.top_n_img_pred:
                for index, idx in enumerate(top_s_idx[batch_index]):
                    counter += 1
                    idx = str(idx)
                    if idx not in self.img_class_index2kws:
                        continue
                    kw_name = random.choice(self.img_class_index2kws[idx])
                    if kw_name not in kw_names:
                        kw_names.add(kw_name)
                        kw_probs.append(round(top_s_prob[batch_index][index].item(), 3))

                # TODO cannot find appropriate kw in prediction, randomly pick one
                if counter > 20:
                    if len(kw_names) == 0:
                        kw_names.add(random.choice(self.img_class_index2kws["612"]))
                        kw_probs.append(0)
                    break

            kw_seq = list()
            for name in kw_names:
                kw_seq.extend([self.word2idx[c] for c in name] + [self.word2idx['<kw_seg>']])
            kw_seq = kw_seq[:-1]

            batch_kw_names.append(list(kw_names))
            batch_kw_probs.append(kw_probs)
            if len(kw_seq) > max_char:
                max_char = len(kw_seq)
            batch_kw_seqs.append(torch.Tensor(kw_seq).view(-1, 1).long().to(device))
        kw_char_idx = pad_sequence(batch_kw_seqs, batch_first=True, padding_value=self.pad_idx).squeeze(-1)

        return kw_char_idx, batch_kw_names, batch_kw_probs
