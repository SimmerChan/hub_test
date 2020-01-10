# encoding=utf-8

"""
@author: SimmerChan

@email: huangshilei@corp.netease.com

@file: imbalance_loss.py

@desc:
"""
import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiLabelImbalanceLoss(nn.Module):
    def __init__(self, class_num, pos_weight):
        super(MultiLabelImbalanceLoss, self).__init__()
        self.factor = torch.Tensor([1] * class_num).view(1, -1).to(device)
        self.status = torch.Tensor([0] * class_num).view(1, -1).bool().to(device)
        self.pos_weight = pos_weight
        self.tmp_tensor = torch.Tensor([0.01] * class_num).view(1, -1).to(device)

    def forward(self, logits, label):
        """
        According to
        "Tencent ML-Images: A Large-Scale Multi-Label Image Database for Visual Representation Learning"
        :param logits: model output without sigmoid layer (bs, class_num)
        :param label: (bs, class_num)
        :return:
        """
        bs, class_num = logits.size()
        status = label.bool().any(dim=0)
        status_mask = (self.status == status).view(1, -1)
        self.factor[status_mask] += 1
        self.factor[~status_mask] = 1
        self.status = status

        comp_1 = torch.cat((self.tmp_tensor, torch.log10(10 / (0.01 + self.factor)).to(device)), dim=0)
        comp_0 = torch.cat((self.tmp_tensor, torch.log10(10 / (8. + self.factor)).to(device)), dim=0)

        r_1, _ = torch.max(comp_1, dim=0)
        r_0, _ = torch.max(comp_0, dim=0)
        r_1 = r_1.repeat((bs, 1))
        r_0 = r_0.repeat((bs, 1))

        p_log_logits = torch.log(torch.sigmoid(logits))
        n_log_logits = torch.log(1 - torch.sigmoid(logits))

        # p_weight = self.pos_weight.repeat((bs, 1)).to(device)
        # loss = -(label * p_log_logits * r_1 * p_weight + (1 - label) * n_log_logits * r_0)

        loss = -(label * p_log_logits * r_1 + (1 - label) * n_log_logits * r_0)
        loss = loss.mean()

        return loss