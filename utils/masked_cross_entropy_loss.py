# encoding=utf-8

"""
@author: SimmerChan

@email: huangshilei@corp.netease.com

@file: masked_cross_entropy_loss.py

@desc:
source: https://github.com/howardyclo/pytorch-seq2seq-example/blob/master/seq2seq.ipynb
"""
import torch
import torch.nn.functional as F


def sequence_mask(sequence_length, max_len=None):
    """
    Caution: Input and Return are VARIABLE.
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.expand_as(seq_range_expand))
    mask = seq_range_expand < seq_length_expand

    return mask


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.

    The code is same as:

    weight = torch.ones(tgt_vocab_size)
    weight[padding_idx] = 0
    criterion = nn.CrossEntropyLoss(weight.cuda(), size_average)
    loss = criterion(logits_flat, losses_flat)
    """
    # TODO logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))

    # TODO log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)

    # TODO target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)

    # TODO losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # TODO losses: (batch, max_len)
    losses = losses_flat.view(*target.size())

    # TODO mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))

    # TODO Note: mask need to bed casted to float!
    losses = losses * mask.float()

    loss = losses.sum() / mask.float().sum()

    # TODO (batch_size * max_tgt_len,)
    pred_flat = log_probs_flat.max(1)[1]

    # TODO (batch_size * max_tgt_len,) => (batch_size, max_tgt_len)
    pred_seqs = pred_flat.view(*target.size())

    # TODO (batch_size, max_len) => (batch_size * max_tgt_len,)
    mask_flat = mask.view(-1)

    # TODO `.float()` IS VERY IMPORTANT !!!
    # TODO https://discuss.pytorch.org/t/batch-size-and-validation-accuracy/4066/3
    # TODO The problem is that pred == y returns a ByteTensor, which has only an 8-bit range. Hence, after a particular batch-size, the sum was overflowing, and hence the wrong results.
    # TODO Actually, `eq` func return ByteTensor, `masked_select` return the source Tensor type, namely ByteTensor in this example, so we need convert it to float in case of overflowing.
    num_corrects = int(pred_flat.eq(target_flat.squeeze(1)).masked_select(mask_flat).float().data.sum())
    num_words = int(length.data.sum())

    return loss, pred_seqs, num_corrects, num_words