# encoding=utf-8

"""
@author: SimmerChan

@email: huangshilei@corp.netease.com

@file: label_weight.py

@desc:
"""
import json
import torch
import glob
import os


def get_multi_label_pos_weight():
    """
    统计每个类别有多少图片，计算pos_weight，用于multi-label分类对不同类别进行加权
    :return:
    """
    old_index2count = dict()
    with open('/root/img2poem_data/valid_image_class_count.txt', 'r', encoding='utf-8') as f:
        for line in f:
            old_index, count, _ = line.strip().split('\t')
            old_index2count[old_index] = int(count)

    old_index2new_index = dict()
    all_new_index = json.load(open('/root/img2poem_data/all_new_index.json', 'r', encoding='utf-8'))
    for k, v in all_new_index.items():
        if len(v) == 2:
            old_index2new_index[v[1]] = k

    new_index2count = dict()
    for o, n in old_index2new_index.items():
        new_index2count[int(n)] = old_index2count[o]

    all_imgs_count = 0

    for p in glob.glob('/root/img2poem_data/new_index_img/*'):
        index = int(os.path.split(p)[1])
        img_num = len(glob.glob(p + '/*'))
        new_index2count[index] = img_num
        all_imgs_count += img_num

    s_count = sorted(new_index2count.items(), key=lambda x: x[0])
    all_imgs_count += len(glob.glob('/root/img2poem_data/valid_images/*'))

    pos_weight = []

    for k, v in s_count:
        weight = (all_imgs_count - v) / float(v)
        # weight = 1 - float(v) / all_imgs_count
        # print(k, v, weight)
        pos_weight.append(weight)

    return torch.Tensor(pos_weight)


if __name__ == '__main__':
    get_multi_label_pos_weight()