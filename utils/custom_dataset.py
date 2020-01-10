# encoding=utf-8

"""
@author: SimmerChan

@email: huangshilei@corp.netease.com

@file: custom_dataset.py

@desc:
"""

from torch.utils.data import Dataset
import glob
from PIL import Image
from torchvision import transforms
import json
import os
from collections import Counter, defaultdict
import torch
import random
from tqdm import tqdm
import pickle
import h5py
import numpy as np

random.seed(0)
current_path = os.path.realpath(__file__)
root_path = current_path.rsplit('/', 2)[0]


class PoemCorpus:
    def __init__(self):
        self.data = list()
        self.word2idx = json.load(open('/root/img2poem_data/poem_word2idx.json', 'r', encoding='utf-8'))
        self.word_freq = defaultdict(int)

        save_path = os.path.join(root_path, 'data/poem_data.pkl')
        if not os.path.exists(save_path):
            with open(os.path.join(root_path, 'data/poem.txt'), 'r', encoding='utf-8') as f:
                for line in f:
                    poem = line.strip()
                    self.data.append(poem)

            random.shuffle(self.data)
            split_point = int(len(self.data) * 0.95)
            self.train_data = self.data[:split_point]
            self.dev_data = self.data[split_point:]
            pickle.dump([self.train_data, self.dev_data], open(save_path, 'wb'))
        else:
            data = pickle.load(open(save_path, 'rb'))
            self.train_data = data[0]
            self.dev_data = data[1]

        print('Poem data set:')
        print('Train num: {}'.format(len(self.train_data)))
        print('Dev num: {}'.format(len(self.dev_data)))
        print('Vocab num: {}'.format(len(self.word2idx)))


class PoemDataset(Dataset):
    def __init__(self, data, word2idx):
        super(PoemDataset, self).__init__()
        self.data = data
        self.word2idx = word2idx

    def __getitem__(self, index):
        poem = self.data[index]
        if len(poem) == 24:
            start_token = '<five>'
        elif len(poem) == 32:
            start_token = '<seven>'
        else:
            raise ValueError('Not valid poem len')

        poem_idx = [self.word2idx[start_token]] + [self.word2idx.get(c, self.word2idx['<unk>']) for c in poem] + [self.word2idx['<eos>']]
        length = len(poem_idx)
        poem_idx = poem_idx + [self.word2idx['<pad>'] for _ in range(34 - length)]
        return torch.Tensor(poem_idx).long(), torch.Tensor([length]).long()

    def __len__(self):
        return len(self.data)


class ImgCorpus:
    def __init__(self):
        self.category2idx = dict()
        self.data = list()

        save_path = os.path.join(root_path, 'data/img_data.pkl')
        if not os.path.exists(save_path):
            img_path = os.path.join(root_path, 'data/img/*')
            dir_paths = glob.glob(img_path)
            paths = tqdm(dir_paths)
            for index, dir_path in enumerate(paths):
                paths.set_description("Load Images {}/{} dirs".format(index + 1, len(paths)))
                dir_name = os.path.split(dir_path)[1].encode('utf-8', errors='surrogateescape').decode('utf-8')

                if dir_name not in self.category2idx:
                    idx = len(self.category2idx)
                    self.category2idx[dir_name] = idx

                label = self.category2idx[dir_name]
                tmp_path = os.path.join(dir_path, '*')
                img_paths = glob.glob(tmp_path)
                for img_path in img_paths:
                    with open(img_path, 'rb') as f:
                        self.data.append([Image.open(f).copy(), label])

            random.shuffle(self.data)
            split_point = int(len(self.data) * 0.95)
            self.train_data = self.data[:split_point]
            self.dev_data = self.data[split_point:]
            pickle.dump([self.train_data, self.dev_data, self.category2idx], open(save_path, 'wb'))
        else:
            data = pickle.load(open(save_path, 'rb'))
            self.train_data = data[0]
            self.dev_data = data[1]
            self.category2idx = data[2]

        print('Image data set:')
        print('Train num: {}'.format(len(self.train_data)))
        print('Dev num: {}'.format(len(self.dev_data)))
        print('Class num: {}'.format(len(self.category2idx)))


class ImgDataset(Dataset):
    def __init__(self, data):
        super(ImgDataset, self).__init__()
        self.data = data
        self.transforms = transforms.Compose([
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img, label = self.data[index]
        img = self.transforms(img)
        return img, torch.Tensor([label]).long()

    def __len__(self):
        return len(self.data)


class PairCorpus:
    def __init__(self):

        self.word2idx = {'<pad>': 0, '<unk>': 1, '<five>': 2, '<seven>': 3, '<eos>': 4}
        self.word_freq = defaultdict(int)
        self.data = list()
        self.category2idx = dict()

        save_path = os.path.join(root_path, 'data/img_pair_data.pkl')
        if not os.path.exists(save_path):
            img_corpus = ImgCorpus()
            category2idx = img_corpus.category2idx

            img_dir_paths = glob.glob(os.path.join(root_path, 'data/img/*'))
            paths = tqdm(img_dir_paths)
            for index, img_dir_path in enumerate(paths):
                paths.set_description("Load Images {}/{} dirs".format(index + 1, len(paths)))
                poem_dir_path = img_dir_path.replace('/img/', '/poem/')
                dir_name = os.path.split(img_dir_path)[1].encode('utf-8', errors='surrogateescape').decode('utf-8')
                label = category2idx[dir_name]
                imgs = list()
                for img_path in glob.glob(os.path.join(img_dir_path, '*')):
                    with open(img_path, 'rb') as f:
                        imgs.append(Image.open(f).copy())
                poems = [line.strip() for line in open(os.path.join(poem_dir_path, 'poem.txt'), 'r', encoding='utf-8').readlines()]
                self.build_pair(imgs, poems, label, dir_name)

                self.train_data = self.data[:-1000]
                self.dev_data = self.data[-1000:]

            for w, f in Counter(self.word_freq).most_common(len(self.word_freq)):
                idx = len(self.word2idx)
                self.word2idx[w] = idx

            pickle.dump([self.train_data, self.dev_data, category2idx, self.word2idx], open(save_path, 'wb'))
        else:
            data = pickle.load(open(save_path, 'rb'))
            self.train_data = data[0]
            self.dev_data = data[1]
            self.category2idx = data[2]
            self.word2idx = data[3]

        print('Image-poem data set:')
        print('Train num: {}'.format(len(self.train_data)))
        print('Dev num: {}'.format(len(self.dev_data)))
        print('Vocab num: {}'.format(len(self.word2idx)))

    def build_pair(self, imgs, poems, label, dir_name):
        random.shuffle(poems)

        img_num = len(imgs)
        category_chars = ''.join(dir_name.split('_'))

        for index, poem in enumerate(poems):

            for char in poem:
                self.word_freq[char] += 1

            if index < img_num:
                self.data.append([imgs[index], list(poem), label, category_chars])
            else:
                self.data.append([random.choice(imgs), list(poem), label, category_chars])
                # break


class PairDataset(Dataset):
    def __init__(self, data: list, word2idx: dict, pad_idx=0):
        super(PairDataset, self).__init__()
        self.data = data
        self.pad_idx = pad_idx
        self.max_len = 35
        self.max_kw_char = 12
        self.transforms = transforms.Compose([
            # transforms.ColorJitter(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.word2idx = word2idx

    def __getitem__(self, index):
        img, poem, label, kw_chars = self.data[index]

        img = self.transforms(img)

        poem_len = len(poem)
        assert poem_len == 24 or poem_len == 32, print("Not valid 5-char or 7-char quatrain!")
        if poem_len == 24:
            start_token_idx = self.word2idx['<five>']
        else:
            start_token_idx = self.word2idx['<seven>']

        poem_idx = [start_token_idx] + [self.word2idx[c] for c in poem] + [self.word2idx['<eos>']] + [self.pad_idx for _
                                                                                                      in range(
                self.max_len - poem_len - 2)]

        kw_chars_len = len(kw_chars)
        kw_chars_idx = [self.word2idx[c] for c in kw_chars] + [self.pad_idx for _ in range(self.max_kw_char - len(kw_chars))]

        # return img, torch.Tensor(poem_idx).long(), torch.Tensor([poem_len]).long(), torch.Tensor(kw_chars_idx).long()
        return img, torch.Tensor(poem_idx).long(), torch.Tensor([poem_len]).long(), torch.Tensor([kw_chars_len]).long(), torch.Tensor(kw_chars_idx).long()
        # return img, torch.Tensor(poem_idx).long()

    def __len__(self):
        return len(self.data)


class MultiLabelImgCorpus:
    def __init__(self):
        h5py_file = h5py.File('/root/img2poem_data/img_dataset.h5', 'r')
        self.train_img = h5py_file['train_img']
        self.dev_img = h5py_file['dev_img']
        self.train_label = h5py_file['train_label']
        self.dev_label = h5py_file['dev_label']

        self.class_num = self.train_label.shape[1]

        print('Train data num: {}'.format(self.train_img.shape[0]))
        print('Dev data num: {}'.format(self.dev_img.shape[0]))
        print('Class num: {}'.format(self.class_num))


class MultiLabelImgDataset(Dataset):
    def __init__(self, img, label):
        super(MultiLabelImgDataset, self).__init__()
        self.imgs = img
        self.labels = label
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img = self.transforms(self.imgs[index])
        label = torch.Tensor(self.labels[index])
        return img, label

    def __len__(self):
        return len(self.imgs)


class MultiLabelPairCorpus:
    def __init__(self):
        h5py_file = h5py.File('/root/img2poem_data/pair_dataset.h5', 'r')
        self.train_poem = h5py_file['train_poem']
        self.train_kws = h5py_file['train_kws']
        self.train_img = h5py_file['train_img']

        self.dev_poem = h5py_file['dev_poem']
        self.dev_kws = h5py_file['dev_kws']
        self.dev_img = h5py_file['dev_img']

        self.word2idx = json.load(open('/root/img2poem_data/poem_word2idx.json', 'r', encoding='utf-8'))
        self.vocab_size = len(self.word2idx)

        print('Train data num: {}'.format(self.train_img.shape[0]))
        print('Dev data num: {}'.format(self.dev_img.shape[0]))
        print('Vocab num: {}'.format(self.vocab_size))


class MultiLabelPairDataset(Dataset):
    def __init__(self, poem, kws, img):
        super(MultiLabelPairDataset, self).__init__()
        self.poems = poem
        self.kws = kws
        self.imgs = img
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        poem = torch.Tensor(self.poems[index].astype(np.int16)).long()
        kws = torch.Tensor(self.kws[index].astype(np.int16)).long()
        img = self.transforms(self.imgs[index])
        return poem, kws, img

    def __len__(self):
        return len(self.imgs)


def pair_data_collate(data: list):
    data.sort(key=lambda x: x[3].item(), reverse=True)

    img, poem, poem_len, kw_len, kw_chars_idx = zip(*data)

    return torch.stack(img, dim=0), torch.stack(poem, dim=0), torch.stack(poem_len, dim=0), torch.stack(kw_chars_idx, dim=0), torch.stack(kw_len, dim=0).squeeze(1)