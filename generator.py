# encoding=utf-8

"""
@author: SimmerChan

@email: huangshilei@corp.netease.com

@file: generator.py

@desc:
"""

import torch
from model.img2poem_transformer import Img2PoemTransformer
from torchvision import transforms
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Image2PoemGenerator:
    def __init__(self, checkpoint, cnn_checkpoint, using_kws=False):
        """

        :param checkpoint: loaded checkpoint -> dict ['word2idx', 'model', 'best_epoch', 'train_loss', 'dev_loss']
        :param cnn_checkpoint: loaded checkpoint -> dict ['model', 'class_num', 'best_epoch', 'train_acc', 'dev_acc']
        :param using_kws:
        """

        self.word2idx = checkpoint['word2idx']
        vocab_size = len(self.word2idx)
        self.img_input_size = 224

        # self.model = Img2Text(cnn_ckpoint, vocab_size, word2idx=self.word2idx)
        # self.model = Img2Poem(cnn_ckpoint, vocab_size=vocab_size, word2idx=self.word2idx, using_kws=using_kws)

        img_class_index2kws = json.load(open('./data/all_new_index2kw.json', 'r', encoding='utf-8'))
        self.using_kws = using_kws
        if self.using_kws:
            self.gen_kws = True
        else:
            self.gen_kws = False
        self.model = Img2PoemTransformer(cnn_checkpoint, img_class_index2kws, vocab_size, self.word2idx, using_kws, self.gen_kws)

        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)
        self.model.eval()

        self.types = {'五言': '<five>', '七言': '<seven>'}
        self.decode_lens = {'五言': 25, '七言': 33}

    def generate(self, img, poem_type='五言', search_type='beam search', **kwargs):
        img = self._image_prepocess(img)

        beam_size = kwargs['beam_size']
        top_k = kwargs['top_k']
        top_p = kwargs['top_p']

        start_idx = self.word2idx[self.types[poem_type]]

        if search_type == 'beam search':
            predict_seqs, kw_names, kw_probs = self.model.beam_search(img, start_idx=start_idx, beam_size=beam_size, end_idx=4, decode_len=self.decode_lens[poem_type], using_penalize=True)
        elif search_type == 'top-k' or search_type == 'top-p':
            predict_seqs, kw_names, kw_probs = self.model.top_search(img, start_idx=start_idx, beam_size=beam_size, end_idx=4, decode_len=self.decode_lens[poem_type], top_k=top_k, top_p=top_p, using_penalize=True)
        elif search_type == 'hybrid':
            predict_seqs, kw_names, kw_probs = self.model.hybrid_search(img, start_idx=start_idx, beam_size=beam_size, end_idx=4, decode_len=self.decode_lens[poem_type], top_k=top_k, top_p=top_p, using_penalize=True)
        else:
            raise ValueError('Not supported search methods.')

        predict_seq = predict_seqs[0]

        return predict_seq, self.decode_lens[poem_type]

    def _image_prepocess(self, image):
        if image.mode != "RGB":
            image = image.convert('RGB')

        transforms_pipe = transforms.Compose([
            transforms.Resize((self.img_input_size, self.img_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transforms_pipe(image)
        image = image.unsqueeze(0).to(device)

        return image


if __name__ == '__main__':
    model_ck = './checkpoints/best_img2poem.pt'
    cnn_ck = './checkpoints/best_vgg.pt'

    import time

    print('Loading Model..')
    s = time.time()
    generator = Image2PoemGenerator(model_ck, cnn_ck, using_kws=True)
    print('Done in {}s.'.format(time.time() - s))

    while True:
        try:
            path = input("Input image path:")
            print(generator.generate(path, poem_type='七言'))
        except KeyboardInterrupt:
            print('Game exits.')
            break
