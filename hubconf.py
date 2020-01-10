# encoding=utf-8

from generator import Image2PoemGenerator
import torch
from io import BytesIO

dependencies = ["torch", "torchvision", "pillow<7.0.0"]


def model():
    s2s_ckpoint = torch.hub.load_state_dict_from_url('https://github.com/SimmerChan/hub_test/releases/download/v1.0/best_img2poem_transformer_wo_kws_1.pt', progress=True)
    cnn_ckpoint = torch.hub.load_state_dict_from_url('https://github.com/SimmerChan/hub_test/releases/download/v1.0/best_multi_label_encoder_weight_16.pt', progress=True)
    print(type(s2s_ckpoint))
    print(type(cnn_ckpoint))
    return Image2PoemGenerator(BytesIO(s2s_ckpoint), BytesIO(cnn_ckpoint))