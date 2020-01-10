# encoding=utf-8

from generator import Image2PoemGenerator
import torch

dependencies = ["torch", "torchvision"]


def model():
    s2s_ckpoint = torch.hub.load_state_dict_from_url('https://github.com/SimmerChan/hub_test/releases/download/v1.0/best_img2poem_transformer_wo_kws_1.pt', progress=True)
    cnn_ckpoint = torch.hub.load_state_dict_from_url('https://github.com/SimmerChan/hub_test/releases/download/v1.0/best_multi_label_encoder_weight_16.pt', progress=True)
    return Image2PoemGenerator(s2s_ckpoint, cnn_ckpoint)