# encoding=utf-8

from generator import Image2PoemGenerator
import torch

dependencies = ["torch", "torchvision"]


def model():
    s2s_ckpoint = torch.hub.load_state_dict_from_url('checkpoint', progress=False)
    cnn_ckpoint = torch.hub.load_state_dict_from_url('checkpoint', progress=False)
    return Image2PoemGenerator(s2s_ckpoint, cnn_ckpoint)