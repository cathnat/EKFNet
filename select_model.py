import numpy as np
import torch
from compare_model.NovelCNN import NovelCNN
from compare_model.GCTNet import Generator, Discriminator
from compare_model.ResCNN import ResCNN
from compare_model.EEGDnet import DeT
from compare_model.DuoCL import DuoCL

device = torch.device("cpu")
def pick_models(netname):
    if netname == 'NovelCNN':
        model = NovelCNN().to(device)
    elif netname == 'GCTNet':
        G = Generator().to(device)
        D = Discriminator().to(device)
        return G, D
    elif netname == 'EEGDnet':
        model = DeT(seq_len=512, patch_len=64, depth=6, heads=1)
    elif netname == 'DuoCL':
        model = DuoCL()
    elif netname == 'ResCNN':
        model = ResCNN()
    return model