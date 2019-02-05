# Structures of autoencoders. You can add (and try) VAE, AAE and DAAE too.

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# This is a simple autoencoder to try. You can add other classes here.

class AutoEncoder(nn.Module):
    def __init__(self, nz=2):
        super(AutoEncoder, self).__init__()
        self.nz = nz

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, nz),   # compress to 2 features to get 2D repr.
        )
        self.decoder = nn.Sequential(
            nn.Linear(nz, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # compress to get an image between (0, 1)
        )

        self.useCUDA = torch.cuda.is_available()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def lossfunc(self, decoded, original):
        loss = nn.MSELoss()   # you might want to try BCE too
        return loss(decoded, original)
