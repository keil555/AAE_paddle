
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as transforms
from paddle.vision import datasets
from paddle.vision.datasets import MNIST
from paddle.io  import random_split
from paddle.io import DataLoader
import pickle

N = 1200
STD = 5
z_dim = 8

def reparameterization(mu, logvar):
    std = paddle.exp(logvar / 2)
    sampled_z = paddle.normal(0, STD, (mu.shape[0], z_dim))
    z = sampled_z * std + mu
    return z

class Encoder(nn.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1*28*28, N),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.mu = nn.Linear(N, z_dim)
        self.logvar = nn.Linear(N, z_dim)
        self.direct = nn.Linear(N, z_dim)

    def forward(self, img):
        img_flat = paddle.reshape(img, shape = (img.shape[0], -1) )
        # 编码输出
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, N),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, 1*28*28),
            nn.Sigmoid(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = paddle.reshape(img_flat, shape = [img_flat.shape[0], 1,28,28] )
        return img


class Discriminator(nn.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, N),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.Dropout(0.2),
            nn.ReLU( ),
            nn.Linear(N, 1),
            nn.Sigmoid(),
        )              

    def forward(self, z):
        validity = self.model(z)
        return validity