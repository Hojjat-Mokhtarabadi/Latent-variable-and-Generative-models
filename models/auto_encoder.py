import torch
from torch import nn


class VanillaAE(nn.Module):
    def __init__(self, num_layers=4, im_shape=28, latent_dim=128):
        super(VanillaAE, self).__init__()
        self._latent_dim = latent_dim
        self._num_layers = num_layers
        self._im_shape = 28*28

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._im_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self._latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self._latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self._im_shape)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)

        return out, z


class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        pass
