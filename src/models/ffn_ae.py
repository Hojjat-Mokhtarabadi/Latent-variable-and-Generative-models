from itertools import count
import torch
from torch import nn
from torch.nn.functional import mse_loss


class FFNAe(nn.Module):
    def __init__(self, im_shape, im_size, **configs):
        super(FFNAe, self).__init__()

        self.latent_dim = configs['latent_dim']
        self.num_layers = configs['num_layers']
        self.im_size = im_size
        self.im_shape = im_shape

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.im_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.im_size)
        )

    def reconstruct(self, x):
        img = self.decoder(self.encoder(x))
        img = img.view(x.size(0), *(self.im_shape))
        return img

    def criterian(self, target, **kwargs):
        pred = kwargs['pred']
        batch_size = target.size(0)
        target = target.view(batch_size, -1)
        return mse_loss(pred, target)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)

        return {'pred': out, 'latent': z}
