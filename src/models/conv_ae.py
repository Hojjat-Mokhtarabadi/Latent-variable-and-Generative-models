from torch import nn
from torch.nn.functional import mse_loss


class ConvAE(nn.Module):
    def __init__(self, im_shape, im_size, **configs):
        super(ConvAE, self).__init__()

        self.x_dim = im_shape
        self.latent_dim = configs['latent_dim']
        self.num_filters = configs['num_filters']
        self.filter_size = 5

        channels, w, h = self.x_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, self.num_filters,
                      kernel_size=self.filter_size),
            nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters*2,
                      kernel_size=self.filter_size),
        )

        enc_final_dim = self.num_filters * 2 * 20 * 20
        self.latent_space = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_final_dim, self.latent_dim),
            nn.Linear(self.latent_dim, enc_final_dim),
            nn.Unflatten(dim=-1, unflattened_size=(self.num_filters*2, 20, 20))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.num_filters*2, self.num_filters, kernel_size=self.filter_size),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_filters, channels,
                               kernel_size=self.filter_size)
        )

    def criterian(self, target, **kwargs):
        pred = kwargs['pred']
        return mse_loss(pred, target)

    def reconstruct(self, img):
        rec = self.decoder(self.latent_space(self.encoder(img)))
        return rec

    def forward(self, x):
        z = self.latent_space(self.encoder(x))
        out = self.decoder(z)

        return {'pred': out, 'latent': z}
