import torch
from torch import nn
import torch.nn.functional as F


class VanilllaVAE(nn.Module):
    def __init__(self, im_shape, im_size, **configs):
        super().__init__()

        self.ch, self.h, self.w = im_shape
        self.latent_dim = configs['latent_dim']
        self.kernel_size = configs['kernel_size']
        self.hid_dims = [32, 64, 128]
        self.final_hid_dim = 128

        # compute final spatial dims after convolution layers
        for _ in self.hid_dims:
            self.h = self.h - self.kernel_size + 1
            self.w = self.w - self.kernel_size + 1

        self.enc_out_size = self.final_hid_dim * self.h * self.w

        # encoder
        enc_modules = []
        in_ch = self.ch
        for dim in self.hid_dims:
            enc_modules.append(self._conv(in_ch, dim))
            in_ch = dim

        self.enc = nn.Sequential(*enc_modules)

        # guassian params
        self.mue = nn.Linear(self.enc_out_size, self.latent_dim)
        self.logvar = nn.Linear(self.enc_out_size, self.latent_dim)

        # projection from latent dim to original dim
        self.sample_proj = nn.Linear(self.latent_dim, self.enc_out_size)

        # decoder
        dec_modules = []
        hid_dims = self.hid_dims[::-1]
        for dim_idx, dim in enumerate(hid_dims[:-1]):
            dec_modules.append(self._deconv(dim, hid_dims[dim_idx+1]))

        dec_modules.append(nn.ConvTranspose2d(hid_dims[-1], self.ch,
                                              self.kernel_size))

        self.dec = nn.Sequential(*dec_modules)

    def _conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch,
                      self.kernel_size),
            nn.ReLU())

    def _deconv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch,
                               self.kernel_size),
            nn.ReLU())

    def gaussain_params(self, p_z):
        """
        it is considered all samples distribution is gaussian and so is latent space.

        shapes:
            p_z -> (bs, ch, h, w)
            p_z flatten -> (bs, ch*h*w)
            mue = logvar -> (bs, latent_dim)
        """
        p_z = torch.flatten(p_z, start_dim=1)
        mue = self.mue(p_z)
        logvar = self.logvar(p_z)
        return mue, logvar

    def _z(self, mue, logvar):
        """ 
        in order to the stochasticity of sampling we can not update the corresponding 
        parameters, thus we apply reparameterazation trick which lets us compute gradients 
        as well as injecting stochastisity.

        shapes:
            eps -> (bs, latent_dim)
            mue -> (bs, latent_dim)
            logvar ->(bs, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = eps * std + mue

        return sample

    def criterian(self, target, **kwargs):
        """
        KL-Divergance between a normal distribution and multivariate distribution is:
            0.5 * sum(sigma * mue^2 - 1 - Ln(sigma)).

        shapes:
            reconst_x = x -> (bs, ch, h, w)
            mue = logvar  -> (bs, latent_dim)
        """
        mue = kwargs['mue']
        logvar = kwargs['logvar']
        reconst_x = kwargs['reconst_x']
        x = target
        mse = F.mse_loss(reconst_x, x)
        kl_div = (0.5 * (logvar.exp() + (mue ** 2) -
                  1 - logvar).sum(dim=1)).mean(dim=0)

        loss = mse + kl_div
        return loss
        # F.kl_div

    def sample(self, num_samples):
        sample = self.sample_proj(torch.randn(num_samples, self.latent_dim))
        sample = sample.view(-1, self.final_hid_dim, self.h, self.w)
        generated = self.dec(sample)
        return generated

    def reconstruct(self, x):
        return self.forward(x)['reconst_x']

    def forward(self, x):
        enc_out = self.enc(x)
        mue, sigma = self.gaussain_params(enc_out)
        z = self.sample_proj(self._z(mue, sigma))
        reconst = self.dec(z.view(-1, self.final_hid_dim, self.h, self.w))

        return {'reconst_x': reconst, 'mue': mue, 'logvar': sigma}
