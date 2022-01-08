import torch
from torch import nn
from torch import random
import torch.nn.functional as F


class VanillaGAN(nn.Module):
    def __init__(self, im_shape, im_size, **configs):
        super().__init__()

        self.noise_dim = configs['noise_dim']
        self.img_dim = im_size
        self.im_shape = im_shape
        self.device = configs['device']

        self.generator = Generator(self.noise_dim, self.img_dim)
        self.discriminator = Discriminator(self.img_dim)

    def generator_phase(self, img):
        bs = img.size(0)
        random_noise = self._random_noise((bs, self.noise_dim))
        generated_img = self.generator(random_noise)
        pred = self.discriminator(generated_img)

        return pred

    def discriminator_phase(self, img):
        bs = img.size(0)
        img = img.view(bs, -1)
        random_noise = self._random_noise((bs, self.noise_dim))
        generated_img = self.generator(random_noise)

        # print(img.shape)
        # print(generated_img.shape)

        img_batch = torch.cat([img, generated_img])
        pred = self.discriminator(img_batch)

        return pred

    def sample(self, num):
        noise = self._random_noise((num, self.noise_dim))
        generated_img = self.generator(noise)
        return generated_img.view(-1, *(self.im_shape))

    def _random_noise(self, size):
        if (next(self.generator.parameters())).is_cuda:
            noise = torch.randn(size, device='cuda')
        else:
            noise = torch.randn(size, device='cpu')
        return noise

    def criterian(self, target, **kwargs):
        # phase = kwargs['phase']
        pred = kwargs['pred']

        loss = F.binary_cross_entropy_with_logits(pred.squeeze(), target)
        return loss


class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.img_dim = img_dim

        self.gen_model = nn.Sequential(
            *get_linear_clone(3, self.noise_dim, self.img_dim, 512))

    def forward(self, x):
        generated_img = self.gen_model(x)
        return generated_img


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.img_dim = img_dim

        self.disc_pred = nn.Sequential(
            nn.Flatten(),
            *get_linear_clone(3, self.img_dim, 1, 512))

    def forward(self, x):
        disc_pred = self.disc_pred(x)
        return disc_pred


def get_linear_clone(layers, in_dim, out_dim, hidden_dim):
    modules = []
    for _ in range(layers-1):
        modules.append(nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()))
        in_dim = hidden_dim

    modules.append(nn.Linear(hidden_dim, out_dim))
    return modules
