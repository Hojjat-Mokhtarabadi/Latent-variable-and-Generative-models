from .conv_ae import ConvAE
from .vanilla_vae import VanilllaVAE
from .ffn_ae import FFNAe
from .vanilla_gan import Generator, Discriminator, VanillaGAN


MODELS_ = {
    'FFN_AE': FFNAe,
    'Conv_AE': ConvAE,
    'Vanilla_VAE': VanilllaVAE,
    'Vanilla_GAN': VanillaGAN
}
