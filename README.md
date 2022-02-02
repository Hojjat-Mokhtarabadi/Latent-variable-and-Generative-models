# Latent variable and Generative models
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Pytorch 1.9](https://img.shields.io/badge/pytorch-1.9-orange.svg)](https://pytorch.org/)
[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/Hojjat-Mokhtarabadi/Latent-variable-and-Generative-models)

The collection of different latent variable and generative models. This repository contains two popular family of generative models: VAEs and GANs

### Variational Auto-Encoder [(VAE)](https://arxiv.org/abs/1312.6114)
In simple words, variational autoencoder is a kind of 'Approximate Density Estimation' methods which try to transform the complex high-dimenstional input distibution to a tracktable and known dirstribution. The basic assumption in VAE is that eveything is Gaussain, thus the encoder part maps evey input to a normal dirstribution, then a random sample from normal distribution is passed through decoder net.
To generate new samples we only need to sample from normal dirstribution and pass it through decoder

### Generative Adversarial Net [(GAN)](https://arxiv.org/abs/1406.2661)
Unlike the other method intoduced above, in this type of models we actully want to learn the complex high-dimentional distribution of data but the there is no a straight way to do that so instead, let's learn the transformation from a random noise to the distribution of data. The current model consists of two networks: Generator and Discriminator and the objective is formed as a 'Mini-Max' game in which every network tries to fool the other one and improve itself.

Some intuitive resources on VAE and GAN:
- [Generative Models - Stanford](https://www.youtube.com/watch?v=5WoItGTWV54)
- [GAN - Ali Ghodsi](https://www.youtube.com/watch?v=7G4_Y5rsvi8)
- [Understanding Generative Adversarial Networks](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29)
- [VAE - Ali Ghodsi](https://www.youtube.com/watch?v=uaaqyVS9-rM)
- [Understanding Variational Autoencoders](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)


### Requirements
- Python = 3.7
- Pytorch = 1.9


### Install
```bash
git clone https://github.com/Hojjat-Mokhtarabadi/Latent-variable-and-Generative-models.git
cd Latent-variable-and-Generative-models
python3 -m pip install -r requirements.txt
```

### Train
Note: the family type and configuration file should be specified in`run.sh`
```bash
bash run.sh
```



