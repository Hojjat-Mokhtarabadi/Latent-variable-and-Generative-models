import yaml
import argparse

from trainer import Trainer
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

from models.auto_encoder import VanillaAE, ConvAE
from dataset.dataset import *

parse = argparse.ArgumentParser("Latent Varaible and Generative Models")
parse.add_argument("--config",
                   dest="filename",
                   metavar="FILE",
                   help='path to the config file',
                   default='configs/vanilla_ae.yaml')

args = parse.parse_args()

with open(args.filename, 'r') as f:
    try:
        configs = yaml.safe_load(f)
        exp_args = configs['exp_params']
        model_args = configs['model_params']
        trainer_args = configs['trainer_params']
    except yaml.YAMLError as y:
        print(y)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# print("----Prepare Data----")
if exp_args['dataset'] == "MNIST":
    mnist_ds = MNISTDS(exp_args['data_path'])
    train_set, test_set = mnist_ds()
else:
    print("enter your dataset")

train_loader = DataLoader(train_set, batch_size=exp_args['batch_size'],
                          shuffle=True, num_workers=exp_args['num_workers'])
test_loader = DataLoader(test_set, batch_size=exp_args['batch_size'],
                         shuffle=True, num_workers=exp_args['num_workers'])

# print("----Model----")
# configuration
model = VanillaAE(latent_dim=model_args['latent_dim'])
optimizer = optim.Adam(model.parameters(), lr=exp_args['lr'])
criterian = nn.MSELoss()
trainer = Trainer(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterian,
    device)


if __name__ == '__main__':
    print(model)
    trainer.fit(trainer_args['max_epoch'])
