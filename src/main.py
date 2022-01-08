import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from torch import optim
from models import *
from dataset.dataset import *
from utils import *


def main(configs, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_args = configs['exp_params']
    model_args = configs['model_params']
    fit_args = configs['trainer_params']

    print("Preparing data...")
    if exp_args['dataset'] == "MNIST":
        mnist_ds = MNISTDS(exp_args['data_path'])
        train_set, test_set, data_shape, data_size = mnist_ds()
        # reshape = data_size if exp_args['sample_reshape'] else data_shape
    elif exp_args['dataset'] == "CIFAR10":
        cifar10_ds = CIFAR10DS(exp_args['data_path'])
        train_set, test_set, data_shape, data_size = cifar10_ds()
    else:
        print("enter your dataset")

    train_loader = DataLoader(train_set, batch_size=exp_args['batch_size'],
                              shuffle=True, num_workers=exp_args['num_workers'])
    test_loader = DataLoader(test_set, batch_size=exp_args['batch_size'],
                             shuffle=True, num_workers=exp_args['num_workers'])

    print("Preparing model...")
    # configuration
    model_args['device'] = device
    model = MODELS_[model_args['name']](
        data_shape, data_size, **model_args)
    print(model)
    if args.ae:
        # ### AE family settings
        optimizer = optim.Adam(model.parameters(), lr=exp_args['lr'])
        if args.eval:
            eval(model, test_loader)
            return

        trainer = AETrainer(
            model,
            optimizer,
            train_loader,
            test_loader,
            device)
        trainer.fit_ae(fit_args)

    elif args.gan:
        # ### GAN family settings
        gan_model = model
        gen_optim = optim.Adam(
            gan_model.generator.parameters(), lr=exp_args['lr'])
        disc_optim = optim.Adam(
            gan_model.discriminator.parameters(), lr=exp_args['lr'])
        if args.eval:
            eval(gan_model.generator, test_loader)
            return

        trainer = GANTrainer(
            gan_model,
            gen_optim,
            disc_optim,
            train_loader,
            test_loader,
            device)
        trainer.fit_gan(**fit_args)
    else:
        raise ValueError("Model family must be defined")

# def eval(model):
#     model.sample()


if __name__ == '__main__':
    parse = argparse.ArgumentParser("Latent Varaible and Generative Models")
    parse.add_argument("-c",
                       "--config",
                       dest="configuration_file",
                       metavar="FILE",
                       help='path to the config file',
                       default='configs/conv_ae.yaml')
    parse.add_argument("--eval", action='store_true')
    parse.add_argument("--gan", action='store_true')
    parse.add_argument("--ae", action='store_true')
    args = parse.parse_args()
    with open(args.configuration_file, 'r') as f:
        try:
            configs = yaml.safe_load(f)
        except yaml.YAMLError as y:
            print(y)

    main(configs, args)
