import torch
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import torchvision.utils as v
from utils import *


class AETrainer:
    def __init__(self, model, optimizer, train_loader, test_loader, device):
        self._model = model.to(device)
        self._criterian = model.criterian
        self._optimizer = optimizer
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._device = device

    def train_step_ae(self):
        self._model.train()
        train_loss = AverageMeter()
        for idx, (x, _) in enumerate(self._train_loader):
            x = x.to(self._device)
            model_preds = self._model(x)
            loss = self._criterian(target=x, **model_preds)
            # print(loss)
            # print("in train")
            # print(x.shape)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            train_loss.update(loss.item(), 1)

        return train_loss.avg()

    def fit_ae(self, fit_params):
        if fit_params['resume'] is True:
            pass
        max_epoch = fit_params['max_epoch']
        print("Start training...")
        for e in trange(max_epoch):
            train_loss_avg = self.train_step_ae()

            if e % 1 == 0:
                print(
                    f"Epoch {e} [Train Loss: {train_loss_avg}]")
                checkpoint = {
                    'model_state': self._model.state_dict(),
                    'oprimizer_state': self._optimizer.state_dict(),
                    'epoch': e
                }
                torch.save(checkpoint, '../ae_checkpoint.pth')

            if e == max_epoch-1:
                eval_generative_model(9, self._model)
