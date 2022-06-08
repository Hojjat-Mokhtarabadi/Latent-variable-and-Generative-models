import torch
from tqdm import trange
from utils import AverageMeter
from utils.utils import eval_generative_model


class GANTrainer:
    def __init__(self, model, g_optimizer, d_optimizer, train_loader, test_loader,  device):
        self._model = model.to(device)
        self._criterian = model.criterian
        self._g_optim = g_optimizer
        self._d_optim = d_optimizer
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._device = device

    def train_step_gan(self):
        self._model.train()
        avg_d_loss = AverageMeter()
        avg_g_loss = AverageMeter()
        for x, _ in self._train_loader:
            x = x.to(self._device)
            y_real = 1
            y_fake = 0
            #* #### D
            y = torch.cat([torch.ones(x.size(0), device=self._device),
                          torch.zeros(x.size(0), device=self._device)])
            disc_pred = self._model.discriminator_phase(x)
            d_kwargs = {'phase': 'D', 'pred': disc_pred}
            disc_loss = self._criterian(target=y, **d_kwargs)

            self._d_optim.zero_grad()
            disc_loss.backward()
            self._d_optim.step()

            #* #### G
            y = torch.ones(x.size(0), device=self._device)
            gen_pred = self._model.generator_phase(x)
            g_kwargs = {'phase': 'G', 'pred': gen_pred}
            gen_loss = self._criterian(target=y, **g_kwargs)

            self._g_optim.zero_grad()
            gen_loss.backward()
            self._g_optim.step()

            avg_d_loss.update(disc_loss.item(), 1)
            avg_g_loss.update(gen_loss.item(), 1)

        return avg_g_loss.avg(), avg_d_loss.avg()

    def fit_gan(self, **fit_params):
        max_epoch = fit_params['max_epoch']
        print("Start training...")
        for e in trange(max_epoch):
            g_loss, d_loss = self.train_step_gan()

            if e % 1 == 0:
                print(
                    f"Epoch {e} [D Train Loss: {d_loss}, G Train Loss: {g_loss}]")
                checkpoint = {
                    'model_state': self._model.state_dict(),
                    'd_oprimizer_state': self._d_optim.state_dict(),
                    'g_optimizer_state': self._g_optim.state_dict(),
                    'epoch': e
                }
                torch.save(checkpoint, '../gan_checkpoint.pth')

            if e == max_epoch-1:
                eval_generative_model(9, self._model)
