import torch
import matplotlib.pyplot as plt
import torchvision.utils as v


class AverageMeter:
    def __init__(self):
        self.value = 0
        self.num = 0

    def update(self, value, num):
        self.value += value
        self.num += num

    def avg(self):
        return self.value/self.num


@torch.no_grad()
def on_validation_end(size, model, loader):
    model = model.to('cpu')
    test_sample = next(iter(loader))[0]
    # generate
    # gen = model.sample(size).detach().cpu()
    # reconstruct
    rec = model.reconstruct(test_sample[:size]).detach().cpu()
    nrow = int(size ** 0.5)
    pd = 2
    real_grid = v.make_grid(test_sample[:size], padding=pd, nrow=nrow)
    rec_grid = v.make_grid(rec, padding=pd, nrow=nrow)
    # gen_grid = v.make_grid(gen, padding=pd, nrow=nrow)

    plt.subplot(131), plt.imshow(
        real_grid.permute(1, 2, 0)), plt.title('real')
    plt.subplot(132), plt.imshow(
        rec_grid.permute(1, 2, 0)), plt.title("rec")
    # plt.subplot(133), plt.imshow(
    #     gen_grid.permute(1, 2, 0)), plt.title("gen")

    plt.show()
