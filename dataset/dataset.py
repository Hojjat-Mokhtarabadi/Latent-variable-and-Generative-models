from torch.utils.data import Dataset, dataset
from torchvision import datasets, transforms


class MNISTDS():
    def __init__(self, root):
        self._train_set = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        self._test_set = datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

    def __call__(self):
        return self._train_set, self._test_set


class CustomDS(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
