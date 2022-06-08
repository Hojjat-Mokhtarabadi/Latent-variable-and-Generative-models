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
        data_shape = (1, 28, 28)
        data_size = 1 * 28 * 28
        return self._train_set, self._test_set, data_shape, data_size


class CIFAR10DS():
    def __init__(self, root):
        self._train_set = datasets.CIFAR10(
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
        data_shape = (3, 32, 32)
        data_size = 3 * 32 * 32
        return self._train_set, self._test_set, data_shape, data_size


class CelebADS():
    def __init__(self, root):
        self._train_set = datasets.CelebA(
            root=root, 
            split='train',
            download=True,
            Transform=transforms.ToTensor()
        )
        self._test_set = datasets.CelebA(
            root=root,
            split='validation',
            download=True, 
            transform=transforms.ToTensor()
        )
        
    def __call__(self):
        data_shape = (3, 32, 32)
        data_size = 3 * 32 * 32
        return self._train_set, self._test_set, data_shape, data_size

class CustomDS(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
