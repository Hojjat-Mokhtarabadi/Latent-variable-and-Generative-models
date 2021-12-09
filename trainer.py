
from tqdm import tqdm


class Trainer():
    def __init__(self, model, train_loader, test_loader, optimizer, loss, device):
        self._model = model.to(device)
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._optimizer = optimizer
        self._loss = loss
        self._device = device

    def train_step(self):
        self._model.train()

        for (X, y) in tqdm(self._train_loader):
            X, y = X.to(self._device), y.to(self._device)

            y_hat, z = self._model(X)

            batch_size = X.shape[0]

            y = X.view(batch_size, -1)
            loss = self._loss(y_hat, y)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def test_step(self):
        pass

    def fit(self, epochs=2):
        for e in range(epochs):
            print(e)
            self.train_step()
