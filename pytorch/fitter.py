import torch
from torch import nn
from pytorch_lookups import OPTIMIZERS, LOSSFNS


class Fitter:

    def __init__(self, model, epochs=1, batch_size=1,
                 loss_fn="mse", optimizer="sgd", optimparams=(), lr=0.1,
                 device="cpu"):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = device

        self.loss = LOSSFNS[loss_fn.lower()]
        self.optimizer = OPTIMIZERS[optimizer.lower()](model.parameters(), lr,
                                                       *optimparams)

    def _backward(self, pred, y):
        loss = self.loss(pred, y)
        self.optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        return loss

    def fit(self):
        self.model.train()
        # batches = self.gen_batches(batch_size)
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1} of {self.epochs}")
            loss_ttl = 0
            for batch_n, (y, X) in enumerate(self.model.train_ds):
                pred = self.model(X)
                loss_ttl += self._backward(pred, y).item()
            print(f"Loss: {loss_ttl/self.model.train_data_count}")
