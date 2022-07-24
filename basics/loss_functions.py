import numpy as np


class Loss:

    def __init__(self, loss_name):
        self.loss_name = loss_name
        loss_fns = {
            "mse": self._mse_loss(),
            "ce": self._ce_loss()
        }
        self.loss, self.d_loss = loss_fns.get(loss_name, loss_fns["mse"])

    def _mse_loss(self):
        def loss(target, predicted, data_size):
            return np.power(target - predicted, 2) / (2 * data_size)

        def d_loss(target, predicted, datasize):
            return (target - predicted) / datasize
        return loss, d_loss


    def _nll_loss(self):
        def loss():
            pass

        def d_loss():
            pass
        return loss, d_loss


    def _svm_loss(self):
        def loss():
            pass

        def d_loss():
            pass
        return loss, d_loss


    def _ce_loss(self):
        def loss(target, predicted):
            return - sum(target * np.log(predicted))

        def d_loss(target, predicted):
            return target - predicted
        return loss, d_loss