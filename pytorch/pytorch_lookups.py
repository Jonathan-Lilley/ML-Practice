from torch import nn
from torch import optim


class LinearActivation(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor


ACTIVATIONS = {
    "lrelu": nn.LeakyReLU,
    "multihead_attn": nn.MultiheadAttention,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "linear": LinearActivation
}

LOSSFNS = {
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
}

OPTIMIZERS = {
    "sgd": optim.SGD,
}