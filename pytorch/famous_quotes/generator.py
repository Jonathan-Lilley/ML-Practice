import torch
from torch import nn


class Generator(nn.Module):
    """RNN Generator to generate quotes"""
    def __init__(self, input_size, hidden_size, classes,
                 num_layers=2, bidirectional=False):
        super(Generator, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size,
                          num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size, classes)
        self.activation = nn.Softmax()
        
    def forward(self, inputs, hiddens):
        o = inputs
        o, hiddens = self.rnn(o, hiddens)
        o = self.linear(o)
        o = self.activation(o)
        return o, hiddens
        
    def initialize_hidden(self):
        return torch.ones(self.hidden_size)
