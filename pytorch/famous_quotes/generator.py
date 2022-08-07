import torch
import random
from torch import nn


class Generator(nn.Module):
    """RNN Generator to generate quotes"""
    def __init__(self, input_size, hidden_size, classes, auth_size,
                 num_layers=1, bidirectional=False):
        super(Generator, self).__init__()
        
        self.auth_size = auth_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size,
                          num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size, classes)
        self.activation = nn.Softmax(dim=1)
        
    def forward(self, inputs, hiddens):
        inp = inputs.clone()
        r, hiddens = self.rnn(inp.clone(), hiddens.clone())
        a = self.linear(r.clone())
        o = self.activation(a.clone())
        return o, hiddens
        
    def initialize_hidden(self):
        return torch.ones(1, self.hidden_size)

    def generate_random(self):
        return torch.rand(1, self.hidden_size)

    def generate_auth(self):
        idx = random.randint(0, self.auth_size)
        auth_tens = torch.zeros(self.auth_size)
        auth_tens[idx] = 1
        return auth_tens