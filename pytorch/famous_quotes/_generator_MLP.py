import torch
from torch import nn


class Generator(nn.Module):
    """Generator"""
    def __init__(self, rand_size, hidden_size, output_size, auth_count,
                 num_layers=3):
        super(Generator, self).__init__()
        self.rand_size = rand_size
        self.auth_count = auth_count
        self.input_layer = nn.Linear(rand_size + 1, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                            for ii in range(num_layers-2)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activ = nn.Tanh()
    
    def forward(self):
        o = torch.cat((self._generate_inp(), self._generate_auth()), axis=-1)
        o = self.input_layer(o)
        o = self.activ(o)
        if any(self.hidden_layers):
            for layer in self.hidden_layers:
                o = layer(o)
                o = self.activ(o)
        o = self.output_layer(o)
        o = self.activ(o)
        return o

    def _generate_inp(self):
        return torch.rand(self.rand_size)
    
    def _generate_auth(self):
        return torch.randint(self.auth_count, (1,)).type(torch.long)