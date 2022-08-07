from doctest import OutputChecker
from torch import nn


class Discriminator(nn.Module):
    """Discriminator"""
    def __init__(self, input_size, hidden_size=100, num_layers=3):
        super(Discriminator, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                            for ii in range(num_layers-2)])
        self.output_layer = nn.Linear(hidden_size, 2)
        self.hidden_activ = nn.Sigmoid()
        self.out_activ = nn.Sigmoid()

    def forward(self, inputs):
        o = inputs
        o = self.input_layer(o.clone())
        o = self.hidden_activ(o.clone())
        if any(self.hidden_layers):
            for layer in self.hidden_layers:
                o = layer(o.clone())
                o = self.hidden_activ(o.clone())
        o = self.output_layer(o.clone())
        o = self.out_activ(o.clone())
        return o
        
    