from __future__ import annotations
from typing import Tuple, List

import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):

    # There will be len(layer_dimensions)-1 linear layers in the network.
    # layer_dimensions[0] is input size, and layer_dimensions[-1] is output size
    def __init__(self, layer_dimensions : Tuple = (106, 256, 256, 58)):
        super().__init__()
        self.layer_dimensions = layer_dimensions
        self.linear_layers : nn.ModuleList = nn.ModuleList([nn.Linear(layer_dimensions[i], layer_dimensions[i+1])
            for i in range(len(layer_dimensions)-1)])

    @classmethod
    def from_other(cls, other : NeuralNetwork):
        network = NeuralNetwork(other.layer_dimensions)
        network.load_state_dict(other.state_dict())
        return network

    def forward(self, x : torch.Tensor):
        for linear_layer in self.linear_layers[:-1]:
            x = torch.relu(linear_layer(x))

        x = self.linear_layers[-1](x)
        return x


