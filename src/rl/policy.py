from __future__ import annotations
from abc import ABC, abstractmethod

from .neural_network import NeuralNetwork


class Policy(ABC):

    def __init__(self, network : NeuralNetwork):
        self.neural_network = network

    @abstractmethod
    def make_init_copy(self):
        raise NotImplemented

    @abstractmethod
    def sample(self, state):
        raise NotImplemented

    @abstractmethod
    def sample_with_log_prob(self, state):
        raise NotImplemented
