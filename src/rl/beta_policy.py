
from .neural_network import NeuralNetwork
from .policy import Policy
import torch
from torch.distributions import Beta
import torch.nn.functional as F

class BetaPolicy(Policy):

    def __init__(self, network : NeuralNetwork):
        super().__init__(network)
        self._alpha = None
        self._beta = None

    def make_init_copy(self):
        return BetaPolicy(NeuralNetwork.from_other(self.neural_network))

    def get_dist(self, state: torch.Tensor):
        y = self.neural_network(state)
        y = (F.softplus(y) + 2).clamp(max=100)
        self._alpha, self._beta = torch.chunk(y, 2, dim=-1)
        return Beta(self._alpha, self._beta)

    def sample_with_log_prob(self, state: torch.Tensor):
        dist = self.get_dist(state)
        raw_action = dist.sample()
        action = self.scale_action(raw_action)
        log_prob = dist.log_prob(raw_action.clamp(1e-4, 1-1e-4)).sum()
        return action, log_prob

    def sample(self, state : torch.Tensor):
        dist = self.get_dist(state)
        raw_action = dist.sample()
        action = self.scale_action(raw_action)
        return action

    def get_statistics(self):
        return {"mean(abs(alpha-beta))": (self._alpha - self._beta).abs().mean().detach().item(),
                "mean(alpha+beta)": (self._alpha + self._beta).mean().detach().item()}

    def scale_action(self, action):
        return (action - 0.5) * 2

    def unscale_action(self, action):
        return (action + 1) / 2

