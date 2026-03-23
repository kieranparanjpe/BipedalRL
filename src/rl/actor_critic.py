from __future__ import annotations

import os

import torch
import pandas as pd
from torch import Tensor

from .hyperparameters import Hyperparameters
from .environment import Environment
from .neural_network import NeuralNetwork
from .policy import Policy
from .reward import Reward
from typing import TYPE_CHECKING, Optional, Sequence, Any

if TYPE_CHECKING:
    from src.robot.robot import Robot


def cap_tensor_magnitude_(x: torch.Tensor, max_norm: float, eps: float = 1e-8) -> torch.Tensor:
    """
    In-place: scales x down only if ||x|| > max_norm (preserves direction). In place on x.
    """
    norm = x.norm()
    if norm > max_norm:
        x.mul_(max_norm / (norm + eps))
    return x

def cap_tensors_global_magnitude_(
    tensors: Sequence[torch.Tensor],
    max_norm: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    In-place global-norm cap across a list/tuple of tensors.
    Scales all tensors by the same factor if global norm exceeds max_norm.
    Returns the pre-cap global norm.
    """
    if not tensors:
        return torch.tensor(0.0)

    device = tensors[0].device
    total_squared = torch.zeros((), device=device)
    for t in tensors:
        total_squared = total_squared + t.pow(2).sum()

    total_norm = total_squared.sqrt()
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for t in tensors:
            t.mul_(scale)

    return total_norm


class ActorCritic:

    def __init__(self, environment : Environment, policy : Policy, value_function : NeuralNetwork,
                 reward : Reward, robot : Robot,
                 hyperparams : Optional[Hyperparameters] = None):
        self.environment = environment
        self.policy_1 = policy # used for inference
        self.value_function_1 = value_function # used for inference
        self.value_function_2 = NeuralNetwork.from_other(value_function) # used for bootstrapping
        self.reward = reward
        self.robot = robot
        self.hyperparams = hyperparams if hyperparams is not None else Hyperparameters()

        self.value_eligibility_trace = [torch.zeros_like(p) for p in value_function.parameters()]
        self.policy_eligibility_trace =[torch.zeros_like(p) for p in policy.neural_network.parameters()]

        self.episodeStatistics = []
        self.timestepStatistics = []
        self.total_timesteps = 0

    def reset_eligibility_traces(self):
        torch._foreach_zero_(self.value_eligibility_trace)
        torch._foreach_zero_(self.policy_eligibility_trace)

    def train_episode(self, environment):
        environment.reset()
        self.reset_eligibility_traces()
        self.reward.reset_episode()

        bootstrapping_value_function = self.value_function_1 if self.hyperparams.value_function_changeout is None \
            else self.value_function_2

        decay = 1
        total_reward = 0
        timesteps = 0
        while environment.is_running():
            # if we are using 2 value functions, update learned params only every n steps.
            if self.hyperparams.value_function_changeout is not None:
                if self.total_timesteps % self.hyperparams.value_function_changeout == 0:
                    self.value_function_2.load_state_dict(self.value_function_1.state_dict())

            # Get current state S
            current_state = torch.from_numpy(self.robot.get_state_sin_cos_no_accel()).float()
            # Sample best action A, based on S. Also get the log probability of A, ln[P(A)]
            action, log_prob_policy = self.policy_1.sample_with_log_prob(current_state)

            # Take action A
            self.robot.set_ctrls(action.detach().cpu().numpy())

            environment.step() # Step the environment.

            # Find next state  after stepping env. S'
            next_state = torch.from_numpy(self.robot.get_state_sin_cos_no_accel()).float()

            # Get next reward
            reward = torch.tensor(self.reward.reward(), dtype=torch.float32).detach()
            total_reward += reward.item()
            terminal = self.reward.is_terminal()

            value_function_current_state, value_function_next_state = self.estimate_value_function(
                bootstrapping_value_function, current_state, next_state
            )

            grad_value_function_current_state, log_grad_policy_current_state = self.calculate_gradients(
                log_prob_policy, value_function_current_state
            )

            td_error = self.calculate_td_error(reward, value_function_current_state,
                value_function_next_state, terminal)

            self.update_eligibility_traces(decay, grad_value_function_current_state, log_grad_policy_current_state)

            # update weights
            self.update_weights(td_error)

            decay *= self.hyperparams.discount_factor_decay if self.hyperparams.discount_factor_decay is not None \
                else self.hyperparams.discount_factor


            self.timestepStatistics.append(
                {
                    "timestep": self.total_timesteps,
                    "abs(td error)": abs(td_error.item()),
                    "reward": reward.item()
                }
                | self.policy_1.get_statistics())
            self.total_timesteps += 1
            timesteps += 1
            if terminal:
                break

        return total_reward / timesteps

    def update_weights(self, td_error: Tensor):
        with torch.no_grad():
            value_params = list(self.value_function_1.parameters())
            policy_params = list(self.policy_1.neural_network.parameters())

            td = td_error.detach()  # stays on device

            value_deltas = torch._foreach_mul(self.value_eligibility_trace, self.hyperparams.value_learning_rate)
            # need [] * len for broadcasting
            value_deltas = torch._foreach_mul(value_deltas, [td] * len(value_deltas))
            cap_tensors_global_magnitude_(value_deltas, self.hyperparams.max_value_weight_update)
            torch._foreach_add_(value_params, value_deltas)

            policy_deltas = torch._foreach_mul(self.policy_eligibility_trace, self.hyperparams.policy_learning_rate)
            policy_deltas = torch._foreach_mul(policy_deltas, [td] * len(policy_deltas))
            cap_tensors_global_magnitude_(policy_deltas, self.hyperparams.max_policy_weight_update)
            torch._foreach_add_(policy_params, policy_deltas)

    def update_eligibility_traces(self, decay: int, grad_value_function_current_state: tuple[Tensor, ...],
                                  log_grad_policy_current_state: tuple[Tensor, ...]):
        # update value eligibility trace
        torch._foreach_mul_(
            self.value_eligibility_trace,
            self.hyperparams.discount_factor * self.hyperparams.value_trace_decay,
        )
        torch._foreach_add_(self.value_eligibility_trace, grad_value_function_current_state)
        cap_tensors_global_magnitude_(
            self.value_eligibility_trace,
            self.hyperparams.max_value_trace,
        )

        # update policy eligibility trace
        # the foreach function, ending in _ is in place.
        torch._foreach_mul_(
            self.policy_eligibility_trace,
            self.hyperparams.discount_factor * self.hyperparams.policy_trace_decay,
        )
        torch._foreach_add_(self.policy_eligibility_trace, torch._foreach_mul(log_grad_policy_current_state, decay))
        cap_tensors_global_magnitude_(
            self.policy_eligibility_trace,
            self.hyperparams.max_policy_trace,
        )

    def calculate_td_error(self, reward: Tensor, value_function_current_state: Tensor,
                           value_function_next_state: Tensor, terminal : bool) -> Tensor:
        # find td error
        bootstrap = 0.0 if terminal else self.hyperparams.discount_factor * value_function_next_state
        td_error = (reward + bootstrap - value_function_current_state)

        # clip td error
        td_error.clamp_(-self.hyperparams.max_td_error_mag, self.hyperparams.max_td_error_mag)
        return td_error

    def calculate_gradients(self, log_prob_policy, value_function_current_state: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        # find gradients. they dont track gradient / comp graph by default.
        log_grad_policy_current_state = (
            torch.autograd.grad(log_prob_policy, list(self.policy_1.neural_network.parameters()))
        )

        grad_value_function_current_state = (
            torch.autograd.grad(value_function_current_state, list(self.value_function_1.parameters()))
        )

        # clamp magnitude of gradients in place
        cap_tensors_global_magnitude_(log_grad_policy_current_state, self.hyperparams.max_policy_grad_norm)
        cap_tensors_global_magnitude_(grad_value_function_current_state, self.hyperparams.max_value_grad_norm)
        return grad_value_function_current_state, log_grad_policy_current_state

    def estimate_value_function(self, bootstrapping_value_function: NeuralNetwork, current_state: Tensor,
                                next_state: Tensor) -> tuple[Tensor, Tensor]:
        # estimate the value function for the current state and the next state
        with torch.no_grad():
            # use the bootstrapping value function to bootstrap. v1 if not using 2, v2 if using 2.
            value_function_next_state = bootstrapping_value_function(next_state)
        with torch.enable_grad():
            value_function_current_state = self.value_function_1(current_state)

        return value_function_current_state, value_function_next_state

    def train(self):
        with self.environment as environment:

            episodeNumber = 0
            while environment.is_running():
                print(f"Training Episode: {episodeNumber}")
                total_reward_per_timestep = self.train_episode(environment)

                self.episodeStatistics.append({"episode": episodeNumber, "total_reward_per_timestep": total_reward_per_timestep})
                episodeNumber += 1

    def _hyperparams_columns(self, n_rows: int) -> pd.DataFrame:
        hyperparam_values = vars(self.hyperparams)
        return pd.DataFrame({f'H_{k}': [v] * n_rows for k, v in hyperparam_values.items()})

    def save_stats(self, save_file_directory, instance=0):
        episodeStats = pd.DataFrame(self.episodeStatistics)
        timestepStats = pd.DataFrame(self.timestepStatistics)

        if len(episodeStats) > 0:
            episode_hparams = self._hyperparams_columns(len(episodeStats))
            episodeStats = pd.concat([episode_hparams, episodeStats], axis=1)
            episodeStats.to_csv(os.path.join(save_file_directory, f"episode_data_{instance}.csv"))

        if len(timestepStats) > 0:
            timestep_hparams = self._hyperparams_columns(len(timestepStats))
            timestepStats = pd.concat([timestep_hparams, timestepStats], axis=1)
            timestepStats.to_csv(os.path.join(save_file_directory, f"timestep_data_{instance}.csv"))

