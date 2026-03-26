from __future__ import annotations

import os
import math
import time

import torch
import pandas as pd
from torch import Tensor
from tqdm import tqdm

from .hyperparameters import Hyperparameters
from .environment import Environment
from .neural_network import NeuralNetwork
from .policy import Policy
from .reward import Reward
from typing import TYPE_CHECKING, Optional, Sequence, Callable

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
                 hyperparams : Hyperparameters,
                 timestep_callback : Optional[Callable[[int], None]] = None,
                 generate_timestep_statistics = False,
                 device = torch.device('cpu')):
        self.device = device
        self.environment = environment
        self.policy_1 = policy # used for inference
        self.value_function_1 = value_function # used for inference
        self.value_function_2 = NeuralNetwork.from_other(value_function) # used for bootstrapping
        self.reward = reward
        self.robot = robot
        self.hyperparams = hyperparams

        self.value_eligibility_trace = [torch.zeros_like(p) for p in value_function.parameters()]
        self.policy_eligibility_trace = [torch.zeros_like(p) for p in policy.neural_network.parameters()]

        self.generate_timestep_statistics = generate_timestep_statistics

        self.episode_statistics = []
        self.timestep_statistics = []
        self.total_timesteps = 0
        self.total_episodes = 0

        self.preloaded_episode_statistics = None
        self.preloaded_timestep_statistics = None

        self.timestep_callback = timestep_callback

    def reset_eligibility_traces(self):
        torch._foreach_zero_(self.value_eligibility_trace)
        torch._foreach_zero_(self.policy_eligibility_trace)

    def train(self, instance=0):
        with self.environment as environment:
            while environment.is_running() and self.total_episodes < self.hyperparams.max_episodes:
                start_time = int(time.time())
                timestep_summary = self.train_episode(environment)

                episode_stats = {"episode": self.total_episodes, "global_timestep": self.total_timesteps} | timestep_summary

                self.episode_statistics.append(episode_stats)

                print(f"Trained Instance: {instance} | Episode: {self.total_episodes}/{self.hyperparams.max_episodes}"
                      f" | Latest Avg Reward: {episode_stats['total_reward_per_timestep']} | Time Taken {int(time.time()) - start_time}")

                self.total_episodes += 1


    def train_episode(self, environment) -> dict[str, float]:
        environment.reset()
        self.reset_eligibility_traces()
        self.reward.reset_episode()

        bootstrapping_value_function = self.value_function_1 if self.hyperparams.value_function_changeout is None \
            else self.value_function_2

        decay = 1
        total_reward = torch.zeros(1, dtype=torch.float32, device=self.device)
        timesteps = 0

        timestep_stats_sum = None

        while environment.is_running():
            if self.timestep_callback is not None:
                self.timestep_callback(self.total_timesteps)

            # if we are using 2 value functions, update learned params only every n steps.
            if self.hyperparams.value_function_changeout is not None:
                if self.total_timesteps % self.hyperparams.value_function_changeout == 0:
                    self.value_function_2.load_state_dict(self.value_function_1.state_dict())

            # Get current state S
            current_state = torch.as_tensor(self.robot.get_state_sin_cos_no_accel(), dtype=torch.float32,
                                         device=self.device)

            # Sample best action A, based on S. Also get the log probability of A, ln[P(A)]
            action, log_prob_policy = self.policy_1.sample_with_log_prob(current_state)

            # Take action A
            self.robot.set_ctrls(action.detach().cpu().numpy())

            environment.step() # Step the environment.

            # Find next state  after stepping env. S'
            next_state = torch.as_tensor(self.robot.get_state_sin_cos_no_accel(), dtype=torch.float32,
                                         device=self.device)

            # Get next reward
            reward = torch.tensor(self.reward.reward(), dtype=torch.float32, device=self.device)
            total_reward += reward
            terminal = self.reward.is_terminal(timesteps)

            value_function_current_state, value_function_next_state = self.estimate_value_function(
                bootstrapping_value_function, current_state, next_state
            )

            grad_value_function_current_state, log_grad_policy_current_state, gradient_stats = self.calculate_gradients(
                log_prob_policy, value_function_current_state
            )

            td_error = self.calculate_td_error(reward, value_function_current_state,
                value_function_next_state, terminal)

            trace_stats = self.update_eligibility_traces(decay, grad_value_function_current_state,
                                                 log_grad_policy_current_state)

            # update weights
            update_stats = self.update_weights(td_error)

            decay *= self.hyperparams.discount_factor_decay if self.hyperparams.discount_factor_decay is not None \
                else self.hyperparams.discount_factor

            timestep_stats = (
                {
                    "episode": self.total_episodes,
                    "timestep": self.total_timesteps,
                    "abs(td error)": abs(td_error.item()),
                    "reward": reward.item(),
                    "action_mean": action.detach().mean().item(),
                    "action_std": action.detach().std().item(),
                    "action_l2_norm": action.detach().norm().item(),
                }
                | gradient_stats
                | update_stats
                | trace_stats
                | self.policy_1.get_statistics()
            )

            if self.generate_timestep_statistics:
                self.timestep_statistics.append(timestep_stats)

            if timestep_stats_sum is None:
                timestep_stats_sum = timestep_stats
            else:
                timestep_stats_sum = {k: timestep_stats_sum[k] + timestep_stats[k] for k in timestep_stats_sum.keys() & timestep_stats.keys()}

            self.total_timesteps += 1
            timesteps += 1
            if terminal:
                break

        timestep_summary = {k: timestep_stats_sum[k] / timesteps for k in timestep_stats_sum.keys()} | {
            'episode_length': timesteps, 'total_reward_per_timestep': total_reward.item() / timesteps}
        return timestep_summary

    def update_weights(self, td_error: Tensor) -> dict[str, float]:
        with torch.no_grad():
            value_params = list(self.value_function_1.parameters())
            policy_params = list(self.policy_1.neural_network.parameters())

            td = td_error.detach()  # stays on device

            value_deltas = torch._foreach_mul(self.value_eligibility_trace, self.hyperparams.value_learning_rate)
            # need [] * len for broadcasting
            value_deltas = torch._foreach_mul(value_deltas, [td] * len(value_deltas))
            value_update_norm_before_clip = cap_tensors_global_magnitude_(
                value_deltas, self.hyperparams.max_value_weight_update
            ).item()
            torch._foreach_add_(value_params, value_deltas)

            policy_deltas = torch._foreach_mul(self.policy_eligibility_trace, self.hyperparams.policy_learning_rate)
            policy_deltas = torch._foreach_mul(policy_deltas, [td] * len(policy_deltas))
            policy_update_norm_before_clip = cap_tensors_global_magnitude_(
                policy_deltas, self.hyperparams.max_policy_weight_update
            ).item()
            torch._foreach_add_(policy_params, policy_deltas)

        return {
            "value_update_norm_before_clip": value_update_norm_before_clip,
            "value_update_norm_after_clip": min(value_update_norm_before_clip, self.hyperparams.max_value_weight_update),
            "policy_update_norm_before_clip": policy_update_norm_before_clip,
            "policy_update_norm_after_clip": min(policy_update_norm_before_clip, self.hyperparams.max_policy_weight_update),
        }

    def update_eligibility_traces(self, decay: int, grad_value_function_current_state: tuple[Tensor, ...],
                                  log_grad_policy_current_state: tuple[Tensor, ...]) -> dict[str, float]:
        # update value eligibility trace
        torch._foreach_mul_(
            self.value_eligibility_trace,
            self.hyperparams.discount_factor * self.hyperparams.value_trace_decay,
        )
        torch._foreach_add_(self.value_eligibility_trace, grad_value_function_current_state)
        value_trace_norm_before_clip = cap_tensors_global_magnitude_(
            self.value_eligibility_trace,
            self.hyperparams.max_value_trace,
        ).item()

        # update policy eligibility trace
        # the foreach function, ending in _ is in place.
        torch._foreach_mul_(
            self.policy_eligibility_trace,
            self.hyperparams.discount_factor * self.hyperparams.policy_trace_decay,
        )
        torch._foreach_add_(self.policy_eligibility_trace, torch._foreach_mul(log_grad_policy_current_state, decay))
        policy_trace_norm_before_clip = cap_tensors_global_magnitude_(
            self.policy_eligibility_trace,
            self.hyperparams.max_policy_trace,
        ).item()

        return {
            "policy_trace_norm_before_clip": policy_trace_norm_before_clip,
            "policy_trace_norm_after_clip": min(policy_trace_norm_before_clip, self.hyperparams.max_policy_trace),
            "value_trace_norm_before_clip": value_trace_norm_before_clip,
            "value_trace_norm_after_clip": min(value_trace_norm_before_clip, self.hyperparams.max_value_trace),
        }

    def calculate_td_error(self, reward: Tensor, value_function_current_state: Tensor,
                           value_function_next_state: Tensor, terminal : bool) -> Tensor:
        # find td error
        bootstrap = 0.0 if terminal else self.hyperparams.discount_factor * value_function_next_state
        td_error = (reward + bootstrap - value_function_current_state)

        # clip td error
        td_error.clamp_(-self.hyperparams.max_td_error_mag, self.hyperparams.max_td_error_mag)
        return td_error

    def calculate_gradients(self, log_prob_policy, value_function_current_state: Tensor) \
            -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], dict[str, float]]:
        # find gradients. they dont track gradient / comp graph by default.
        log_grad_policy_current_state = (
            torch.autograd.grad(log_prob_policy, list(self.policy_1.neural_network.parameters()))
        )

        grad_value_function_current_state = (
            torch.autograd.grad(value_function_current_state, list(self.value_function_1.parameters()))
        )

        # clamp magnitude of gradients in place; function returns pre-clip norm
        policy_grad_norm_before_clip = cap_tensors_global_magnitude_(
            log_grad_policy_current_state, self.hyperparams.max_policy_grad_norm
        ).item()
        value_grad_norm_before_clip = cap_tensors_global_magnitude_(
            grad_value_function_current_state, self.hyperparams.max_value_grad_norm
        ).item()

        return grad_value_function_current_state, log_grad_policy_current_state, {
            "policy_grad_norm_before_clip": policy_grad_norm_before_clip,
            "policy_grad_norm_after_clip": min(policy_grad_norm_before_clip, self.hyperparams.max_policy_grad_norm),
            "value_grad_norm_before_clip": value_grad_norm_before_clip,
            "value_grad_norm_after_clip": min(value_grad_norm_before_clip, self.hyperparams.max_value_grad_norm),
        }

    def estimate_value_function(self, bootstrapping_value_function: NeuralNetwork, current_state: Tensor,
                                next_state: Tensor) -> tuple[Tensor, Tensor]:
        # estimate the value function for the current state and the next state
        with torch.no_grad():
            # use the bootstrapping value function to bootstrap. v1 if not using 2, v2 if using 2.
            value_function_next_state = bootstrapping_value_function(next_state)
        with torch.enable_grad():
            value_function_current_state = self.value_function_1(current_state)

        return value_function_current_state, value_function_next_state


    def _hyperparams_columns(self, n_rows: int) -> pd.DataFrame:
        hyperparam_values = vars(self.hyperparams)
        return pd.DataFrame({f'H_{k}': [v] * n_rows for k, v in hyperparam_values.items()})

    @staticmethod
    def _hyperparams_match(loaded_value, current_value, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
        if pd.isna(loaded_value) and current_value is None:
            return True
        if loaded_value is None and current_value is None:
            return True
        if isinstance(loaded_value, bool) or isinstance(current_value, bool):
            return loaded_value == current_value
        if isinstance(loaded_value, (int, float)) and isinstance(current_value, (int, float)):
            return math.isclose(float(loaded_value), float(current_value), rel_tol=rel_tol, abs_tol=abs_tol)
        return loaded_value == current_value

    def load_stats(self, load_file_directory, instance=0, continuation=False):
        episode_statistics = pd.read_csv(os.path.join(load_file_directory, f"episode_data_{instance}.csv"))

        # Check loaded hyperparams match current hyperparams (using first row H_* values).
        h_cols = [c for c in episode_statistics.columns if c.startswith("H_")]
        if h_cols:
            loaded_params = {c[2:]: episode_statistics.iloc[0][c] for c in h_cols}
            current_params = vars(self.hyperparams)
            mismatch = [
                k for k, v in current_params.items()
                if not self._hyperparams_match(loaded_params.get(k, None), v)
            ]
            if mismatch:
                for k in mismatch:
                    if k in loaded_params:
                        cur = current_params.get(k)
                        val = loaded_params[k]
                        if cur is not None:
                            val = type(cur)(val)
                        setattr(self.hyperparams, k, val)

                print(f"WARNING: Loaded hyperparams mismatch for keys: {mismatch}")
        if not continuation:
            return

        # Drop old H_* so save_stats can add one clean set for all rows.
        episode_statistics = episode_statistics.drop(columns=h_cols, errors="ignore")

        # continue from timestep where we left off
        self.total_episodes = int(episode_statistics['episode'].iloc[-1]) + 1
        self.total_timesteps = int(episode_statistics['global_timestep'].iloc[-1]) + 1

        self.preloaded_episode_statistics = episode_statistics

        if self.generate_timestep_statistics:
            timestep_statistics = pd.read_csv(os.path.join(load_file_directory, f"timestep_data_{instance}.csv"))
            timestep_statistics = timestep_statistics.drop(columns=h_cols, errors="ignore")
            self.preloaded_timestep_statistics = timestep_statistics





    def save_stats(self, save_file_directory, instance=0):
        print(f"Saving raw data for instance: {instance}")

        def save_data(array, preloaded, name):
            stats = pd.DataFrame(array)

            if preloaded is not None:
                stats = pd.concat([preloaded, stats], ignore_index=True)

            if len(stats) > 0:
                hparams = self._hyperparams_columns(len(stats))
                stats = pd.concat([hparams, stats], axis=1)
                stats.to_csv(os.path.join(save_file_directory, f"{name}_data_{instance}.csv"))

        save_data(self.episode_statistics, self.preloaded_episode_statistics,'episode')

        if self.generate_timestep_statistics:
            save_data(self.timestep_statistics, self.preloaded_timestep_statistics, 'timestep')




