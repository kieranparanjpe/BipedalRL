from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import pandas as pd

from .environment import Environment
from .neural_network import NeuralNetwork
from .policy import Policy
from .reward import Reward
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.robot.robot import Robot


@dataclass
class Hyperparameters:
    policy_learning_rate : float
    value_learning_rate : float
    policy_trace_decay : float
    value_trace_decay : float
    discount_factor : float
    policy_changeout : int
    value_changeout : int

class ActorCritic:

    def __init__(self, environment : Environment, policy : Policy, value_function : NeuralNetwork,
                 reward : Reward, robot : Robot,
                 hyperparams : Hyperparameters = Hyperparameters(
                     policy_learning_rate=1e-12,
                     value_learning_rate=1e-8,
                     policy_trace_decay=0.95,
                     value_trace_decay=0.95,
                     discount_factor=0.95,
                     policy_changeout=1000,
                     value_changeout=1000
                 )):
        self.environment = environment
        self.policy_1 = policy # used for inference
        self.policy_2 = policy.make_init_copy() # used for bootstrapping
        self.value_function_1 = value_function # used for inference
        self.value_function_2 = NeuralNetwork.from_other(value_function) # used for bootstrapping
        self.reward = reward
        self.robot = robot
        self.hyperparams = hyperparams

        self.value_eligibility_trace = [torch.zeros(p.shape) for p in value_function.parameters()]
        self.policy_eligibility_trace =[torch.zeros(p.shape) for p in policy.neural_network.parameters()]

        self.episodeStatistics = []
        self.timestepStatistics = []
        self.total_timesteps = 0

    def train_episode(self, environment):
        environment.reset()
        for t in self.value_eligibility_trace:
            t.zero_()
        for t in self.policy_eligibility_trace:
            t.zero_()
        decay = 1
        total_reward = 0
        self.reward.reset_episode()
        while environment.is_running():
            ''' if self.total_timesteps % self.hyperparams.value_changeout == 0:
                self.value_function_1, self.value_function_2 = self.value_function_2, self.value_function_1
            if self.total_timesteps % self.hyperparams.policy_changeout == 0:
                self.policy_1, self.policy_2 = self.policy_2, self.policy_1'''

            self.value_function_1.zero_grad()
            self.policy_1.neural_network.zero_grad()

            current_state = torch.from_numpy(self.robot.get_state_sin_cos_no_accel()).float()

            with torch.enable_grad():
                action, log_prob_policy = self.policy_1.sample_with_log_prob(current_state)
            self.robot.set_ctrls(action.detach().cpu().numpy())

            environment.step()

            next_state = torch.from_numpy(self.robot.get_state_sin_cos_no_accel()).float()
            reward = torch.tensor(self.reward.reward(), dtype=torch.float32).detach()
            total_reward += reward.item()

            terminal = self.reward.is_terminal()

            with torch.no_grad():
                value_function_next_state = self.value_function_1(next_state)
            with torch.enable_grad():
                value_function_current_state = self.value_function_1(current_state)

            log_grad_policy_current_state = (
                torch.autograd.grad(log_prob_policy, list(self.policy_1.neural_network.parameters()))
            )

            grad_value_function_current_state = (
                torch.autograd.grad(value_function_current_state, list(self.value_function_1.parameters()) )
            )

            bootstrap = 0.0 if terminal else self.hyperparams.discount_factor * value_function_next_state
            td_error = (reward + bootstrap - value_function_current_state).detach()


            for i in range(len(self.value_eligibility_trace)):
                self.value_eligibility_trace[i] = (
                        self.hyperparams.discount_factor * self.hyperparams.value_trace_decay
                        * self.value_eligibility_trace[i] + grad_value_function_current_state[i].detach()
                )


            for i in range(len(self.policy_eligibility_trace)):
                self.policy_eligibility_trace[i] = (
                        self.hyperparams.discount_factor * self.hyperparams.policy_trace_decay
                        * self.policy_eligibility_trace[i] + decay * log_grad_policy_current_state[i].detach())

            # for p in self.value_function.parameters():
            with torch.no_grad():
                for p, eligibility in zip(self.value_function_1.parameters(), self.value_eligibility_trace):
                    p += self.hyperparams.value_learning_rate * td_error * eligibility
                for p, eligibility in zip(self.policy_1.neural_network.parameters(), self.policy_eligibility_trace):
                    p += self.hyperparams.policy_learning_rate * td_error * eligibility

            decay *= self.hyperparams.discount_factor
            self.timestepStatistics.append({"timestep": self.total_timesteps, "td error": td_error.item()})
            self.total_timesteps += 1

            if terminal:
                break

        return total_reward

    def train(self):
        with self.environment as environment:

            episodeNumber = 0
            while environment.is_running():
                print(f"Training Episode: {episodeNumber}")
                total_reward = self.train_episode(environment)

                self.episodeStatistics.append({"episode": episodeNumber, "total_reward": total_reward})
                episodeNumber += 1


    def plot_stats(self, save_directory=None, suffix=None):
        episodeStats = pd.DataFrame(self.episodeStatistics)
        timestepStats = pd.DataFrame(self.timestepStatistics)

        suffix = f"_{int(time.time())}" if suffix is None else suffix

        if len(episodeStats) > 0:
            episodeStats.plot(x="episode", y=["total_reward"])
            plt.title('total reward vs episodes')
            if save_directory is not None:
                plt.savefig(os.path.join(save_directory, f'total_reward_vs_episodes{suffix}'))
            plt.show()

        timestepStats.plot(x="timestep", y=["td error"])
        plt.title('td error vs timestep')
        if save_directory is not None:
            plt.savefig(os.path.join(save_directory, f'td_error_vs_timestep{suffix}'))
        plt.show()