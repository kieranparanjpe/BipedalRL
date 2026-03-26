from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .reward import Reward
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..robot.robot import Robot

class RewardCube(Reward):

    def __init__(self, robot : Robot, target_position : npt.NDArray[np.float32],
                 completed_distance = 0.5, max_steps_without_improvement=1000,
                 max_timesteps : int = 50000):
        super().__init__(robot, max_timesteps)
        if target_position.shape != (3,):
            raise ValueError("target position has wrong shape. should be (3,)")
        self.target_position = target_position

        self.completed_distance = completed_distance

        self.robot.compute_forward_kinematics()
        root_position = self.robot.get_world_position(self.robot.root_name).reshape(3)
        self.starting_square_distance = self.square_distance_to_target(root_position)
        self.starting_distance = self.distance_to_target(root_position)
        self.steps_without_improvement = 0
        self.max_steps_without_improvement = max_steps_without_improvement
        self.highest_reward = -float("inf")

    def square_distance_to_target(self, root_position):
        difference = self.target_position - root_position
        return np.dot(difference[0:2], difference[0:2]).item()

    def distance_to_target(self, root_position):
        return np.sqrt(self.square_distance_to_target(root_position))

    def reward(self) -> float:
        self.robot.compute_forward_kinematics()
        root_position = self.robot.get_world_position(self.robot.root_name).reshape(3)

        '''current_square_distance = self.square_distance_to_target(root_position)
        distance_reward = 1 - current_square_distance / self.starting_square_distance'''

        current_distance = self.distance_to_target(root_position)
        distance_reward = 1 - current_distance / self.starting_distance

        if current_distance < self.completed_distance:
            distance_reward += 2

        if distance_reward > self.highest_reward:
            self.highest_reward = distance_reward
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        return distance_reward

    def is_terminal(self, timestep) -> bool:
        self.robot.compute_forward_kinematics()
        root_position = self.robot.get_world_position(self.robot.root_name).reshape(3)
        current_square_distance = self.square_distance_to_target(root_position)
        return (self.steps_without_improvement > self.max_steps_without_improvement
                or current_square_distance < self.completed_distance**2
                or timestep > self.max_timesteps)

    def reset_episode(self):
        self.highest_reward = -float("inf")
        self.steps_without_improvement = 0
        self.robot.compute_forward_kinematics()
        root_position = self.robot.get_world_position(self.robot.root_name).reshape(3)
