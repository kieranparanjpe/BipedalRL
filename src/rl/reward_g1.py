from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .reward import Reward
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..robot.robot import Robot

class RewardG1(Reward):

    def __init__(self, robot : Robot, target_position : npt.NDArray[np.float32],
                 chest_body_name : str, completed_distance = 0.5, floor_distance = 0.5):
        super().__init__(robot)
        if target_position.shape != (3,):
            raise ValueError("target position has wrong shape. should be (3,)")
        self.target_position = target_position
        self.chest_body_name = chest_body_name

        self.completed_distance = completed_distance
        self.floor_distance = floor_distance

        self.robot.compute_forward_kinematics()
        root_position = self.robot.get_world_position(self.robot.root_name).reshape(3)
        self.starting_square_distance = self.square_distance_to_target(root_position)
        self.starting_distance = self.distance_to_target(root_position)
        self.steps_on_floor = 0

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

        chest_position = self.robot.get_world_position(self.chest_body_name).reshape(3)
        torso_above_hips = 0.2 if chest_position[2] > root_position[2] else 0

        upright_factor = (np.array([0, 0, 1]).reshape((1,3)) @ self.robot.get_world_rotation(
            self.chest_body_name).reshape((3, 3))[:, 2]).item() * 0.5

        return distance_reward + torso_above_hips + upright_factor

    def is_terminal(self) -> bool:
        self.robot.compute_forward_kinematics()
        chest_position = self.robot.get_world_position(self.chest_body_name).reshape(3)
        root_position = self.robot.get_world_position(self.robot.root_name).reshape(3)
        current_square_distance = self.square_distance_to_target(root_position)

        if chest_position[2] < self.floor_distance:
            self.steps_on_floor += 1

        return self.steps_on_floor > 2000 or current_square_distance < self.completed_distance**2

    def reset_episode(self):
        self.steps_on_floor = 0
