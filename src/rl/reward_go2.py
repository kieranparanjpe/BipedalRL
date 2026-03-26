from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .reward import Reward
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..robot.robot import Robot

class RewardGo2(Reward):

    def __init__(self, robot : Robot, target_position : npt.NDArray[np.float32],
                 feet_body_names : List[str], completed_distance = 0.5, floor_distance = 0.25,
                 max_timesteps : int = 10000):
        super().__init__(robot, max_timesteps)
        if target_position.shape != (3,):
            raise ValueError("target position has wrong shape. should be (3,)")
        self.target_position = target_position
        self.feet_body_names = feet_body_names

        self.completed_distance = completed_distance
        self.floor_distance = floor_distance

        self.robot.compute_forward_kinematics()
        root_position = self.robot.get_world_position(self.robot.root_name).reshape(3)
        self.starting_square_distance = self.square_distance_to_target(root_position)
        self.starting_distance = self.distance_to_target(root_position)
        self.steps_on_floor = 0
        self.steps_upside_down = 0
        self.starting_chest_height = root_position[2]


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

        chest_above_floor = 0.5 * (root_position[2] - self.floor_distance) / (self.starting_chest_height -
                                                                             self.floor_distance)

        # take the dot product between (0, 0, 1) and final entry of rotation matrix.
        upright_factor = ((np.array([0, 0, 1]).reshape((1,3)) @
                          self.robot.get_world_rotation(self.robot.root_name).reshape((3, 3))[:, 2])
                          .item() * 0.5)

        chest_above_feet = 0
        for foot in self.feet_body_names:
            foot_position = self.robot.get_world_position(foot)
            chest_above_feet += 0.1 if root_position[2] > foot_position[2] else 0

        return distance_reward + upright_factor + chest_above_floor + chest_above_feet

    def is_terminal(self, timestep) -> bool:
        self.robot.compute_forward_kinematics()
        root_position = self.robot.get_world_position(self.robot.root_name).reshape(3)
        current_square_distance = self.square_distance_to_target(root_position)

        upright = ((np.array([0, 0, 1]).reshape((1,3)) @
                          self.robot.get_world_rotation(self.robot.root_name).reshape((3, 3))[:, 2])
                          .item()) > 0

        if root_position[2] < self.floor_distance:
            self.steps_on_floor += 1
        if not upright:
            self.steps_upside_down += 1



        return (self.steps_on_floor > 50 or current_square_distance < self.completed_distance**2 or
                self.steps_upside_down > 50 or timestep > self.max_timesteps)

    def reset_episode(self):
        self.steps_on_floor = 0
        self.steps_upside_down = 0
        self.robot.compute_forward_kinematics()
        root_position = self.robot.get_world_position(self.robot.root_name).reshape(3)
