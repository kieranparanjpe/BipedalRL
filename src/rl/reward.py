from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..robot.robot import Robot

class Reward(ABC):

    def __init__(self, robot : Robot, max_timesteps : int):
        self.robot = robot
        self.max_timesteps = max_timesteps

    @abstractmethod
    def reward(self) -> float:
        raise NotImplemented

    @abstractmethod
    def is_terminal(self, timestep : int) -> bool:
        raise NotImplemented

    @abstractmethod
    def reset_episode(self):
        raise NotImplemented
