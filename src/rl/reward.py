from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..robot.robot import Robot

class Reward(ABC):

    def __init__(self, robot : Robot):
        self.robot = robot

    @abstractmethod
    def reward(self) -> float:
        raise NotImplemented

    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplemented

    @abstractmethod
    def reset_episode(self):
        raise NotImplemented
