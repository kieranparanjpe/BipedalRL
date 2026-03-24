from abc import ABC, abstractmethod

from io_controller import IOController


class Environment(ABC):

    def __init__(self):
        self._end_environment = False

    @abstractmethod
    def step(self):
        raise NotImplemented

    @abstractmethod
    def __enter__(self):
        raise NotImplemented

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplemented

    @abstractmethod
    def is_running(self):
        raise NotImplemented

    @abstractmethod
    def reset(self):
        raise NotImplemented

    def end_environment(self):
        self._end_environment = True


