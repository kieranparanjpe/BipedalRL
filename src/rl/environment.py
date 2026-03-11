from abc import ABC, abstractmethod


class Environment(ABC):

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