from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for an agent to act on CO environments"""

    _SUPPORTED_PROBLEMS = {}

    def __init__(self, problem: str, *args, **kwargs):
        problem = problem.lower()
        assert problem in self._SUPPORTED_PROBLEMS

    @abstractmethod
    def reset_episode(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def act(self, obs, *args, **kwargs):
        pass
