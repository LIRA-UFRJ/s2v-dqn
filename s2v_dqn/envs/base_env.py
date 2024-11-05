import dataclasses
from abc import ABC, abstractmethod
from collections import namedtuple

import networkx as nx
import numpy as np

from s2v_dqn.instances.instance_generator import InstanceGenerator

EnvInfo = namedtuple("EnvInfo", field_names=["observation", "reward", "done"])


@dataclasses.dataclass
class BaseEnv(ABC):
    instance_generator: InstanceGenerator
    graph_params: dict
    graph: nx.Graph = None
    xv: np.ndarray = None
    normalize_reward: bool = True

    @abstractmethod
    def reset(self, seed: int = None) -> EnvInfo:
        """
        Reset the environment and generates a new graph instance
        :return: The observation for the new environment
        """
        pass

    @abstractmethod
    def get_observation(self):
        """
        Return the observation for the environment, which consists of all of the needed
        info to retrieve the current state, e.g. the graph, the partial solution, etc.
        :return: The observation for the current state of the environment
        """
        pass

    @abstractmethod
    def get_reward(self, action: int) -> float:
        """
        Get reward for given action in the current state
        :param action: the action taken
        :return: Reward for given action in the current state
        """
        pass

    @abstractmethod
    def step(self, action: int) -> EnvInfo:
        """
        Update the current state with a new action taken by the agent
        :param action: The action to be taken. If this is invalid, this method will throw an error
        :return: The observation for the updated state of the environment
        """
        pass

    @abstractmethod
    def get_best_solution(self, exact_solution_max_size: int, **kwargs):
        """
        When that's possible, get the optimal solution for the instance
        When that's not possible, get a proxy of the best answer with some solver
        :param exact_solution_max_size: threshold for graph size up to which the exact solution is calculated
        :return: The best possible computed solution for the instance
        """
        pass

    @abstractmethod
    def get_current_solution(self):
        """
        Get the solution for the current state of the environment
        If the environment is not solved, this method will throw an error
        :return: The solution for the current state of the environment
        """
        pass
