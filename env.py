from collections import namedtuple
from typing import Callable, Dict, Optional

import networkx as nx
import numpy as np
from graphenv.examples.tsp.graph_utils import make_complete_planar_graph


EnvInfo = namedtuple("EnvInfo", field_names=["observation", "reward", "done"])

def graph_generator(n_min, n_max):
    n = np.random.randint(n_min, n_max+1)
    return make_complete_planar_graph(N=n)

class TSPEnv():
    def __init__(
        self,
        n_min: int = 10,
        n_max: int = 20,
        graph_generator: Callable[None, nx.Graph] = graph_generator,
        **kwargs
    ) -> None:
        self.n_min = n_min
        self.n_max = n_max
        self.graph_generator = graph_generator
        self.reset()

    def reset(self):
        self.G = self.graph_generator(n_min=self.n_min, n_max=self.n_max)
        self.n = self.G.number_of_nodes()
        self.xv = np.zeros(self.n)
        self.xv[0] = 1
        self.tour = [0]
        return (self.tour.copy(), self.xv.copy())
    
    def get_tour_cost(self):
        return sum(self.G[u][v]["weight"] for (u, v) in zip(self.tour[:-1], self.tour[1:]))

    def get_reward(self, action):
        u, v = self.tour[-1], action
        return -self.G[u][v]["weight"] if "weight" in self.G[u][v] else -float("inf")
    
    def step(self, action: int) -> EnvInfo:
        assert 0 <= action < self.n, f"Vertex {action} should be in the range [0, {self.n-1}]"
        assert (self.xv[action] == 0.0 or (action == self.tour[0] and len(self.tour) == self.n)), f"Vertex {action} already visited"
        # Get reward
        reward = self.get_reward(action)
        
        # Compute new state
        self.xv[action] = 1.0
        self.tour.append(action)
        next_state = self.tour
        
        # Check if done. Should start at some vertex, 
        done = len(self.tour) == self.n+1
        
        # Return all info for step
        env_info = EnvInfo((next_state.copy(), self.xv.copy()), reward, done)
        return env_info
