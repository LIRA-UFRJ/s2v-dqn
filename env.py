from collections import namedtuple
from typing import Callable, Dict, Optional

import networkx as nx
import numpy as np
from graph_utils import make_complete_planar_graph


EnvInfo = namedtuple("EnvInfo", field_names=["observation", "reward", "done"])

def graph_generator(n_min, n_max, k_nearest=None, seed=None) -> nx.Graph:
    n = np.random.randint(n_min, n_max+1)
    G = make_complete_planar_graph(N=n, seed=seed).to_directed()

    # add attribute to indicate the closest neighbors of each node
    for u in range(n):
        neighbors_rank = sorted(range(n), key=lambda v: G[u][v]["weight"] if v != u else 0)
        for i, vth_closest in enumerate(neighbors_rank):
            if vth_closest not in G[u]: continue
            G[u][vth_closest]["closest"] = i

    # keep only k nearest neighbors
    if k_nearest is not None:
        edges_to_remove = [(u,v) for u, v in G.edges if G[u][v]["closest"] > k_nearest]
        G.remove_edges_from(edges_to_remove)
    return G

class TSPEnv():
    def __init__(
        self,
        n_min: int = 10,
        n_max: int = 20,
        k_nearest: Optional[int] = None,
        graph_generator: Callable[[int, int, Optional[int], Optional[int]], nx.Graph] = graph_generator,
        negate_reward: bool = False,
        **kwargs
    ) -> None:
        self.n_min = n_min
        self.n_max = n_max
        self.graph_generator = graph_generator
        self.negate_reward = negate_reward
        # TODO: remove this once training converges for single graph
        self.G = self.graph_generator(n_min=self.n_min, n_max=self.n_max, k_nearest=k_nearest)
        self.reset()

    def reset(self):
        # self.G = self.graph_generator(n_min=self.n_min, n_max=self.n_max)

        # precompute some node features
        self.nodes_pos = np.array([self.G.nodes[u]["pos"] for u in self.G.nodes])
        self.graph_adj_matrix = nx.to_numpy_array(self.G, weight=None)
        self.graph_weights = nx.to_numpy_array(self.G, weight="weight")

        self.n = self.G.number_of_nodes()
        self.xv = np.zeros((self.n, 1))
        self.xv[0,0] = 1
        self.tour = [0]
        # TODO: delete
        # return (self.tour.copy(), self.xv.copy())
        return self.get_observation()
    
    def get_tour_cost(self):
        return sum(self.G[u][v]["weight"] for (u, v) in zip(self.tour[:-1], self.tour[1:]))

    def get_observation(self):
        """
        An observation is a tuple with shape (n_vertices, n_node_features + 2 * n_vertices)
        where each vertex entry is composed of:
            - node features:
                - 0/1 if contained in current solution
                - 0/1 if final path node
                - coordinates
            - adjacency matrix
            - edge features (currently only edge weight)
        """        
        # 0/1 if final path node
        last = np.zeros((self.n, 1))
        last[self.tour[-1]] = 1
        
        ret = np.concatenate([self.xv, last, self.nodes_pos, self.graph_adj_matrix, self.graph_weights], axis=-1)

        return ret
    
    def get_reward(self, action):
        u, v = self.tour[-1], action
        reward = -self.G[u][v]["weight"] if v in self.G[u] and "weight" in self.G[u][v] else -float("inf")
        return -reward if self.negate_reward else reward
    
    def step(self, action: int) -> EnvInfo:
        assert 0 <= action < self.n, f"Vertex {action} should be in the range [0, {self.n-1}]"
        assert (self.xv[action] == 0.0 or (action == self.tour[0] and len(self.tour) == self.n)), f"Vertex {action} already visited"

        # Collect reward
        reward = self.get_reward(action)
        
        # Compute new state
        self.xv[action] = 1.0
        self.tour.append(action)
        next_state = self.tour
        
        # Done if tour contains all vertices and returns to first tour vertex
        done = len(self.tour) == self.n + 1
        
        # Return all info for step
        env_info = EnvInfo(self.get_observation(), reward, done)
        return env_info
