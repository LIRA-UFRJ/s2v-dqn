from collections import namedtuple
from typing import Callable, Dict, Optional

import networkx as nx
import numpy as np
from graph_utils import make_complete_planar_graph


EnvInfo = namedtuple("EnvInfo", field_names=["observation", "reward", "done"])

def graph_generator(n_min, n_max, pos_lim, k_nearest=None, seed=None) -> nx.Graph:
    n = np.random.randint(n_min, n_max+1)
    G = make_complete_planar_graph(N=n, seed=seed, pos_lim=pos_lim).to_directed()

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
        pos_lim: float = 1e3,
        k_nearest: Optional[int] = None,
        graph_generator: Callable[[int, int, Optional[int], Optional[int]], nx.Graph] = graph_generator,
        negate_reward: bool = False,
        normalize_reward: bool = True,
        normalize_observations: bool = True,
        cycle: bool = True, 
        **kwargs
    ) -> None:
        self.n_min = n_min
        self.n_max = n_max
        self.pos_lim = pos_lim
        self.graph_generator = graph_generator
        self.negate_reward = negate_reward
        self.normalize_reward = normalize_reward
        self.normalize_observations = normalize_observations
        self.start_vertex = 0
        self.cycle = cycle
        # TODO: remove this once training converges for single graph
        # self.G = self.graph_generator(n_min=self.n_min, n_max=self.n_max, k_nearest=k_nearest)
        self.reset()

    def reset(self):
        self.G = self.graph_generator(n_min=self.n_min, n_max=self.n_max, pos_lim=self.pos_lim)

        # precompute some node features
        self.nodes_pos = np.array([self.G.nodes[u]["pos"] for u in self.G.nodes])
        self.graph_adj_matrix = nx.to_numpy_array(self.G, weight=None)
        self.graph_weights = nx.to_numpy_array(self.G, weight="weight")

        self.n = self.G.number_of_nodes()
        self.xv = np.zeros((self.n, 1))
        self.xv[self.start_vertex, 0] = 1
        self.tour = [self.start_vertex]
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
                - 0/1 if contained in current solution   # range [0,1]
                - 0/1 if final path node                 # range [0,1]
                - coordinates                            # range [0,1]
            - adjacency matrix                           # range [0,1]
            - edge features (currently only edge weight) # range [0, sqrt(2)]
        """
        # 0/1 if final path node
        last = np.zeros((self.n, 1))
        last[self.tour[-1]] = 1
        
        if self.normalize_observations:
            ret = np.concatenate([
                self.xv,
                last,
                self.nodes_pos / self.pos_lim,
                self.graph_adj_matrix,
                self.graph_weights / self.pos_lim
            ], axis=-1)
        else:
            ret = np.concatenate([
                self.xv,
                last,
                self.nodes_pos,
                self.graph_adj_matrix,
                self.graph_weights
            ], axis=-1)
        
        return ret
    
    def get_reward(self, action):
        def get_weight(u, v):
            return -self.G[u][v]["weight"] if v in self.G[u] and "weight" in self.G[u][v] else -float("inf")
        u, v = self.tour[-1], action
        reward = get_weight(u, v)
        if len(self.tour) + 1 == self.n and self.cycle:
            reward += get_weight(v, self.start_vertex)
        
        reward_signal_adjusted = -reward if self.negate_reward else reward
        # reward_norm_adjusted = reward_signal_adjusted / self.n if self.normalize_reward else reward_signal_adjusted
        reward_norm_adjusted = reward_signal_adjusted / self.pos_lim if self.normalize_reward else reward_signal_adjusted
        return reward_norm_adjusted
    
    def step(self, action: int) -> EnvInfo:
        assert 0 <= action < self.n, f"Vertex {action} should be in the range [0, {self.n-1}]"
        assert (self.xv[action] == 0.0 or (action == self.tour[0] and len(self.tour) == self.n)), f"Vertex {action} already visited"

        # Collect reward
        reward = self.get_reward(action)
        
        # Compute new state
        # print(f'[env] {self.xv=}')
        # print(f'[env] {action=}')
        self.xv[action] = 1.0
        # print(f'[env] {self.xv=}')
        self.tour.append(action)
        
        # Done if tour contains all vertices and returns to first tour vertex
        done = len(self.tour) == self.n
        
        # Return all info for step
        env_info = EnvInfo(self.get_observation(), reward, done)
        return env_info

#     def final_step(self):
#         reward = self.get_reward(self.start_vertex)
        
class MVCEnv():
    ERDOS_RENYI = 1
    BARABASI_ALBERT = 2
    DEFAULT_ERDOS_RENYI_PROBABILITY = 0.4
    DEFAULT_BARABASI_ALBERT_DEGREE = 4
    
    def __init__(
        self,
        n_min: int = 10,
        n_max: int = 20,
        normalize_reward: bool = True,
        graph_type = BARABASI_ALBERT,
        **kwargs
    ) -> None:
        self.n_min = n_min
        self.n_max = n_max
        self.normalize_reward = normalize_reward
        self.graph_type = graph_type
        self.start_vertex = 0
        self.graph_param = kwargs.get(
            "graph_param",
            MVCEnv.DEFAULT_ERDOS_RENYI_PROBABILITY if graph_type == MVCEnv.ERDOS_RENYI else MVCEnv.DEFAULT_BARABASI_ALBERT_DEGREE
        )
        # TODO: remove this once training converges for single graph
        # self.G = self.graph_generator(n_min=self.n_min, n_max=self.n_max, k_nearest=k_nearest)
        self.reset()

    def reset(self):
        self.G = self.generate_graph(n_min=self.n_min, n_max=self.n_max)

        # precompute some node features
        self.graph_adj_matrix = nx.to_numpy_array(self.G, weight=None)

        self.n = self.G.number_of_nodes()
        self.n_edges = self.G.number_of_edges()
        self.covered_edges = 0
        self.xv = np.zeros(self.n)
        return self.get_observation()
    
    def generate_graph(self, n_min, n_max):
        n = np.random.randint(n_min, n_max + 1)
        if self.graph_type == MVCEnv.ERDOS_RENYI:
            return nx.erdos_renyi_graph(n, p=self.graph_param)
        elif self.graph_type == MVCEnv.BARABASI_ALBERT:
            return nx.barabasi_albert_graph(n, m=self.graph_param)
        raise ValueError("graph_type must be MVCEnv.ERDOS_RENYI or MVCEnv.BARABASI_ALBERT")
    
    def get_solution_score(self):
        return self.xv.sum()
    
    def get_observation(self):
        """
        An observation is a tuple with shape (n_vertices, n_node_features + 2 * n_vertices)
        where each vertex entry is composed of:
            - node features:
                - 0/1 if contained in current solution   # range [0,1]
            - adjacency matrix                           # range [0,1]
        """
        # 0/1 if final path node
        ret = np.column_stack([
            self.xv,
            self.graph_adj_matrix,
        ])
        # TODO: swap dims
        # ret = np.vstack([
        #     self.xv,
        #     self.graph_adj_matrix
        # ])

        return ret
    
    def get_reward(self, action):
        reward = -1
        reward_norm_adjusted = reward / self.n if self.normalize_reward else reward
        return reward_norm_adjusted
    
    def step(self, action: int) -> EnvInfo:
        assert 0 <= action < self.n, f"Vertex {action} should be in the range [0, {self.n-1}]"
        assert self.xv[action] == 0.0, f"Vertex {action} already visited"

        # Collect reward
        reward = self.get_reward(action)
        
        # Compute new state
        self.xv[action] = 1.0
        
        # Covered edges increases by number of neighbors that were not in the solution 
        self.covered_edges += np.dot(self.graph_adj_matrix[action], 1 - self.xv)
        
        # Done if chosen nodes covers all edges
        done = self.covered_edges == self.n_edges
        
        # Return all info for step
        env_info = EnvInfo(self.get_observation(), reward, done)
        return env_info
