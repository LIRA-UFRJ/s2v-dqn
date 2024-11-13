import dataclasses

import networkx as nx
import numpy as np

from s2v_dqn.envs.mvc.mvc_solver import MVCSolver
from s2v_dqn.envs.base_env import BaseEnv, EnvInfo


@dataclasses.dataclass
class MVCEnv(BaseEnv):
    # use_pyg: bool = False
    n_node_features: int = 1
    n_edge_features: int = 0

    def __post_init__(self):
        self.fixed_graph = self.graph is not None
        self.reset()

    def reset(self, seed: int = None):
        if not self.fixed_graph:
            self.graph = self.instance_generator.generate_graph(seed)

        # precompute some node features
        self.graph_adj_matrix = nx.to_numpy_array(self.graph, weight=None)

        self.n_vertices = self.graph.number_of_nodes()
        self.n_edges = self.graph.number_of_edges()
        self.covered_edges = 0
        self.xv = np.zeros(self.n_vertices)
        self.solution_size = 0
        return self.get_observation()

    def get_observation(self):
        """
        An observation is a tuple with shape (n_vertices, n_node_features + 2 * n_vertices)
        where each vertex entry is composed of:
            - node features:
                - 0/1 if contained in current solution   # range [0,1]
            - adjacency matrix                           # range [0,1]
        """
        # 0/1 if final path node
        state = np.column_stack([
            self.xv,
            self.graph_adj_matrix,
        ])
        # TODO: swap dims
        # ret = np.vstack([
        #     self.xv,
        #     self.graph_adj_matrix
        # ])
        edge_features = np.zeros((self.n_vertices, self.n_vertices, 0))

        return state, edge_features

    def step(self, action: int) -> EnvInfo:
        assert 0 <= action < self.n_vertices, f"Vertex {action} should be in the range [0, {self.n_vertices - 1}]"
        assert self.xv[action] == 0.0, f"Vertex {action} already visited"

        # Collect reward
        reward = -1 / self.n_vertices if self.normalize_reward else -1

        # Compute new state
        self.xv[action] = 1.0
        self.solution_size += 1

        # Covered edges increases by number of neighbors that were not in the solution
        self.covered_edges += np.dot(self.graph_adj_matrix[action], 1 - self.xv)

        # Done if chosen nodes covers all edges
        done = self.covered_edges == self.n_edges

        # Return all info for step
        env_info = EnvInfo(self.get_observation(), reward, done)
        return env_info

    def get_best_solution(self, exact_solution_max_size: int, **kwargs) -> float:
        return MVCSolver.get_solution(self.graph, exact_solution_max_size)

    def get_current_solution(self):
        return self.solution_size
