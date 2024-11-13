import dataclasses
import logging

import networkx as nx
import numpy as np

from s2v_dqn.envs.base_env import BaseEnv, EnvInfo
from s2v_dqn.envs.tsp.tsp_solver import TSPSolver
from s2v_dqn.graph_type import GraphType


@dataclasses.dataclass
class TSPEnv(BaseEnv):
    # use_pyg: bool = False
    normalize_observations: bool = True
    positive_reward: bool = False
    add_node_anywhere: bool = True
    n_node_features: int = 4
    n_edge_features: int = 3

    def __post_init__(self):
        self.fixed_graph = self.graph is not None
        self.start_vertex = 0
        self.reset()
        self.reward_sign = 1 if self.positive_reward else -1
        self.FAIL_REWARD = -10 * self.instance_generator.graph_params['max_coord']

    def reset(self, seed: int = None):
        if not self.fixed_graph:
            # self.graph, self.original_graph = self.instance_generator.generate_graph(seed)
            if self.instance_generator.graph_type == GraphType.EUCLIDEAN:
                self.graph, self.weighted_graph = self.instance_generator.generate_graph(seed)
            else:
                self.graph = self.instance_generator.generate_graph(seed)

        # precompute some node features
        self.nodes_coords = np.array([self.graph.nodes[u]["coords"] for u in self.graph.nodes])
        self.graph_adj_matrix = nx.to_numpy_array(self.graph, weight=None)
        self.graph_weights = nx.to_numpy_array(self.weighted_graph, weight="weight")

        self.n_vertices = self.graph.number_of_nodes()

        self.xv = np.zeros((self.n_vertices, 1))
        self.xv[self.start_vertex, 0] = 1
        self.tour = [self.start_vertex]
        self.tour_size = 0

        self.max_coord = self.instance_generator.graph_params["max_coord"]

        return self.get_observation()

    def get_observation(self):
        """
        An observation here is a pair of:
            - Tuple with shape (n_vertices, n_node_features + n_vertices) where each vertex entry is composed of:
                - node features:
                    - 0/1 if contained in current solution   # range [0,1]
                    - 0/1 if final path node                 # range [0,1]
                    - coordinates                            # range [0,1]
                - adjacency matrix                           # range [0,1]
            - Edge features with shape (n_vertices, n_vertices, n_edge_features):
                - normalized edge weight                     # range [0, sqrt(2)]
                -
                - #
        """
        # 0/1 if final path node
        last = np.zeros((self.n_vertices, 1))
        last[self.tour[-1]] = 1

        norm = self.max_coord if self.normalize_observations else 1
        state = np.concatenate([
            self.xv,
            last,
            self.nodes_coords / norm,
            self.graph_adj_matrix,
            # self.graph_weights / norm
        ], axis=-1)
        edge_features = np.zeros((self.n_vertices, self.n_vertices, self.n_edge_features))
        edge_features[:, :, 0] = self.graph_weights / norm
        for u, v in self.graph.edges:
            edge_features[u, v, 1] = self.xv[u]
            edge_features[u, v, 2] = 1.0 * (self.xv[u] != self.xv[v])

        return state, edge_features

    def step(self, action: int) -> EnvInfo:
        assert 0 <= action < self.n_vertices, f"Vertex {action} should be in the range [0, {self.n_vertices - 1}]"

        # if action is invalid, finish episode with a very negative reward
        if self.xv[action] == 1:
            logging.warning(f"Invalid action: {action}")
            return EnvInfo(self.get_observation(), self.FAIL_REWARD, True)

        # assert (
        #     self.xv[action] == 0
        #     # or (action == self.tour[0] and len(self.tour) == self.n)
        # ), f"Vertex {action} already visited"

        if len(self.tour) == 1:
            self.xv[action] = 1.0
            tour_delta = 2 * self.graph_weights[self.tour[-1], action]
            self.tour_size += tour_delta
            self.tour.append(action)
            done = len(self.tour) == self.n_vertices
            reward = tour_delta / self.max_coord if self.normalize_reward else tour_delta
            env_info = EnvInfo(self.get_observation(), self.reward_sign * reward, done)
            return env_info

        # insert new node at the end of the tour
        idx_to_insert = len(self.tour)
        u, v = self.tour[-1], self.tour[0]
        best_tour_delta = self.graph_weights[u][action] + self.graph_weights[action][v] - self.graph_weights[u][v]

        # helper function with insertion at any point
        if self.add_node_anywhere:
            for i in range(len(self.tour)):
                u, v = self.tour[i], self.tour[i+1] if i+1 < len(self.tour) else 0
                tour_delta = self.graph_weights[u][action] + self.graph_weights[action][v] - self.graph_weights[u][v]
                if tour_delta < best_tour_delta:
                    idx_to_insert = i + 1
                    best_tour_delta = tour_delta

        # Collect reward
        reward = best_tour_delta / self.max_coord if self.normalize_reward else best_tour_delta
        reward *= self.reward_sign

        # print(reward)

        # Compute new state
        self.xv[action] = 1.0
        self.tour_size += best_tour_delta
        self.tour.insert(idx_to_insert, action)

        # Done if tour contains all vertices and returns to first tour vertex
        done = len(self.tour) == self.n_vertices

        # Return all info for step
        env_info = EnvInfo(self.get_observation(), reward, done)
        return env_info

#     def final_step(self):
#         reward = self.get_reward(self.start_vertex)

    def get_best_solution(self, exact_solution_max_size: int, **kwargs) -> float:
        return TSPSolver.get_solution(self.graph, exact_solution_max_size, return_path=kwargs.get("return_path", False))

    # def get_tour_cost(self):
    #     return sum(self.graph[u][v]["weight"] for (u, v) in zip(self.tour[:-1], self.tour[1:]))

    def get_current_solution(self):
        return self.tour_size
