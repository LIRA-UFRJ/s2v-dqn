from collections import deque

import networkx as nx
import numpy as np

from s2v_dqn.graph_type import GraphType


class InstanceGenerator:
    def __init__(self,
                 n_min: int,
                 n_max: int,
                 graph_type: GraphType,
                 graph_params: dict):
        self.n_min = n_min
        self.n_max = n_max
        self.graph_type = graph_type
        self.graph_params = graph_params

    def _generate_euclidean_graph(self, n: int, rng=None):
        graph = nx.complete_graph(n)

        max_coord = self.graph_params["max_coord"]
        coords = rng.rand(n, 2) * max_coord

        for ei, ej in graph.edges:
            graph[ei][ej]["weight"] = np.sqrt(sum((coords[ei] - coords[ej]) ** 2))

        for node in graph.nodes:
            graph.nodes[node]["coords"] = coords[node, :]

        return graph

    def generate_graph(self, seed: int = None):
        rng = np.random.RandomState(abs(seed % (2**32))) if seed is not None else np.random
        n = rng.randint(self.n_min, self.n_max + 1)
        if self.graph_type == GraphType.ERDOS_RENYI:
            while True:
                _graph = nx.erdos_renyi_graph(n, p=self.graph_params["p"], seed=rng)
                check = True
                check = check and _graph.number_of_edges() > 0
                if 'connected' in self.graph_params:
                    check = check and self.graph_params['connected'] == nx.is_connected(_graph)
                if check:
                    break
            return _graph
        elif self.graph_type == GraphType.BARABASI_ALBERT:
            return nx.barabasi_albert_graph(n, m=self.graph_params["m"], seed=rng)
        elif self.graph_type == GraphType.EUCLIDEAN:
            graph = self._generate_euclidean_graph(n, rng=rng)
            weighted_graph = graph.copy()

            # keep only k nearest neighbors
            k_nearest = self.graph_params.get("k_nearest")
            if k_nearest:
                graph = graph.to_directed()
                # add attribute to indicate the closest neighbors of each node
                for u in range(n):
                    neighbors_rank = sorted(range(n), key=lambda v: graph[u][v]["weight"] if v != u else 0)
                    for i, vth_closest in enumerate(neighbors_rank):
                        if vth_closest not in graph[u]:
                            continue
                        graph[u][vth_closest]["closest"] = i

                edges_to_remove = [(u, v) for u, v in graph.edges if graph[u][v]["closest"] > k_nearest]
                graph.remove_edges_from(edges_to_remove)
                graph = graph.to_undirected()

            return graph, weighted_graph
        raise ValueError(f"Invalid graph_type. It must be one of {list(GraphType.__members__.keys())}")
#
# def graph_generator(n_min, n_max, pos_lim, k_nearest=None, seed=None) -> nx.Graph:
#     n = np.random.randint(n_min, n_max + 1)
#     graph = make_complete_planar_graph(n=n, seed=seed, pos_lim=pos_lim).to_directed()
#
#     # add attribute to indicate the closest neighbors of each node
#     for u in range(n):
#         neighbors_rank = sorted(range(n), key=lambda v: graph[u][v]["weight"] if v != u else 0)
#         for i, vth_closest in enumerate(neighbors_rank):
#             if vth_closest not in graph[u]:
#                 continue
#             graph[u][vth_closest]["closest"] = i
#
#     # keep only k nearest neighbors
#     if k_nearest is not None:
#         edges_to_remove = [(u, v) for u, v in graph.edges if graph[u][v]["closest"] > k_nearest]
#         graph.remove_edges_from(edges_to_remove)
#
#     return graph
#
#
# def make_complete_planar_graph(n, seed: int = None, pos_lim: float = 1e3) -> nx.Graph:
#     """Returns a fully connected graph with xy positions for each
#     node and edge weights equal to pairwise distances.
#
#     Args:
#         n: Number of nodes in graph.
#         seed: Random seed for reproducibility. Defaults to None.
#
#     Returns:
#         Networkx complete graph with Euclidean distance weights.
#     """
#
#     np.random.seed(seed)
#
#     # Complete graph on points in xy-plane with pairwise distances as edge weights
#     graph = nx.complete_graph(n)
#
#     coords = np.random.rand(n, 2) * pos_lim
#
#     for ei, ej in graph.edges:
#         graph[ei][ej]["weight"] = np.sqrt(sum((coords[ei] - coords[ej]) ** 2))
#
#     for node in graph.nodes:
#         graph.nodes[node]["pos"] = coords[node, :]
#
#     return graph
#
#
# def generate_graph(n_min: int, n_max: int, graph_type: GraphType, graph_param: Union[float, int]):
#     n = np.random.randint(n_min, n_max + 1)
#     if graph_type == GraphType.ERDOS_RENYI:
#         return nx.erdos_renyi_graph(n, p=graph_param)
#     elif graph_type == GraphType.BARABASI_ALBERT:
#         return nx.barabasi_albert_graph(n, m=graph_param)
#     elif graph_type == GraphType.PLANAR:
#         graph = make_complete_planar_graph(n=n, seed=seed, pos_lim=pos_lim).to_directed()
#
#         # add attribute to indicate the closest neighbors of each node
#         for u in range(n):
#             neighbors_rank = sorted(range(n), key=lambda v: graph[u][v]["weight"] if v != u else 0)
#             for i, vth_closest in enumerate(neighbors_rank):
#                 if vth_closest not in graph[u]:
#                     continue
#                 graph[u][vth_closest]["closest"] = i
#
#         # keep only k nearest neighbors
#         if k_nearest is not None:
#             edges_to_remove = [(u, v) for u, v in graph.edges if graph[u][v]["closest"] > k_nearest]
#             graph.remove_edges_from(edges_to_remove)
#
#         return graph
#     raise ValueError("graph_type must be MVCEnv.ERDOS_RENYI or MVCEnv.BARABASI_ALBERT")
