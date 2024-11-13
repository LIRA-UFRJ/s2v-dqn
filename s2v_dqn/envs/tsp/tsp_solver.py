from typing import Callable, Union, Tuple, List

import networkx as nx
import numpy as np
from concorde.tsp import TSPSolver as ConcordeTSPSolver
from python_tsp.exact import solve_tsp_dynamic_programming


class TSPSolver:
    @staticmethod
    def get_solution(graph: nx.Graph, exact_solution_max_size: int = 0, return_path: bool = False) -> float:
        n = graph.number_of_nodes()
        if n <= exact_solution_max_size:
            dist_matrix = np.array([[graph[u].get(v, {}).get('weight', float('inf')) for v in range(n)] for u in range(n)])
            solution = solve_tsp_dynamic_programming(dist_matrix)
            if not return_path:
                solution = solution[1]
            return solution
        else:
            return TSPSolver._get_concorde_solution(graph, return_path)

    @staticmethod
    def _get_concorde_solution(graph: nx.Graph, time_bound: float = 5.0, return_path: bool = False) -> Union[float, Tuple[float, List[int]]]:
        pos = nx.get_node_attributes(graph, 'coords')
        pos_numpy = np.array(list(pos.values()))
        solver = ConcordeTSPSolver.from_data(pos_numpy[:, 0], pos_numpy[:, 1], "EUC_2D")

        # # Redirect stdout
        # save_stdout = sys.stdout
        # save_stderr = sys.stderr
        # sys.stdout = open(os.devnull, 'w')

        solution = solver.solve(time_bound=time_bound, verbose=False)

        # Recover original stdout
        # sys.stdout = save_stdout

        # print(solution.optimal_value)
        def dist(u: int, v: int):
            return graph[u].get(v, {}).get("weight", float("inf"))
        actual_solution = sum([dist(u, v) for (u, v) in zip(solution.tour[:-1], solution.tour[1:])]) + dist(solution.tour[-1], solution.tour[0])
        if return_path:
            return actual_solution, solution.tour
        else:
            return actual_solution

    @staticmethod
    def _solve_exact(graph: nx.Graph,
                     cost_fn: Callable[[nx.Graph, int, int], float] = lambda graph, u, v: graph[u][v]["weight"]) -> float:
        n = graph.number_of_nodes()

        # dp[mask][u] represents the min tour distance starting from vertex 0,
        # ending at vertex u and using only vertices encoded in mask (bits set to 1)
        dp = [[float('inf')] * n for _ in range(1 << n)]
        dp[0][0] = 0.0

        # for each mask (vertices in the partial solution)
        for mask in range(1, (1 << n), 2):
            # for each possible new vertex j to be added
            for j in range(n):
                if mask & (1 << j):
                    continue
                # for each possible vertex k that is the last in partial solution (mask)
                for k in range(n):
                    if mask & (1 << k):
                        dp[mask][j] = min(dp[mask][j], dp[mask ^ (1 << k)][k] + cost_fn(graph, k, j))

        # returning to first vertex 0
        ans = min((dp[((1 << n) - 1) - (1 << k)][k] + cost_fn(graph, k, 0)) for k in range(1, n))

        return ans

