import networkx as nx
from pulp import LpVariable, LpInteger, LpMinimize, LpProblem, lpSum, PULP_CBC_CMD, value


class MVCSolver:
    @staticmethod
    def get_solution(graph: nx.Graph, exact_solution_max_size: int = 0) -> float:
        if graph.number_of_nodes() <= exact_solution_max_size:
            return MVCSolver._get_exact_solution(graph)
        else:
            raise NotImplemented("Heuristic solution not implemented yet")

    @staticmethod
    def _get_exact_solution(graph: nx.Graph):
        """
        Exact solver for the MVC problem, using MILP formulation
        :param env: MVCEnv representing instance of the problem
        :return: optimal solution for the instance
        """
        node_names = list(map(str, range(graph.number_of_nodes())))
        node_vars = LpVariable.dicts("Node", node_names, 0, 1, LpInteger)

        prob = LpProblem("myProblem", LpMinimize)
        prob += lpSum(node_vars)

        for (u, v) in graph.edges:
            prob += node_vars[str(u)] + node_vars[str(v)] >= 1

        prob.solve(PULP_CBC_CMD(msg=False))
        return value(prob.objective)
