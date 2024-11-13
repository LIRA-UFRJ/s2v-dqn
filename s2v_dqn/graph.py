import numpy as np


class Graph:
    def __init__(self, n):
        self.n = n
        self.adj_matrix = np.zeros((n, n))
        self.adj_list = [[] for _ in range(n)]

    def add_edge(self, u, v, directed=False):
        if self.adj_matrix[u][v] == 0:
            self.adj_matrix[u][v] = 1
            self.adj_list[u].append(v)
        if not directed and self.adj_matrix[v][u] == 0:
            self.adj_matrix[u][v] = 1
            self.adj_list[u].append(v)
