from typing import Callable, List

import networkx as nx
import numpy as np

def solve(
    G: nx.Graph,
    cost_fn: Callable[[nx.Graph, int, int], float] = lambda G, u, v: G[u][v]["weight"]
) -> float:
    n = G.number_of_nodes()
    
    # dp[mask][u] represents the min tour distance starting from vertex 0,
    # ending at vertex u and using only vertices encoded in mask (bits set to 1)
    dp = [[float('inf')] * n for _ in range(1<<n)]
    dp[0][0] = 0.0

    # for each mask (vertices in the partial solution)
    for mask in range(1, (1<<n), 2):
        # for each possible new vertex j to be added
        for j in range(n):
            if (mask & (1<<j)):
                continue
            # for each possible vertex k that is the last in partial solution (mask)
            for k in range(n):
                if (mask & (1<<k)):
                    dp[mask][j] = min(dp[mask][j], dp[mask^(1<<k)][k] + cost_fn(G, k, j))

	# returning to first vertex 0
    ans = min((dp[((1<<n)-1)-(1<<k)][k] + cost_fn(G, k, 0)) for k in range(1, n))

    return ans

def solve2(
    G: nx.Graph,
    cost_fn: Callable[[nx.Graph, int, int], float] = lambda G, u, v: G[u].get(v, {}).get("weight", float('inf'))
) -> float:
    n = G.number_of_nodes()
    
    # dp[mask][u] represents the min tour distance starting from vertex 0,
    # ending at vertex u and using only vertices encoded in mask (bits set to 1)
    dp = [[float('inf')] * n for _ in range(1<<n)]
    dp[0][0] = 0.0

    # for each mask (vertices in the partial solution)
    for mask in range(1<<n):
        # for each possible new vertex j to be added
        for j in range(n):
            if (mask & (1<<j)):
                continue
            # for each possible vertex k that is the last in partial solution (mask)
            # dp[mask][j] = min(
            #     [dp[mask^(1<<k)][k] + cost_fn(G, k, j) for k in range(n) if (mask & (1<<j))]
            #     + [float('inf')]
            # )
            for k in range(n):
                dp[mask | (1<<j)][k] = min(dp[mask | (1<<j)][k], dp[mask][j] + cost_fn(G, j, k))
            
    return dp[(1<<n) - 1][0]
