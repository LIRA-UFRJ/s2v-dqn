import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QNetwork(nn.Module):
    def __init__(self, embed_dim, T, seed=None):
        super().__init__()
        if seed:
            torch.manual_seed(seed)

        self.embed_dim = embed_dim
        self.T = T
        self.G = None
        self.num_nodes = 0
        self.embedding_memo = dict()

        # Define learnable parameters
        self.theta1 = nn.Parameter(torch.randn(embed_dim))
        self.theta2 = nn.Parameter(torch.randn((embed_dim, embed_dim)))
        self.theta3 = nn.Parameter(torch.randn((embed_dim, embed_dim)))
        self.theta4 = nn.Parameter(torch.randn(embed_dim))
        self.theta5 = nn.Parameter(torch.randn(2*embed_dim))
        self.theta6 = nn.Parameter(torch.randn((embed_dim, embed_dim)))
        self.theta7 = nn.Parameter(torch.randn((embed_dim, embed_dim)))

    def update_model_info(self, G):
        self.G = G
        self.num_nodes = G.number_of_nodes()
        # TODO: check this memo
        self.embedding_memo = dict()

    def forward(self, xvs, vs) -> torch.Tensor:
        action_values = []
        # print('xvs:', xvs)
        # print('vs:', vs)
        # print('xvs.shape:', xvs.shape)
        # print('vs.shape:', vs.shape)
        for xv, v in zip(xvs, vs.numpy()):
            # Calculate embeddings
            # shape = (batch_size, n)
            mu = self._calc_embeddings(xv)
            # Calculate Q values
            # shape = (batch_size, 1)
            vertex = v[0] if isinstance(v, np.ndarray) else v
            action_value = self._calc_Q(mu, vertex)
            action_values.append(action_value)

        return torch.vstack(action_values)

    def _calc_embeddings(self, xv):
        """Calculate embeddings for each vertex, given current state"""
        if xv in self.embedding_memo:
            print('xv in self.embedding_memo')
            return self.embedding_memo[xv]
        
        G = self.G
        mu = [[torch.zeros(self.embed_dim) for _ in self.G] for _ in range(self.T+1)]
        for t in range(1, self.T+1):
            for u in self.G:
                neighbors = list(self.G.neighbors(u))
                mu[t][u] = torch.nn.functional.relu(
                    self.theta1 * xv[u] + \
                    self.theta2 @ torch.stack([mu[t-1][v] for v in neighbors], dim=0).sum(dim=0) + \
                    self.theta3 @ torch.stack(
                        [torch.nn.functional.relu(self.theta4 * self.G[v][u]["weight"]) for v in neighbors],
                        dim=0
                    ).sum(dim=0)
                )
        # TODO: check if this memoization is not shared between episodes
        self.embedding_memo[xv] = mu[self.T]
        return mu[self.T]

    def _calc_Q(self, mu, v):
        """Calculate Q(h(S), v)"""
        # print('len(mu):', len(mu))
        return torch.dot(
            self.theta5,
            torch.nn.functional.relu(
                torch.concat([
                    self.theta6 @ torch.stack(mu, dim=0).sum(dim=0),
                    self.theta7 @ mu[v]
                ],
                dim=0
            ))
        )
