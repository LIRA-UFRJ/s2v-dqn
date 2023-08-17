import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


K_CLOSEST = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class QNetwork(nn.Module):
#     def __init__(self, embed_dim, T, seed=None):
#         super().__init__()
#         if seed:
#             torch.manual_seed(seed)

#         self.embed_dim = embed_dim
#         self.T = T
#         self.G = None
#         self.num_nodes = 0
#         self.embedding_memo = dict()

#         # Define learnable parameters
#         self.theta1 = nn.Parameter(torch.randn(embed_dim))
#         self.theta2 = nn.Parameter(torch.randn((embed_dim, embed_dim)))
#         self.theta3 = nn.Parameter(torch.randn((embed_dim, embed_dim)))
#         self.theta4 = nn.Parameter(torch.randn(embed_dim))
#         self.theta5 = nn.Parameter(torch.randn(2*embed_dim))
#         self.theta6 = nn.Parameter(torch.randn((embed_dim, embed_dim)))
#         self.theta7 = nn.Parameter(torch.randn((embed_dim, embed_dim)))

#     def update_model_info(self, G):
#         self.G = G
#         self.num_nodes = G.number_of_nodes()
#         # TODO: check this memo
#         self.embedding_memo = dict()

#     def forward(self, xvs, vs) -> torch.Tensor:
#         action_values = []
#         # breakpoint()
#         for xv, v in zip(xvs, vs):
#             # Calculate embeddings
#             # shape = (batch_size, n)
#             mu = self._calc_embeddings(xv)
#             # Calculate Q values
#             # shape = (batch_size, 1)
#             action_value = self._calc_Q(mu, v.item())
#             action_values.append(action_value)

#         return torch.vstack(action_values)

#     def _calc_embeddings(self, xv):
#         """Calculate embeddings for each vertex, given current state"""
#         if xv in self.embedding_memo:
#             print('xv in self.embedding_memo')
#             return self.embedding_memo[xv]
        
#         G = self.G
#         mu = [[torch.zeros(self.embed_dim).to(device) for _ in self.G] for _ in range(self.T+1)]
#         for t in range(1, self.T+1):
#             for u in self.G:
#                 neighbors = list(self.G.neighbors(u))
#                 # TODO: filter K closest vertices
#                 neighbors = list(v for v in self.G.neighbors(u) if G[u][v]["closest"] <= K_CLOSEST)
# #                 print(torch.stack([mu[t-1][v] for v in neighbors], dim=0).sum(dim=0))
# #                 breakpoint()
#                 mu[t][u] = torch.nn.functional.relu(
#                     self.theta1 * xv[u] + \
#                     self.theta2 @ torch.stack([mu[t-1][v] for v in neighbors], dim=0).sum(dim=0) + \
#                     self.theta3 @ torch.stack(
#                         [torch.nn.functional.relu(self.theta4 * self.G[v][u]["weight"]) for v in neighbors],
#                         dim=0
#                     ).sum(dim=0)
#                 )
#         # TODO: check if this memoization is not shared between episodes
#         self.embedding_memo[xv] = mu[self.T]
#         return mu[self.T]

#     def _calc_Q(self, mu, v):
#         """Calculate Q(h(S), v)"""
#         # print('len(mu):', len(mu))
#         return torch.dot(
#             self.theta5,
#             torch.nn.functional.relu(
#                 torch.concat([
#                     self.theta6 @ torch.stack(mu, dim=0).sum(dim=0),
#                     self.theta7 @ mu[v]
#                 ],
#                 dim=0
#             ))
#         )

class MPNN(nn.Module):
#     def __init__(self, T=4, embed_dim=64, n_node_features=4, n_edge_features=1, bias=False):
    def __init__(self, embed_dim=64, T=4, n_node_features=4,
                 n_edge_features=1, bias=False):
        super().__init__()

        self.T = T
        self.embed_dim = embed_dim
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        
        self.embedding_layer = EmbeddingLayer(
            embed_dim=embed_dim,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            bias=bias
        )
        self.q_layer = QNetwork(embed_dim=embed_dim, bias=bias)
    
    def forward(self, state):
        # TODO: remove this
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(device, dtype=torch.float32)
        if state.dim() == 2:
            state = state.unsqueeze(0)

        n = state.shape[1]
        node_features = state[:, :, :self.n_node_features]
        adj = state[:, :, self.n_node_features:(self.n_node_features + n)]
        edge_features = state[:, :, (self.n_node_features + n):]
        
        # calculate node embeddings
        embeddings = torch.zeros(state.shape[0], state.shape[1], self.embed_dim).to(device, dtype=torch.float32)
        for _ in range(self.T):
            embeddings = self.embedding_layer(embeddings, adj, node_features, edge_features)

        # calculate \hat{Q} based on embeddings and given vertices
        q_hat = self.q_layer(embeddings)
        return q_hat

class EmbeddingLayer(nn.Module):
#     def __init__(self, embed_dim, n_node_features, n_edge_features, bias=False):
    def __init__(self, embed_dim, n_node_features, n_edge_features=1, bias=False):
        super().__init__()
        self.theta1 = nn.Linear(n_node_features, embed_dim, bias=bias)
        self.theta2 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.theta3 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.theta4 = nn.Linear(n_edge_features, embed_dim, bias=bias)
        
    def forward(self, prev_embeddings, adj, node_features, edge_features):
        # node_features.shape = (batch_size, n_vertices, n_node_features)
        # x1.shape = (batch_size, n_vertices, embed_dim)
#         breakpoint()
        x1 = self.theta1(node_features)

        # adj.shape = (batch_size, n_vertices, n_vertices)
        # prev_embeddings.shape = (batch_size, n_vertices, embed_dim)
        # x2.shape = (batch_size, n_vertices, embed_dim)
        x2 = self.theta2(torch.matmul(adj, prev_embeddings))

        # edge_features.shape = (batch_size, n_vertices, n_vertices, n_edge_features)
        # x4.shape = (batch_size, n_vertices, n_vertices, embed_dim)
        if edge_features.dim() == 3:
            edge_features = edge_features.unsqueeze(-1)
        x4 = F.relu(self.theta4(edge_features))
        
        # adj.shape = (batch_size, n_vertices, n_vertices)
        # x4.shape = (batch_size, n_vertices, n_vertices, embed_dim)
        # sum_neighbor_edge_embeddings.shape = (batch_size, n_vertices, embed_dim)
        # x3.shape = (batch_size, n_vertices, embed_dim)
        sum_neighbor_edge_embeddings = (adj.unsqueeze(-1) * x4).sum(dim=-2)
        x3 = self.theta3(sum_neighbor_edge_embeddings)

        # ret.shape = (batch_size, n_vertices, embed_dim)
        ret = F.relu(x1 + x2 + x3)

        return ret

class QNetwork(nn.Module):
    '''
    Given node embeddings, calculate Q_hat for all vertices
    '''
    def __init__(self, embed_dim, bias=False):
        super().__init__()
        self.theta5 = nn.Linear(2*embed_dim, 1, bias=bias)
        self.theta6 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.theta7 = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, embeddings):
        # embeddings.shape = (batch_size, n_vertices, embed_dim)
        # sum_embeddings.shape = (batch_size, embed_dim)
        # x6.shape = (batch_size, embed_dim)
        sum_embeddings = embeddings.sum(dim=1)
        x6 = self.theta6(sum_embeddings)
        
        # repeat graph embedding for all vertices
        # x6.shape = (batch_size, embed_dim)
        # embeddings.shape[1] = n_vertices
        # x6_repeated.shape = (batch_size, n_vertices, embed_dim)
        x6_repeated = x6.unsqueeze(1).repeat(1, embeddings.shape[1], 1)
        
        # embeddings.shape = (batch_size, n_vertices, embed_dim)
        # x7.shape = (batch_size, n_vertices, embed_dim)
        x7 = self.theta7(embeddings)
        
        # x6.shape = x7.shape = (batch_size, n_vertices, embed_dim)
        # features.shape = (batch_size, n_vertices, 2*embed_dim)
        # x5.shape = (batch_size, n_vertices, 1)
        features = F.relu(torch.cat([x6, x7], dim=-1))
        x5 = self.theta5(features)
        
        # out.shape = (batch_size, n_vertices)
        out = x5.squeeze(-1)
        
        return out        
        

# class QNetworkLayers(nn.Module):
#     def __init__(self, embed_dim, T, seed=None):
#         super().__init__()
#         if seed:
#             torch.manual_seed(seed)

#         self.embed_dim = embed_dim
#         self.T = T
#         self.G = None
#         self.num_nodes = 0
#         self.embedding_memo = dict()

#         # Define learnable parameters
#         self.theta1 = nn.Linear(1, embed_dim, bias=False)
#         self.theta2 = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.theta3 = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.theta4 = nn.Linear(1, embed_dim, bias=False)
#         self.theta5 = nn.Linear(2*embed_dim, 1, bias=False)
#         self.theta6 = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.theta7 = nn.Linear(embed_dim, embed_dim, bias=False)

#     def update_model_info(self, G):
#         self.G = G
#         self.num_nodes = G.number_of_nodes()
#         # TODO: check this memo
#         self.embedding_memo = dict()

#     def forward(self, xvs, vs) -> torch.Tensor:
#         action_values = []
#         for xv, v in zip(xvs, vs.numpy()):
#             # Calculate embeddings
#             # shape = (batch_size, n)
#             mu = self._calc_embeddings(xv)
#             # Calculate Q values
#             # shape = (batch_size, 1)
#             vertex = v[0] if isinstance(v, np.ndarray) else v
#             action_value = self._calc_Q(mu, vertex)
#             action_values.append(action_value)

#         return torch.vstack(action_values)

#     def _calc_embeddings(self, xv):
#         """Calculate embeddings for each vertex, given current state"""
#         if xv in self.embedding_memo:
#             print('xv in self.embedding_memo')
#             return self.embedding_memo[xv]
        
#         G = self.G
#         mu = [[torch.zeros(self.embed_dim) for _ in self.G] for _ in range(self.T+1)]
#         for t in range(1, self.T+1):
#             for u in self.G:
#                 neighbors = list(self.G.neighbors(u))
#                 # TODO: filter K closest vertices
#                 neighbors = list(v for v in self.G.neighbors(u) if G[u][v]["closest"] <= K_CLOSEST)
#                 mu[t][u] = torch.nn.functional.relu(
#                     self.theta1(xv[u]) + \
#                     self.theta2(torch.stack([mu[t-1][v] for v in neighbors], dim=0).sum(dim=0)) + \
#                     self.theta3(torch.stack(
#                         [torch.nn.functional.relu(self.theta4(self.G[v][u]["weight"])) for v in neighbors],
#                         dim=0
#                     ).sum(dim=0))
#                 )
#         # TODO: check if this memoization is not shared between episodes
#         self.embedding_memo[xv] = mu[self.T]
#         return mu[self.T]

#     def _calc_Q(self, mu, v):
#         """Calculate Q(h(S), v)"""
#         # print('len(mu):', len(mu))
#         return torch.dot(
#             self.theta5,
#             torch.nn.functional.relu(
#                 torch.concat([
#                     self.theta6 @ torch.stack(mu, dim=0).sum(dim=0),
#                     self.theta7 @ mu[v]
#                 ],
#                 dim=0
#             ))
#         )
