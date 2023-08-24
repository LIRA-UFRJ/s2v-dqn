import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


K_CLOSEST = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MPNN(nn.Module):
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
    '''
    Calculate embeddings for all vertices
    '''
    def __init__(self, embed_dim, n_node_features, n_edge_features=1, bias=False):
        super().__init__()
        self.theta1 = nn.Linear(n_node_features, embed_dim, bias=bias)
        self.theta2 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.theta3 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.theta4 = nn.Linear(n_edge_features, embed_dim, bias=bias)
        
    def forward(self, prev_embeddings, adj, node_features, edge_features):
        # node_features.shape = (batch_size, n_vertices, n_node_features)
        # x1.shape = (batch_size, n_vertices, embed_dim)
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
        sum_neighbor_edge_embeddings = (adj.unsqueeze(-1) * x4).sum(dim=2)
        
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
        features = F.relu(torch.cat([x6_repeated, x7], dim=-1))
        x5 = self.theta5(features)
        
        # out.shape = (batch_size, n_vertices)
        out = x5.squeeze(-1)
        
        return out        
