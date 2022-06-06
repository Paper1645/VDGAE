import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class AffinityLayer_Cora(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        super(AffinityLayer_Cora, self).__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_vert = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)
        self.linear_edge = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)

        self.attn = nn.Linear(in_features=n_heads, out_features=n_heads, bias=False)  # Linear layer to compute the attention score e_ij
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=-2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vertex: torch.Tensor, edge: torch.Tensor):
        '''
            vertex is the input nodes embeddings of shape [batch, n_nodes, in_features]
            edge is the input edges embeddings of shape [batch, n_edges, in_features]
            inci_mat is the incidence matrix of the graph, need to be transposed
        '''
        n_nodes = vertex.shape[0]  # number of nodes
        n_edges = edge.shape[0]  # number of edges

        g_vertex = self.linear_vert(vertex).view(n_nodes, self.n_heads, self.n_hidden)  # the initial transformation, g = Wh, [batch_size, n_node, n_head, n_hidden]
        g_edge = self.linear_edge(edge).view(n_edges, self.n_heads, self.n_hidden)  # [batch_size, n_edge, n_head, n_hidden]

        g_head = []

        for i in range(self.n_heads):
            g_head.append(torch.mm(g_edge[:, i, :], g_vertex[:, i, :].t()).unsqueeze(dim=-1))  # [batch_size, n_edge, n_node, 1]

        g_concat = torch.cat(g_head, dim=-1)  # [batch_size, n_edge, n_node, n_head]

        e = self.activation(self.attn(g_concat))  # e_ij = LeakyReLU(a^T [g_i||g_j]), the shape of e is [batch_size, n_edge, n_node, n_head]
        e = e.squeeze(dim=-1).sum(dim=-1)

        return e


class AffinityLayer_Edge(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        super(AffinityLayer_Edge, self).__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_vert = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)
        self.linear_edge = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)

        self.attn = nn.Linear(in_features=n_heads, out_features=n_heads, bias=False)  # Linear layer to compute the attention score e_ij
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=-2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vertex: torch.Tensor, edge: torch.Tensor):
        '''
            vertex is the input nodes embeddings of shape [batch, n_nodes, in_features]
            edge is the input edges embeddings of shape [batch, n_edges, in_features]
            inci_mat is the incidence matrix of the graph, need to be transposed
        '''
        batch_size = vertex.shape[0]
        n_nodes = vertex.shape[1]  # number of nodes
        n_edges = edge.shape[1]  # number of edges

        g_vertex = self.linear_vert(vertex).view(batch_size, n_nodes, self.n_heads, self.n_hidden)  # the initial transformation, g = Wh, [batch_size, n_node, n_head, n_hidden]
        g_edge = self.linear_edge(edge).view(batch_size, n_edges, self.n_heads, self.n_hidden)  # [batch_size, n_edge, n_head, n_hidden]

        # calculate similarity(edge_i, vertex_j), is the QK in the attention formula
        g_head = []
        for i in range(self.n_heads):
            g_head.append(torch.bmm(g_edge[:, :, i, :], g_vertex[:, :, i, :].permute(0, 2, 1)).unsqueeze(dim=-1))  # [batch_size, n_edge, n_node, 1]
        g_concat = torch.cat(g_head, dim=-1)  # [batch_size, n_edge, n_node, n_head]
        e = self.activation(self.attn(g_concat))  # e_ij = LeakyReLU(a^T [g_i||g_j]), the shape of e is [batch_size, n_edge, n_node, n_head]
        e = e.squeeze(dim=-1).sum(dim=-1)

        return e


class AffinityNodeLayer_Cora(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        super(AffinityNodeLayer_Cora, self).__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_vert = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)
        self.linear_edge = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)

        self.attn = nn.Linear(in_features=n_heads, out_features=n_heads, bias=False)  # Linear layer to compute the attention score e_ij
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=-2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vertex: torch.Tensor):
        '''
            vertex is the input nodes embeddings of shape [batch, n_nodes, in_features]
            edge is the input edges embeddings of shape [batch, n_edges, in_features]
            inci_mat is the incidence matrix of the graph, need to be transposed
        '''
        n_nodes = vertex.shape[0]  # number of nodes

        g_vertex = self.linear_vert(vertex).view(n_nodes, self.n_heads, self.n_hidden)  # the initial transformation, g = Wh, [batch_size, n_node, n_head, n_hidden]
        # g_vertex_H = self.linear_vert(vertex_H).view(n_nodes, self.n_heads, self.n_hidden)

        g_head = []

        for i in range(self.n_heads):
            g_head.append(torch.mm(g_vertex[:, i, :], g_vertex[:, i, :].t()).unsqueeze(dim=-1))  # [batch_size, n_edge, n_node, 1]

        g_concat = torch.cat(g_head, dim=-1)  # [batch_size, n_edge, n_node, n_head]

        e = self.activation(self.attn(g_concat))  # e_ij = LeakyReLU(a^T [g_i||g_j]), the shape of e is [batch_size, n_edge, n_node, n_head]
        e = e.squeeze(dim=-1).sum(dim=-1)

        return e


