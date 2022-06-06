import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import *

torch.cuda.manual_seed(1)

class GetEdge_Cora(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GetEdge_Cora, self).__init__(**kwargs)
        self.weight = self.glorot_init(input_dim, output_dim)
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        output_dim = 1
        init_range = np.sqrt(1.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range

        return nn.Parameter(initial)

    def forward(self, inputs, adj):
        x = inputs
        x = torch.bmm(adj, x)
        # outputs = self.activation(x)
        return x #outputs


class Decode_Cora(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, out_features: int, n_heads: int, dropout: float):
        super(Decode_Cora, self).__init__()
        self.affinity_layer = AffinityLayer_Edge(in_features=in_features, out_features=n_hidden, n_heads=n_heads, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, vert: torch.Tensor, edge: torch.Tensor):
        # x = self.dropout(x)
        x = self.affinity_layer(vert, edge)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class DecodeNode_Cora(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, out_features: int, n_heads: int, dropout: float):
        super(DecodeNode_Cora, self).__init__()
        self.affinity_layer = AffinityNodeLayer_Cora(in_features=in_features, out_features=n_hidden, n_heads=n_heads, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, vert: torch.Tensor):
        # x = self.dropout(x)
        x = self.affinity_layer(vert)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class EncoderEdgeLayer_Cora(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, device=None, **kwargs):
        super(EncoderEdgeLayer_Cora, self).__init__(**kwargs)
        self.weight = self.glorot_init(input_dim, output_dim, device)
        self.activation = activation

    def glorot_init(self, input_dim, output_dim, device):
        init_range = np.sqrt(1.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range

        return nn.Parameter(initial).to(device)

    def forward(self, inputs, inci_mat):
        adj = torch.mm(inci_mat, inci_mat.t())
        x = inputs

        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs


class EncoderNodeLayer_Cora(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(EncoderNodeLayer_Cora, self).__init__(**kwargs)
        self.weight = self.glorot_init(input_dim, output_dim)
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(1.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 1.0 * init_range - init_range

        return nn.Parameter(initial).to(device)

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs


class EncoderEdge_Cora(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, device):
        super(EncoderEdge_Cora, self).__init__()

        self.layer_1 = EncoderEdgeLayer_Cora(input_dim=input_dim, output_dim= hidden_dim * 2 + out_dim, device=device)
        # self.layer_2 = GraphConvSparse_Cora_Edge_Inci(input_dim=hidden_dim, output_dim=hidden_dim + out_dim)
        # self.layer_3 = GraphConvSparse_Cora_Edge_Inci(input_dim=hidden_dim, output_dim=out_dim, activation=lambda x: x)
        # self.merge = nn.Linear(in_features=hidden_dim * 1 + out_dim, out_features=out_dim)


    def forward(self, x, adj):
        x_1 = self.layer_1(x, adj)
        # x_2 = self.layer_2(x_1, adj)
        # x_3 = self.layer_3(x_2, adj)
        # out = torch.cat([x_1, x_2], dim=-1)
        # out = torch.cat([out, x_3], dim=-1)

        # out = self.merge(torch.cat([out, x_3], dim=-1))
        return x_1


class EncoderNode_Cora(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(EncoderNode_Cora, self).__init__()


        self.layer_1 = EncoderNodeLayer_Cora(input_dim=input_dim, output_dim= hidden_dim * 2 + out_dim)
        # self.layer_2 = GraphConvSparse_Cora_Edge_Inci(input_dim=hidden_dim, output_dim=hidden_dim + out_dim)
        # self.layer_3 = GraphConvSparse_Cora_Edge_Inci(input_dim=hidden_dim, output_dim=out_dim, activation=lambda x: x)
        # self.merge = nn.Linear(in_features=hidden_dim * 1 + out_dim, out_features=out_dim)


    def forward(self, x, adj):
        x_1 = self.layer_1(x, adj)
        # x_2 = self.layer_2(x_1, adj)
        # x_3 = self.layer_3(x_2, adj)
        # out = torch.cat([x_1, x_2], dim=-1)
        # out = torch.cat([out, x_3], dim=-1)

        # out = self.merge(torch.cat([out, x_3], dim=-1))
        return x_1






