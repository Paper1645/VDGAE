import math
import random
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import igraph
import pdb
from GCN.gcn import *
from GCN.layers import *
import networkx as nx
from sklearn.metrics import f1_score
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP
from torch_scatter import scatter_add
import scipy
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter, Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.utils.num_nodes import maybe_num_nodes
import umap


# class VDGAE(nn.Module):
#     def __init__(self, max_n, max_edge_n, n_vertex_type, feature_dimension, hidden_dim=501, z_dim=56, with_neighbor=False, with_vertex=False, node_classification=False):
#         super(VDGAE, self).__init__()
#         self.max_n_vertex = max_n  # maximum number of vertices
#         self.max_n_edge = max_edge_n  # max number of edges
#         self.n_vertex_type = n_vertex_type  # number of vertex types
#         self.hs = hidden_dim  # hidden state size of each vertex
#         self.nz = z_dim  # size of latent representation z
#         self.gs = hidden_dim  # size of graph state
#         self.with_neighbor = with_neighbor
#         self.with_vertex = with_vertex
#         self.node_classification = node_classification
#         self.max_n = max_n
#
#         # 0. encoding part, cora: 1433, citeseer: 3703, Cornell: 1703, WikiCS: 300
#         self.FEATURE_NODE = feature_dimension
#
#         if not with_vertex:
#             self.FEATURE = max_n
#         else:
#             self.FEATURE = feature_dimension
#             self.gnn_edge_T = GetEdge_Cora(input_dim=self.FEATURE, output_dim=self.FEATURE)
#             self.gnn_edge_H = GetEdge_Cora(input_dim=self.FEATURE, output_dim=self.FEATURE)
#
#         # 0.1 vertex encoder
#         self.mu_vert = nn.Sequential(
#             nn.Linear(in_features=self.FEATURE_NODE, out_features=z_dim + hidden_dim * 2),
#             nn.ReLU(),
#         )
#
#         self.logvar_vert = nn.Sequential(
#             nn.Linear(in_features=self.FEATURE_NODE, out_features=z_dim + hidden_dim * 2),
#             nn.ReLU()
#         )
#
#         # 0.2 edge encoder
#         if not with_neighbor:
#             self.mu_edge_T = nn.Sequential(
#                 nn.Linear(in_features=self.FEATURE, out_features=z_dim + hidden_dim * 2),
#                 nn.ReLU(),
#             )
#
#             self.logvar_edge_T = nn.Sequential(
#                 nn.Linear(in_features=self.FEATURE, out_features=z_dim + hidden_dim * 2),
#                 nn.ReLU()
#             )
#
#             self.mu_edge_H = nn.Sequential(
#                 nn.Linear(in_features=self.FEATURE, out_features=z_dim + hidden_dim * 2),
#                 nn.ReLU(),
#             )
#
#             self.logvar_edge_H = nn.Sequential(
#                 nn.Linear(in_features=self.FEATURE, out_features=z_dim + hidden_dim * 2),
#                 nn.ReLU()
#             )
#         else:
#             self.mu_edge_T = EncoderEdge_Cora(input_dim=self.FEATURE, hidden_dim=hidden_dim, out_dim=z_dim)
#             self.logvar_edge_T = EncoderEdge_Cora(input_dim=self.FEATURE, hidden_dim=hidden_dim, out_dim=z_dim)
#             self.mu_edge_H = EncoderEdge_Cora(input_dim=self.FEATURE, hidden_dim=hidden_dim, out_dim=z_dim)
#             self.logvar_edge_H = EncoderEdge_Cora(input_dim=self.FEATURE, hidden_dim=hidden_dim, out_dim=z_dim)
#
#         # 1. decoding part
#         # 1.1 edge decoder
#         self.gnn_inci_T = Decode_Cora(in_features=hidden_dim * 2 + z_dim, n_hidden=hidden_dim, out_features=hidden_dim * 2, n_heads=8, dropout=0.1)
#         self.gnn_inci_H = Decode_Cora(in_features=hidden_dim * 2 + z_dim, n_hidden=hidden_dim, out_features=hidden_dim * 2, n_heads=8, dropout=0.1)
#
#         # 1.2 vertex decoder
#         self.add_vertex = nn.Sequential(
#             nn.Linear(self.hs * 2 + z_dim, self.hs * 2),
#             nn.ReLU(),
#             nn.Linear(self.hs * 2, self.n_vertex_type),
#         )
#
#         # 2. others
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#
#         self.softmax = nn.Softmax(dim=-1)
#         self.vertex_criterion = nn.CrossEntropyLoss()
#         self.adj_criterion = nn.BCELoss()
#         self.matrix_criterion = nn.MSELoss(reduction='sum')
#
#         self.layerNorm_vert = nn.LayerNorm([self.max_n_vertex, self.FEATURE])
#         self.layerNorm_edge_T = nn.LayerNorm([self.max_n_edge, self.FEATURE])
#         self.layerNorm_edge_H = nn.LayerNorm([self.max_n_edge, self.FEATURE])
#         self.layerNorm_vert_de = nn.LayerNorm([self.max_n_vertex, hidden_dim * 2 + z_dim])
#         self.layerNorm_edge_T_de = nn.LayerNorm([self.max_n_edge, hidden_dim * 2 + z_dim])
#         self.layerNorm_edge_H_de = nn.LayerNorm([self.max_n_edge, hidden_dim * 2 + z_dim])
#
#         self.layerNorm_mean_vert = nn.LayerNorm([max_n, z_dim + hidden_dim * 2])
#         self.layerNorm_mean_edge_T = nn.LayerNorm([max_edge_n, z_dim + hidden_dim * 2])
#         self.layerNorm_mean_edge_H = nn.LayerNorm([max_edge_n, z_dim + hidden_dim * 2])
#         self.layerNorm_logvar_vert = nn.LayerNorm([max_n, z_dim + hidden_dim * 2])
#         self.layerNorm_logvar_edge_T = nn.LayerNorm([max_edge_n, z_dim + hidden_dim * 2])
#         self.layerNorm_logvar_edge_H = nn.LayerNorm([max_edge_n, z_dim + hidden_dim * 2])
#
#     def get_device(self):
#         if self.device is None:
#             self.device = next(self.parameters()).device
#         return self.device
#
#     def encode(self, edges, vert_feature, inci_mat_T, inci_mat_H):
#         if not self.with_vertex:
#             edges_attn_T = F.one_hot(edges[:, 0], inci_mat_T.size(1)).float().to(self.get_device())
#             edges_attn_H = F.one_hot(edges[:, 1], inci_mat_T.size(1)).float().to(self.get_device())
#         else:
#             edges_attn_T = self.gnn_edge_T(vert_feature, inci_mat_T)
#             edges_attn_H = self.gnn_edge_H(vert_feature, inci_mat_H)
#
#         if not self.with_neighbor:
#             mean_vert = self.mu_vert(vert_feature)
#             logvar_vert = self.logvar_vert(vert_feature)
#
#             mean_edge_T = self.mu_edge_T(edges_attn_T)
#             logvar_edge_T = self.logvar_edge_T(edges_attn_T)
#
#             mean_edge_H = self.mu_edge_H(edges_attn_H)
#             logvar_edge_H = self.logvar_edge_H(edges_attn_H)
#         else:
#             mean_vert = self.mu_vert(vert_feature)
#             logvar_vert = self.logvar_vert(vert_feature)
#
#             mean_edge_T = self.mu_edge_T(edges_attn_T, inci_mat_T)
#             logvar_edge_T = self.logvar_edge_T(edges_attn_T, inci_mat_T)
#
#             mean_edge_H = self.mu_edge_H(edges_attn_H, inci_mat_H)
#             logvar_edge_H = self.logvar_edge_H(edges_attn_H, inci_mat_H)
#
#         mean_vert = self.layerNorm_mean_vert(mean_vert)
#         mean_edge_T = self.layerNorm_mean_edge_T(mean_edge_T)
#         mean_edge_H = self.layerNorm_mean_edge_H(mean_edge_H)
#         logvar_vert = self.layerNorm_logvar_vert(logvar_vert)
#         logvar_edge_T = self.layerNorm_logvar_edge_T(logvar_edge_T)
#         logvar_edge_H = self.layerNorm_logvar_edge_H(logvar_edge_H)
#
#         z_vertex = self._GaussianNoise(mean_vert.size()) * torch.exp(0.5 * logvar_vert) + mean_vert
#         z_edge_T = self._GaussianNoise(mean_edge_T.size()) * torch.exp(0.5 * logvar_edge_T) + mean_edge_T
#         z_edge_H = self._GaussianNoise(mean_edge_H.size()) * torch.exp(0.5 * logvar_edge_H) + mean_edge_H
#
#         return (mean_vert, mean_edge_T, mean_edge_H), (logvar_vert, logvar_edge_T, logvar_edge_H), (z_vertex, z_edge_T, z_edge_H)
#
#     def _GaussianNoise(self, size):
#         gaussian_noise = torch.rand(size).to(device)
#         return gaussian_noise
#
#     def _reparameterize(self, mu, logvar, eps_scale=0.01):
#         # return z ~ N(mu, std)
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = torch.randn_like(std) * eps_scale
#             return eps.mul(std).add_(mu)
#         else:
#             return mu
#
#     def calculate_accuracy(self, z):
#         vertex_pred, inci_pred_T, inci_pred_H = self._decode(z)
#
#         inci_pred_T[inci_pred_T.max(dim=-1)[0] < 0.25] = 0.
#         inci_pred_H[inci_pred_H.max(dim=-1)[0] < 0.25] = 0.
#
#         inci_T = F.one_hot(torch.argmax(inci_pred_T, dim=-1, keepdim=False), num_classes=self.max_n_vertex) * -1
#         inci_H = F.one_hot(torch.argmax(inci_pred_H, dim=-1, keepdim=False), num_classes=self.max_n_vertex)
#         inci_pred = inci_T + inci_H
#
#         v_type_pred = torch.max(vertex_pred, dim=-1)[1].view(z[0].size(0), -1)
#
#         return v_type_pred, inci_pred
#
#     def direction_prediction(self, z, edges, edge_label):
#         v_feat = z[0]
#         e_feat_T = z[1]
#         e_feat_H = z[2]
#
#         edge_label = edge_label.cpu().tolist()
#         edges = edges.cpu().tolist()
#
#         acc = 0
#         count = 0
#         pred_lb = []
#
#         v_list = list(range(v_feat.size(0)))
#
#         for e, l, et, eh in zip(edges, edge_label, e_feat_T, e_feat_H):
#             count = count + 1
#
#             e_feat_tail = et.unsqueeze(dim=0)
#             e_feat_head = eh.unsqueeze(dim=0)
#
#             attn_tail = self.gnn_inci_T(v_feat, e_feat_tail)
#             attn_head = self.gnn_inci_H(v_feat, e_feat_head)
#
#             attn_tail[attn_tail.max(dim=-1)[0] < 0.25] = 0.
#             attn_head[attn_head.max(dim=-1)[0] < 0.25] = 0.
#
#             inci_T = torch.argmax(attn_tail, dim=-1, keepdim=False)
#             inci_H = torch.argmax(attn_head, dim=-1, keepdim=False)
#
#             e_t = v_list[inci_T]
#             e_h = v_list[inci_H]
#
#             if e_t == e_h and l == 0:
#                 acc = acc + 1
#                 pred_lb.append(0)
#
#             elif e_t == e[0] and e_h == e[1] and l == 1:
#                 acc = acc + 1
#                 pred_lb.append(1)
#
#             elif e_t == e[0] and e_h == e[1] and l == 0:
#                 acc = acc + 1
#                 pred_lb.append(0)
#
#             elif e_t == e_h and l == 1:
#                 pred_lb.append(0)
#
#             elif e_t != e_h and l == 0:
#                 pred_lb.append(1)
#
#             elif e_t != e_h and l == 1:
#                 pred_lb.append(0)
#
#         acc = float(acc / count)
#         f1 = f1_score(edge_label, pred_lb)
#         return acc, f1
#
#     def _decode(self, z):
#         z_vert = z[0]
#         z_edge_T = z[1]
#         z_edge_H = z[2]
#
#         inci_pred_T = self.gnn_inci_T(z_vert, z_edge_T)
#         inci_pred_H = self.gnn_inci_H(z_vert, z_edge_H)
#
#         vertex_pred = self.add_vertex(z_vert)
#         vertex_pred = F.softmax(vertex_pred, dim=-1)
#
#         return vertex_pred, inci_pred_T, inci_pred_H
#
#     def decode(self, z):
#         vertex_pred, inci_pred_T, inci_pred_H = self._decode(z)
#
#         inci_pred_T[inci_pred_T.max(dim=-1)[0] < 0.25] = 0.
#         inci_pred_H[inci_pred_H.max(dim=-1)[0] < 0.25] = 0.
#
#         inci_T = F.one_hot(torch.argmax(inci_pred_T, dim=-1, keepdim=False), num_classes=self.max_n_vertex) * -1
#         inci_H = F.one_hot(torch.argmax(inci_pred_H, dim=-1, keepdim=False), num_classes=self.max_n_vertex)
#         inci_pred = inci_T + inci_H
#
#         edges_pred = self._to_edges(inci_pred)
#
#         return edges_pred
#
#     def loss(self, mean, logvar, data_batch):
#         mean_vert = mean[0]
#         mean_edge_T = mean[1]
#         mean_edge_H = mean[2]
#
#         logvar_vert = logvar[0]
#         logvar_edge_T = logvar[1]
#         logvar_edge_H = logvar[2]
#
#         inci_pred_T, inci_pred_H, inci_lb_T, inci_lb_H, weight_T, weight_H, vertex_pred, lb = data_batch
#
#         inci_pred_T = self.sigmoid(inci_pred_T)
#         inci_pred_H = self.sigmoid(inci_pred_H)
#
#         edge_loss_T = F.binary_cross_entropy(inci_pred_T, inci_lb_T, reduction='mean', weight=weight_T)
#         edge_loss_H = F.binary_cross_entropy(inci_pred_H, inci_lb_H, reduction='mean', weight=weight_H)
#
#         vertex_loss = self.vertex_criterion(vertex_pred.view(-1, vertex_pred.size(-1)), lb.view(-1))
#
#         mean_cat = [mean_vert, mean_edge_T, mean_edge_H]
#         logvar_cat = [logvar_vert, logvar_edge_T, logvar_edge_H]
#         mean_cat = torch.cat(mean_cat, dim=-2)
#         logvar_cat = torch.cat(logvar_cat, dim=-2)
#
#         kl_divergence = -0.5 * (1 + logvar_cat - mean_cat ** 2 - torch.exp(logvar_cat)).mean()
#
#         if not self.node_classification:
#             loss = edge_loss_H + edge_loss_T + kl_divergence
#         else:
#             loss = edge_loss_H + edge_loss_T + kl_divergence + vertex_loss
#
#         return loss, vertex_loss, torch.zeros(1).to(self.get_device()), edge_loss_T, edge_loss_H, kl_divergence
#
#     def link_predictor(self, Z, g_batch):
#         vid, lb, inci_T, inci_H, w_T, w_H, inci_lb_T, inci_lb_H, edge_p, edge_n, feat, adj, g = g_batch
#         g_recon = self.decode(Z)
#
#         if not torch.is_tensor(inci_T):
#             edge_p = torch.cat(edge_p, dim=0)
#             edge_n = torch.cat(edge_n, dim=0)
#
#         pred = []
#         label = []
#         acc = 0
#
#         n = len(g_recon)
#         for (eg_p, eg_n, g) in zip(edge_p, edge_n, g_recon):
#             edge_list = g.get_edgelist()
#
#             eg_p = eg_p.tolist()
#             eg_n = eg_n.tolist()
#
#             eg_p = (eg_p[0], eg_p[1])
#             eg_n = (eg_n[0], eg_n[1])
#
#             # print(eg_p, eg_n, edge_list)
#
#             if eg_p in edge_list:
#                 acc = acc + 1
#                 pred.append(1)
#                 label.append(1)
#                 # print(1)
#             else:
#                 pred.append(0)
#                 label.append(1)
#                 # print(2)
#             if eg_n not in edge_list:
#                 acc = acc + 1
#                 pred.append(0)
#                 label.append(0)
#                 # print(3)
#             else:
#                 pred.append(1)
#                 label.append(0)
#                 # print(4)
#         acc = float(acc / (n * 2))
#         f1 = f1_score(label, pred)
#
#         return acc, f1
#
#     def reparameterize(self, mu, logvar, eps_scale=0.01):
#         # return z ~ N(mu, std)
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = torch.randn_like(std) * eps_scale
#             return eps.mul(std).add_(mu)
#         else:
#             return mu
#
#     def encode_decode(self, G, adj, feature):
#         mean, logvar, sampled_z = self.encode(G, adj, feature)
#         # z = self.reparameterize(mu, logvar)
#         return self.decode(sampled_z)
#
#     def forward(self, G):
#         mean, logvar, sampled_z = self.encode(G)
#         loss, ver_loss, adj_loss, inci_loss, inci_T_loss, kld = self.loss(mean, logvar, sampled_z, G)
#         return loss
#
#     def generate_sample(self, n):
#         z_vertex = torch.randn(n, self.max_n_vertex, self.nz).to(self.get_device())
#         z_edge_T = torch.randn(n, self.max_n_edge, self.nz).to(self.get_device())
#         z_edge_H = torch.randn(n, self.max_n_edge, self.nz).to(self.get_device())
#         sample = (z_vertex, z_edge_T, z_edge_H)
#         G = self.decode(sample)
#
#         return G



class VDGAE(nn.Module):
    def __init__(self, max_n, max_edge_n, n_vertex_type, hidden_dim=501, z_dim=56, feature_dimension=1703, with_neighbor=False, with_vertex=False, node_classification=False):
        super(VDGAE, self).__init__()
        self.max_n_vertex = max_n  # maximum number of vertices
        self.max_n_edge = max_edge_n  # max number of edges
        self.n_vertex_type = n_vertex_type  # number of vertex types
        self.hs = hidden_dim  # hidden state size of each vertex
        self.nz = z_dim  # size of latent representation z
        self.gs = hidden_dim  # size of graph state
        self.feature_dimension = feature_dimension
        self.with_neighbor = with_neighbor
        self.with_vertex = with_vertex
        self.node_classification = node_classification
        self.device = None

        # 0. encoding part, cora: 1434, citeseer: 3704, Cornell: 1704, WikiCS: 300
        self.FEATURE_NODE = feature_dimension

        if not with_vertex:
            self.FEATURE = max_n
        else:
            self.FEATURE = feature_dimension
            self.gnn_edge_T = GetEdge_Cora(input_dim=self.FEATURE, output_dim=self.FEATURE)
            self.gnn_edge_H = GetEdge_Cora(input_dim=self.FEATURE, output_dim=self.FEATURE)

        # 0.1 vertex encoder
        self.mu_vert = nn.Sequential(
            nn.Linear(in_features=self.FEATURE_NODE, out_features=z_dim + hidden_dim * 2),
            nn.ReLU(),
        )

        self.logvar_vert = nn.Sequential(
            nn.Linear(in_features=self.FEATURE_NODE, out_features=z_dim + hidden_dim * 2),
            nn.ReLU()
        )

        # 0.2 edge encoder
        if not with_neighbor:
            self.mu_edge_T = nn.Sequential(
                nn.Linear(in_features=self.FEATURE, out_features=z_dim + hidden_dim * 2),
                nn.ReLU(),
            )

            self.logvar_edge_T = nn.Sequential(
                nn.Linear(in_features=self.FEATURE, out_features=z_dim + hidden_dim * 2),
                nn.ReLU()
            )

            self.mu_edge_H = nn.Sequential(
                nn.Linear(in_features=self.FEATURE, out_features=z_dim + hidden_dim * 2),
                nn.ReLU(),
            )

            self.logvar_edge_H = nn.Sequential(
                nn.Linear(in_features=self.FEATURE, out_features=z_dim + hidden_dim * 2),
                nn.ReLU()
            )
        else:
            self.mu_edge_T = EncoderEdge_Cora(input_dim=self.FEATURE, hidden_dim=hidden_dim, out_dim=z_dim, device=self.get_device())
            self.logvar_edge_T = EncoderEdge_Cora(input_dim=self.FEATURE, hidden_dim=hidden_dim, out_dim=z_dim, device=self.get_device())
            self.mu_edge_H = EncoderEdge_Cora(input_dim=self.FEATURE, hidden_dim=hidden_dim, out_dim=z_dim, device=self.get_device())
            self.logvar_edge_H = EncoderEdge_Cora(input_dim=self.FEATURE, hidden_dim=hidden_dim, out_dim=z_dim, device=self.get_device())


        # 1. decoding part
        # 1.1 edge decoder
        self.gnn_inci_T = Decode_Cora(in_features=hidden_dim * 2 + z_dim, n_hidden=hidden_dim, out_features=hidden_dim * 2, n_heads=8, dropout=0.1)
        self.gnn_inci_H = Decode_Cora(in_features=hidden_dim * 2 + z_dim, n_hidden=hidden_dim, out_features=hidden_dim * 2, n_heads=8, dropout=0.1)

        # 1.2 vertex decoder
        self.add_vertex = nn.Sequential(
            nn.Linear(self.hs * 2 + z_dim, self.hs * 2),
            nn.ReLU(),
            nn.Linear(self.hs * 2, self.n_vertex_type),
        )

        # 2. others
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=-1)
        self.vertex_criterion = nn.CrossEntropyLoss()
        self.adj_criterion = nn.BCELoss()
        self.matrix_criterion = nn.MSELoss(reduction='sum')

        self.layerNorm_vert = nn.LayerNorm([self.max_n_vertex, self.FEATURE_NODE])
        self.layerNorm_edge_T = nn.LayerNorm([self.max_n_edge, self.FEATURE])
        self.layerNorm_edge_H = nn.LayerNorm([self.max_n_edge, self.FEATURE])
        self.layerNorm_vert_de = nn.LayerNorm([self.max_n_vertex, hidden_dim * 2 + z_dim])
        self.layerNorm_edge_T_de = nn.LayerNorm([self.max_n_edge, hidden_dim * 2 + z_dim])
        self.layerNorm_edge_H_de = nn.LayerNorm([self.max_n_edge, hidden_dim * 2 + z_dim])

        self.layerNorm_mean_vert = nn.LayerNorm([max_n, z_dim + hidden_dim * 2])
        self.layerNorm_mean_edge_T = nn.LayerNorm([max_edge_n, z_dim + hidden_dim * 2])
        self.layerNorm_mean_edge_H = nn.LayerNorm([max_edge_n, z_dim + hidden_dim * 2])
        self.layerNorm_logvar_vert = nn.LayerNorm([max_n, z_dim + hidden_dim * 2])
        self.layerNorm_logvar_edge_T = nn.LayerNorm([max_edge_n, z_dim + hidden_dim * 2])
        self.layerNorm_logvar_edge_H = nn.LayerNorm([max_edge_n, z_dim + hidden_dim * 2])


    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    def encode(self, arch):
        v, lb, ft, inci_T, inci_H, weight_T, weight_H, gsub_batch, edges = arch

        if torch.is_tensor(inci_T):
            inci_mat_T = inci_T.to(self.get_device())
            inci_mat_H = inci_H.to(self.get_device())
            ft = ft.to(self.get_device())
            edges = edges.to(self.get_device())

        if not torch.is_tensor(inci_T):
            inci_mat_T = torch.cat(inci_T, dim=0)
            inci_mat_H = torch.cat(inci_H, dim=0)
            ft = torch.cat(ft, dim=0)
            edges = torch.cat(edges, dim=0)

        if not self.with_vertex:
            batch_size = edges.size()
            edges_T = edges[:, : ,0].view(-1)
            edges_H = edges[:, :, 1].view(-1)

            edges_attn_T = F.one_hot(edges_T, inci_mat_T.size(-1)).float().view(batch_size[0], batch_size[1], -1).to(self.get_device())
            edges_attn_H = F.one_hot(edges_H, inci_mat_T.size(-1)).float().view(batch_size[0], batch_size[1], -1).to(self.get_device())
        else:
            edges_attn_T = self.gnn_edge_T(ft, inci_mat_T)
            edges_attn_H = self.gnn_edge_H(ft, inci_mat_H)

        # normalization
        ft = self.layerNorm_vert(ft)

        edges_attn_T = self.layerNorm_edge_T(edges_attn_T)
        edges_attn_H = self.layerNorm_edge_H(edges_attn_H)

        # vert
        mean_vert = self.mu_vert(ft)
        logvar_vert = self.logvar_vert(ft)

        # edge
        if not self.with_neighbor:
            mean_edge_T = self.mu_edge_T(edges_attn_T)#, inci_mat_T.permute(0, 2, 1))
            logvar_edge_T = self.logvar_edge_T(edges_attn_T)#, inci_mat_T.permute(0, 2, 1))

            mean_edge_H = self.mu_edge_H(edges_attn_H)#, inci_mat_H.permute(0, 2, 1))
            logvar_edge_H = self.logvar_edge_H(edges_attn_H)#, inci_mat_H.permute(0, 2, 1))
        else:
            mean_edge_T = self.mu_edge_T(edges_attn_T, inci_mat_T.permute(0, 2, 1))
            logvar_edge_T = self.logvar_edge_T(edges_attn_T, inci_mat_T.permute(0, 2, 1))

            mean_edge_H = self.mu_edge_H(edges_attn_H, inci_mat_H.permute(0, 2, 1))
            logvar_edge_H = self.logvar_edge_H(edges_attn_H, inci_mat_H.permute(0, 2, 1))

        mean_vert = self.layerNorm_mean_vert(mean_vert)
        mean_edge_T = self.layerNorm_mean_edge_T(mean_edge_T)
        mean_edge_H = self.layerNorm_mean_edge_H(mean_edge_H)
        logvar_vert = self.layerNorm_logvar_vert(logvar_vert)
        logvar_edge_T = self.layerNorm_logvar_edge_T(logvar_edge_T)
        logvar_edge_H = self.layerNorm_logvar_edge_H(logvar_edge_H)

        z_vertex = self._GaussianNoise(mean_vert.size()) * torch.exp(0.5 * logvar_vert) + mean_vert
        z_edge_T = self._GaussianNoise(mean_edge_T.size()) * torch.exp(0.5 * logvar_edge_T) + mean_edge_T
        z_edge_H = self._GaussianNoise(mean_edge_H.size()) * torch.exp(0.5 * logvar_edge_H) + mean_edge_H

        return (mean_vert, mean_edge_T, mean_edge_H), (logvar_vert, logvar_edge_T, logvar_edge_H), (z_vertex, z_edge_T, z_edge_H)

    def _GaussianNoise(self, size):
        gaussian_noise = torch.rand(size).to(self.get_device())
        return gaussian_noise

    def _reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def calculate_accuracy(self, z):
        vertex_pred, inci_pred_T, inci_pred_H = self._decode(z)

        inci_pred_T[inci_pred_T.max(dim=-1)[0] < 0.25] = 0.
        inci_pred_H[inci_pred_H.max(dim=-1)[0] < 0.25] = 0.

        inci_T = F.one_hot(torch.argmax(inci_pred_T, dim=-1, keepdim=False), num_classes=self.max_n_vertex) * -1
        inci_H = F.one_hot(torch.argmax(inci_pred_H, dim=-1, keepdim=False), num_classes=self.max_n_vertex)
        inci_pred = inci_T + inci_H

        v_type_pred = torch.max(vertex_pred, dim=-1)[1].view(z[0].size(0), -1)

        return v_type_pred, inci_pred

    def direction_prediction(self, z, vertices, edges, edge_label):
        v_feat = z[0]
        e_feat_T = z[1]
        e_feat_H = z[2]

        acc = 0
        count = 0
        pred_lb = []

        for vid, e, l, vf, et, eh in zip(vertices, edges, edge_label, v_feat, e_feat_T, e_feat_H):

            e = [e[0], e[1]]

            e_feat_tail = et.unsqueeze(dim=0)
            e_feat_head = eh.unsqueeze(dim=0)
            v_f = vf.unsqueeze(dim=0)

            attn_tail = self.gnn_inci_T(v_f, e_feat_tail).squeeze()
            attn_head = self.gnn_inci_H(v_f, e_feat_head).squeeze()

            # attn_tail = self.softmax(attn_tail)
            # attn_head = self.softmax(attn_head)

            attn_tail[attn_tail.max(dim=-1)[0] < 0.25] = 0.
            attn_head[attn_head.max(dim=-1)[0] < 0.25] = 0.

            inci_T = torch.argmax(attn_tail, dim=-1, keepdim=False)
            inci_H = torch.argmax(attn_head, dim=-1, keepdim=False)

            edge_list_pred = torch.cat([vid.index_select(dim=0, index=inci_T).unsqueeze(dim=0), vid.index_select(dim=0, index=inci_H).unsqueeze(dim=0)], dim=0).t()

            edge_list_pred = edge_list_pred.tolist()

            if e not in edge_list_pred and l == 0:
                acc = acc + 1
                pred_lb.append(0)

            elif e in edge_list_pred and l == 1:
                acc = acc + 1
                pred_lb.append(1)

            elif e in edge_list_pred and l == 0:
                pred_lb.append(1)

            elif e not in edge_list_pred and l == 1:
                pred_lb.append(0)

            # print(e, l, edge_list_pred)

        # print(acc)

        acc = float(acc / len(edges))
        f1 = f1_score(edge_label, pred_lb)
        # print(edge_label, pred_lb)
        # print(acc, f1)
        # exit()
        return acc, f1

    def _decode(self, z):
        z_vert = z[0]
        z_edge_T = z[1]
        z_edge_H = z[2]

        inci_pred_T = self.gnn_inci_T(z_vert, z_edge_T)
        inci_pred_H = self.gnn_inci_H(z_vert, z_edge_H)

        vertex_pred = self.add_vertex(z_vert)
        vertex_pred = F.softmax(vertex_pred, dim=-1)

        return vertex_pred, inci_pred_T, inci_pred_H

    def decode(self, z):
        vertex_pred, inci_pred_T, inci_pred_H = self._decode(z)

        inci_pred_T[inci_pred_T.max(dim=-1)[0] < 0.25] = 0.
        inci_pred_H[inci_pred_H.max(dim=-1)[0] < 0.25] = 0.

        inci_T = F.one_hot(torch.argmax(inci_pred_T, dim=-1, keepdim=False), num_classes=self.max_n_vertex) * -1
        inci_H = F.one_hot(torch.argmax(inci_pred_H, dim=-1, keepdim=False), num_classes=self.max_n_vertex)
        inci_pred = inci_T + inci_H

        edges_pred = self._to_edges(inci_pred)

        return edges_pred

    def loss(self, mean, logvar, z, data_batch):
        v, lb, ft, inci_T, inci_H, weight_T, weight_H, gsub_batch, _ = data_batch

        if torch.is_tensor(inci_T):
            inci_mat_T = inci_T.to(self.get_device())
            inci_mat_H = inci_H.to(self.get_device())
            weight_T = weight_T.to(self.get_device())
            weight_H = weight_H.to(self.get_device())
            ft = ft.to(self.get_device())
            lb = lb.to(self.get_device())

        if not torch.is_tensor(inci_T):
            inci_mat_T = torch.cat(inci_T, dim=0)
            inci_mat_H = torch.cat(inci_H, dim=0)
            weight_T = torch.cat(weight_T, dim=0)
            weight_H = torch.cat(weight_H, dim=0)
            ft = torch.cat(ft, dim=0)
            lb = torch.cat(lb, dim=0)

        mean_vert = mean[0]
        mean_edge_T = mean[1]
        mean_edge_H = mean[2]

        logvar_vert = logvar[0]
        logvar_edge_T = logvar[1]
        logvar_edge_H = logvar[2]

        vertex_pred, inci_pred_T, inci_pred_H = self._decode(z)

        inci_pred_T = self.sigmoid(inci_pred_T)
        inci_pred_H = self.sigmoid(inci_pred_H)

        edge_loss_T = F.binary_cross_entropy(inci_pred_T.view(-1), inci_mat_T.view(-1), reduction='mean', weight=weight_T)
        edge_loss_H = F.binary_cross_entropy(inci_pred_H.view(-1), inci_mat_H.view(-1), reduction='mean', weight=weight_H)
        vertex_loss = self.vertex_criterion(vertex_pred.view(-1, vertex_pred.size(-1)), lb.view(-1))

        mean_cat = [mean_vert, mean_edge_T, mean_edge_H]
        logvar_cat = [logvar_vert, logvar_edge_T, logvar_edge_H]
        mean_cat = torch.cat(mean_cat, dim=-2)
        logvar_cat = torch.cat(logvar_cat, dim=-2)

        kl_divergence = -0.5 * (1 + logvar_cat - mean_cat ** 2 - torch.exp(logvar_cat)).mean()

        if not self.node_classification:
            loss = edge_loss_H + edge_loss_T + kl_divergence
        else:
            loss = edge_loss_H + edge_loss_T + kl_divergence + vertex_loss

        return loss, vertex_loss, torch.zeros(1).to(self.get_device()), edge_loss_T, edge_loss_H, kl_divergence

    def link_predictor(self, Z, g_batch):
        vid, lb, inci_T, inci_H, w_T, w_H, inci_lb_T, inci_lb_H, edge_p, edge_n, feat, adj, g = g_batch
        g_recon = self.decode(Z)

        if not torch.is_tensor(inci_T):
            edge_p = torch.cat(edge_p, dim=0)
            edge_n = torch.cat(edge_n, dim=0)

        pred = []
        label = []
        acc = 0

        n = len(g_recon)
        for (eg_p, eg_n, g) in zip(edge_p, edge_n, g_recon):
            edge_list = g.get_edgelist()

            eg_p = eg_p.tolist()
            eg_n = eg_n.tolist()

            eg_p = (eg_p[0], eg_p[1])
            eg_n = (eg_n[0], eg_n[1])

            # print(eg_p, eg_n, edge_list)

            if eg_p in edge_list:
                acc = acc + 1
                pred.append(1)
                label.append(1)
                # print(1)
            else:
                pred.append(0)
                label.append(1)
                # print(2)
            if eg_n not in edge_list:
                acc = acc + 1
                pred.append(0)
                label.append(0)
                # print(3)
            else:
                pred.append(1)
                label.append(0)
                # print(4)
        acc = float(acc / (n * 2))
        f1 = f1_score(label, pred)

        return acc, f1

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode_decode(self, G, adj, feature):
        mean, logvar, sampled_z = self.encode(G, adj, feature)
        # z = self.reparameterize(mu, logvar)
        return self.decode(sampled_z)

    def forward(self, G):
        mean, logvar, sampled_z = self.encode(G)
        loss, ver_loss, adj_loss, inci_loss, inci_T_loss, kld = self.loss(mean, logvar, sampled_z, G)
        return loss

    def generate_sample(self, n):
        z_vertex = torch.randn(n, self.max_n_vertex, self.nz).to(self.get_device())
        z_edge_T = torch.randn(n, self.max_n_edge, self.nz).to(self.get_device())
        z_edge_H = torch.randn(n, self.max_n_edge, self.nz).to(self.get_device())
        sample = (z_vertex, z_edge_T, z_edge_H)
        G = self.decode(sample)

        return G