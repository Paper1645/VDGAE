import random

import igraph
import numpy
import torch
import torch.nn.functional as F
import numpy as np
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
# from util import *

# G = nx.karate_club_graph()
# # G = G.to_undirected()
# adj_mat = nx.adjacency_matrix(G).todense()
#
# print(adj_mat)
#
# sub_adj = adj_mat[1:6, 1:6]

# G = igraph.Graph.Lattice([10, 10], circular=False)
#
# print(type(G.get_adjacency().data))

# sub_G = nx.Graph(sub_adj)


# res = [0, 1, 2, 3, 4, 5, 'parrot']
# pos = nx.spring_layout(G)
# k = G.subgraph(res)
#
# # nx.draw_networkx(k, pos=pos)
# # othersubgraph = G.subgraph(range(6, G.order()))
# nx.draw_networkx(G, pos=pos, node_color='b')
# nx.draw_networkx(G.subgraph([30, 8, 2, 0, 21, 11, 25, 23, 14]), pos=pos)
#






# some code to test format transformations
# row = '[[4], [0, 1], [3, 1, 0], [3, 0, 1, 1], [1, 1, 1, 1, 1], [2, 1, 1, 0, 1, 1], [5, 1, 1, 1, 1, 1, 0], [2, 0, 0, 1, 0, 0, 1, 0]]'
# row = '[[2], [2, 0], [4, 0, 0], [0, 1, 0, 0], [2, 1, 0, 0, 1], [3, 1, 0, 0, 0, 0], [5, 0, 0, 0, 0, 1, 0], [4, 0, 0, 0, 0, 0, 0, 0], [4, 1, 0, 0, 1, 0, 0, 0, 0], [3, 0, 1, 1, 0, 0, 1, 0, 0, 0], [5, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]'
# g, _ = decode_ENAS_to_igraph(row)
#
# g.add_vertex(0)
# g.add_vertex(1)
# g.add_vertex(2)
# g.add_vertex(3)
# g.add_vertex(4)

# g = igraph.Graph(directed=True)
#
# g.add_vertices(5)
#
# g.vs[0]['type'] = 1
# g.vs[1]['type'] = 3
# g.vs[2]['type'] = 4
# g.vs[3]['type'] = 5
# g.vs[4]['type'] = 2

# g.add_edge(0, 1)
# g.add_edge(0, 3)
# g.add_edge(1, 3)
# g.add_edge(1, 2)
# g.add_edge(2, 4)
# g.add_edge(3, 4)
# g.add_edge(0, 3)
# edges = [(0, 1), (0, 3)]
# g.add_edges(edges)
#
# print(g)
# exit(0)
#
# G_valid = []
#
# for i in range(g.vcount()):
#     print(g.vs[i]['type'])
#     G_valid.append(g.vs[i]['type'])
#
# exit(0)
# for i in range(4):
#     G_valid.append(g)
#
# adj = np.array(g.get_adjacency().data)
# lt = []
# for i in range(adj.shape[0]-1):
#     for j in range(i+1, adj.shape[1]):
#         lt.append(adj[i][j])
# print(lt)
# print(adj)
# exit(0)
# G_valid_str = [decode_igraph_to_ENAS(g) for g in G_valid]
# #
# print(G_valid_str)
# print(set(G_valid_str))
# exit(0)
# string = decode_igraph_to_ENAS(g)
# print(row, string)
# exit(0)
# pdb.set_trace()
# pwd = os.getcwd()
# os.chdir('software/enas/')
# os.system('./scripts/custom_cifar10_macro_final.sh ' + '"' + string + '"')
# os.chdir(pwd)



#
# a = torch.tensor([[-1, 1, 0, 0, 0, 0],
#                   [1, 0, -1, 1, 0, 0],
#                   [0, 0, 1, 0, 1, -1],
#                   [0, 0, 0, 0, -1, 0],
#                   [0, -1, 0, -1, 0, 1]])
#
# b = torch.eye()
# import torch
#
# def _one_hot(idx, length):
#     if type(idx) in [list, range]:
#         if idx == []:
#             return None
#         idx = torch.LongTensor(idx).unsqueeze(0).t()
#         x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
#     else:
#         idx = torch.LongTensor([idx]).unsqueeze(0)
#         x = torch.zeros((1, length)).scatter_(1, idx, 1)
#     return x
#
# print(torch.nn.functional.one_hot(torch.tensor([4, 1, 2, 1, 3])))
# print(_one_hot([3, 1, 2, 1, 3], 5))
# a = torch.nn.functional.pad(a, [0, 6])
# print(a)

# a = torch.tensor([
#          [ 1,  1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1],
#          [-1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1, 1,  1,  1],
#          [-1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1, 1,  1,  1],
#          [ 1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1],
#          [ 1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
#          [ 1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1, -1],
#          [-1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1, 1,  1,  1],
#          [ 1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1]])

# a = torch.tensor([[ 0.0538, -0.4499, -0.2259,  0.1101,  0.4762, -0.5021,  0.4589, -0.2307,
#           0.5532,  0.4053, -0.0862,  0.2431,  0.2109,  0.5186, -0.1762, -0.5644,
#          -0.7007, -0.3207,  0.6699, -0.1785],
#         [-0.6947, -0.4953, -0.5423, -0.6446,  0.0515,  0.0013, -0.2441, -0.6561,
#          -0.0734,  0.5963,  0.3310, -0.4932, -0.6355, -0.1294,  0.4389,  0.2046,
#          -0.1608,  0.4004,  0.2953,  0.3740],
#         [-0.5302, -0.6172, -0.3657, -0.5290,  0.3210, -0.3747, -0.0441, -0.4836,
#           0.3425,  0.2839, -0.0298, -0.0738, -0.4592,  0.2552,  0.0472,  0.1318,
#          -0.4858, -0.0614,  0.6035, -0.0195],
#         [ 0.0441,  0.2427, -0.0388,  0.1847, -0.3229,  0.0923,  0.1941, -0.1167,
#           0.1048, -0.2891,  0.1408, -0.2340,  0.1204, -0.0278, -0.2477,  0.3742,
#           0.1409,  0.2089, -0.0388,  0.0855],
#         [ 0.0659, -0.3924,  0.2825,  0.0774,  0.6767, -0.5129,  0.0227, -0.2043,
#           0.3993,  0.0773,  0.2482,  0.4420, -0.1621,  0.4362, -0.5267, -0.1857,
#          -0.4198, -0.5553,  0.2580, -0.6186],
#         [ 0.1532,  0.4488,  0.0877,  0.1151, -0.6310,  0.6318,  0.0168,  0.2006,
#          -0.5935, -0.0306, -0.1911, -0.4126,  0.2776, -0.4724,  0.5150,  0.2071,
#           0.4827,  0.5019, -0.5707,  0.4624],
#         [-0.2269, -0.1197,  0.3724, -0.3650, -0.0129, -0.1288, -0.3774, -0.2012,
#          -0.2651, -0.2815,  0.0428, -0.0380, -0.3828, -0.1507, -0.1414,  0.6118,
#           0.2266, -0.0157, -0.1653, -0.1782],
#         [-0.3113, -0.2088, -0.0636, -0.1767,  0.0224, -0.0832,  0.0976, -0.6416,
#          -0.1247,  0.4558,  0.5022, -0.3842, -0.3492,  0.0087,  0.0277,  0.3112,
#          -0.0733,  0.2401, -0.0295,  0.1033]])



# print(torch.mean(torch.sum(a.t(), dim=1)))
# print(torch.sum(a.t(), dim=1))
#
# a = torch.tensor([[1.638, 0.638, 0.362],
#                   [0.756, 0.244, 1.244],
#                   [0.344, 0.656, 1.656]])
#
# b = torch.tensor([-1, 1, 0, -1])
#
# a = a * -1
#
# print(torch.softmax(a, dim=1))
# print(torch.abs(b + 1))

# a = torch.tensor([1, 3, 2, 5])
#
# b = torch.tensor([[1, 2, 3, 4],
#                   [2, 3, 4, 1],
#                   [2, 2, 1, 3],
#                   [3, 2, 2, 1]])
#
#
# for i, item in enumerate(a):
#     if item == 2:
#         a = a[:i+1]
#         b = b[:i+1, :i+1]
#
#
#
# print(a)
# print(b)
# print(torch.triu(b, diagonal=1))

# a = np.array([[0, 1, 0, 0, 0],
#               [0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 1],
#               [0, 0, 1, 0, 0],
#               [1, 1, 0, 0, 0]])
# #
# # # a = a + np.transpose(a) * -1
# g = nx.DiGraph(a)
#
# L = nx.line_graph(g)
# adj_L = nx.adjacency_matrix(L).todense()
# print(adj_L)



# g = nx.DiGraph()
# g.add_nodes_from([0, 1, 2, 3])
#
# g.add_edge(0, 3)
# g.add_edge(0, 3)
# print(nx.adjacency_matrix(g).todense())
# inci = nx.incidence_matrix(g, oriented=True).todense()
# print(inci)
#
# print(np.matmul(np.transpose(inci), inci))
#
# print(nx.edges(g))
# print(nx.nodes(g))
#
# adj_L = nx.adjacency_matrix(L).todense()
# print(adj_L)
# print(adj_L + np.transpose(adj_L))
#
# inci_L = nx.incidence_matrix(L, oriented=True).todense()
#
# print(inci_L)
# print(np.matmul(np.transpose(inci_L), inci_L))
# print(nx.nodes(L))
# print(nx.edges(L))
#
# nx.draw(g, pos=nx.circular_layout(g), with_labels=True)
# plt.show()
# nx.draw(L, pos=nx.circular_layout(L), with_labels=True)
# plt.show()

# g = igraph.Graph(directed=True)
# g.add_vertices(4)
#
# edges = [[0, 3], [1, 3], [0, 3]]
# print(edges)
# edges = np.unique(edges, axis=0)
# print(edges)
#
# g.add_edges(edges)
# g.delete_vertices(3)
#
# layout = g.layout('kk')
# igraph.plot(g)

# a = torch.rand(1, 1, 51)
# b = torch.tensor([[[-4., -1., 2.],
#                    [-4., -1., 2.]],
#                   [[-4., -1., 2.],
#                    [-4., -1., 2.]]])
#
# a = F.normalize(a, dim=2, p=2)
# print(a)
# # b = F.normalize(b, dim=1, p=2)

# print(a_n)
# print(b_n)

# ab = torch.cross(a, b, dim=1)
# ba = torch.cross(b, a, dim=1)

# ab = F.normalize(ab, dim=1, p=1)
# ba = F.normalize(ba, dim=1, p=1)

# ab = torch.einsum('i,j->ij', [a, b])
# ba = torch.einsum('i,j->ij', [b, a])

# H = torch.ones(51, 51)
# H = torch.triu(H, diagonal=1)
# H = H + H.t() * -1
# H = H.unsqueeze(dim=0)
# print(H)
#
# aHb = torch.bmm(a, H)
#
# print(aHb)
# aHb = torch.bmm(aHb, a.permute(0, 2, 1))
#
# print(aHb)
# print(torch.bmm(a, a.permute(0, 2, 1)))

# bHa = torch.mm(b, H)
# bHa = torch.mm(bHa, a.t())
#
# print(aHb)
# print(bHa)

# a = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 0, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 0, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 2, 1, 1, 1, 2, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#
# b = torch.arange(a.size(0))
#
# mask = a != 1
#
# print(a[mask][:15])
# print(b[mask][:15])
# # print(b[a!=1])
#
# # print(round(a.size(0) / 2))
#

# a = torch.rand(10, 8)
# b = torch.rand(10, 8)
#
# print(torch.cosine_similarity(a, a, dim=1))
#
# a = torch.tensor([[-0.8, 0.2354, 0.2354, -0.1],
#                   [-0.8, 0.35, 0.3254, -0.1],
#                   [0.8, -0.15, 0.8354, -0.2]])
# b = torch.zeros_like(a)
#
# idx = torch.max(a, dim=1)[1]
# idx1 = torch.tensor([0, 1, 2])
#
# b[idx1, idx] = 1
# #
# # print(b)
# a = torch.tensor([
#         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
#         [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.],
#         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  1.,  0.,  0.,  0.,  0.,  0., -1.],
#         [ 0.,  0.,  0.,  0.,  0.,  1.,  0., -1.],
#         [-1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
#         [-1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
#         [ 0., -1.,  0.,  1.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  1.,  0.,  0.,  0.,  0.,  0., -1.],
#         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

#
# print(a[:, :6])
# for item in a:
#     t = torch.where(item[:6] == 1)[0][0]
#     h = torch.where(item[:6] == -1)[0][0]
#     print(t, h)

# adj = torch.tensor([[0, 1, 0, 0, 0],
#                     [0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 1],
#                     [0, 0, 1, 0, 0],
#                     [1, 1, 0, 0, 0]])
#
# adj = adj + adj.t()
#
# inci = torch.tensor([[-1, 1, 0, 0, 0, 0],
#                      [1, 0, -1, 1, 0, 0],
#                      [0, 0, 1, 0, 1, -1],
#                      [0, 0, 0, 0, -1, 0],
#                      [0, -1, 0, -1, 0, 1]])
#
# print(adj)
# print(inci)
# print(torch.mm(adj, inci))

# a = torch.tensor([3, 2, 3, 3, 0, 0, 0])
# print(torch.count_nonzero(a))
#
# print(a[torch.count_nonzero(a):])

# a = torch.tensor([[[2, 3, 2, 1],
#                    [1, 2, 3, 1],
#                    [3, 3, 1, 2]],
#                   [[1, 1, 2, 1],
#                    [1, 1, 3, 1],
#                    [3, 3, 1, 2]]])
#
# b = torch.repeat_interleave(a, 5)
# print(b)
#
# torch.randn()

# D = torch.tensor([[-1, 1, 0, 0, 0, 0, -1, 0],
#                   [1, 0, -1, 1, 0, 0, 0, 0],
#                   [0, 0, 1, 0, 1, -1, 0, -1],
#                   [0, 0, 0, 0, -1, 0, 0, 1],
#                   [0, -1, 0, -1, 0, 1, 1, 0]])
#
# D_in = torch.tensor([[-1, 0, 0, 0, 0, 0, -1, 0],
#                      [0, 0, -1, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 0, -1, 0, -1],
#                      [0, 0, 0, 0, -1, 0, 0, 0],
#                      [0, -1, 0, -1, 0, 0, 0, 0]])
#
# D_out = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0],
#                       [1, 0, 0, 1, 0, 0, 0, 0],
#                       [0, 0, 1, 0, 1, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0, 1],
#                       [0, 0, 0, 0, 0, 1, 1, 0]])
#
# A = torch.tensor([[0, 1, 0, 0, 1],
#                   [0, 0, 1, 0, 0],
#                   [0, 0, 0, 1, 1],
#                   [0, 0, 1, 0, 0],
#                   [1, 1, 0, 0, 0]])
#
#
# print(torch.mm(D, D.t()))
# print(torch.mm(D.t(), D))
#
#
# print(torch.mm(D_in, D.t()))
# print(torch.mm(D_out, D.t()))
#
# print(torch.diag((A + A.t()).sum(dim=0)) - (A+A.t()))
#
# print(D)

# import pickle
# import gzip
#
# def save_object(obj, filename):
#     result = pickle.dumps(obj)
#     with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
#     dest.close()
#
#
# a = list(range(10000))
#
# save_object(a, 'a.pk')
#
# def load_object(filename):
#     with gzip.GzipFile(filename, 'rb') as source: result = source.read()
#     ret = pickle.loads(result)
#     source.close()
#     return ret
#
# b = load_object('a.pk')
#
# print(b)

# idx = list(range(len(11701)))
# idx = torch.tensor(idx)
#
# for k in range(10):
#     for i in range(graph_args.num_vertex_type):
#         ids = idx[labels == i]
#
#         idx_tmp = list(range(ids.size(0)))
#         random.shuffle(idx_tmp)
#         idx_tmp = torch.tensor(idx_tmp)
#
#         ids = ids.index_select(dim=0, index=idx_tmp)
#
#         idx_train[k].append(ids[:20])
#         idx_test[k].append(ids[20:])
#
# for k in range(10):
#     idx_train[k] = torch.cat(idx_train[k], dim=0)
#     idx_tmp = list(range(idx_train[k].size(0)))
#     random.shuffle(idx_tmp)
#     idx_tmp = torch.tensor(idx_tmp)
#     idx_train[k] = idx_train[k].index_select(dim=0, index=idx_tmp).to(device)
#
#     idx_test[k] = torch.cat(idx_test[k], dim=0)
#     idx_tmp = list(range(idx_test[k].size(0)))
#     random.shuffle(idx_tmp)
#     idx_tmp = torch.tensor(idx_tmp)
#     idx_test[k] = idx_test[k].index_select(dim=0, index=idx_tmp).to(device)


B = torch.tensor([[-1, 1, 0, 0, 0],
                  [1, 0, 0, 0, -1],
                  [0, -1, 1, 0, 0],
                  [0, 1, 0, 0, -1],
                  [0, 0, 1, -1, 0],
                  [0, 0, -1, 0, 1],
                  [0, 0, -1, 1, 0]])


C = torch.tensor([[1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1],
                  [0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0]])

D = torch.tensor([[0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])



print(torch.mm(C, C.t()))

print(torch.mm(D, D.t()))

