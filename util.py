from __future__ import print_function
import torch
import random
import igraph
import argparse
import numpy as np
import scipy.sparse as sp
import pandas as pd
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS
import pickle
import gzip
from tqdm import tqdm

cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_module_state(model, state_name):
    pretrained_dict = torch.load(state_name)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return

def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret



def load_WikiCS_node(dataset_path="WikiCS"):
    print('Loading {} dataset...'.format(dataset_path))
    path = './data/{}/'.format(dataset_path)

    # load graph from file
    g_stem = load_object('./data/{}/graph.pk'.format(dataset_path))
    dataset = WikiCS(root=path)[0]

    features = dataset.x
    labels = dataset.y
    edges = dataset.edge_index

    MAX_NODES = labels.size(0)

    # edges_list = edges.t().tolist()

    # create graph and save to file
    # g_stem = igraph.Graph(directed=True)
    # g_stem.add_vertices(labels.size(0))
    # for i in range(g_stem.vcount()):
    #     g_stem.vs[i]['id'] = i
    #     g_stem.vs[i]['feature'] = features[i].tolist()
    #     g_stem.vs[i]['type'] = labels[i].tolist()
    #
    # pbar = tqdm(edges_list)
    #
    # for i, e in enumerate(pbar):
    #     if e[0] == e[1]:
    #         edges_list.remove(e)
    #     else:
    #         g_stem.add_edge(e[0], e[1])
    #
    # save_object(g_stem, './data/{}/graph.pk'.format(dataset_path))

    # # create oriented graph
    # g_stem = igraph.Graph(directed=True)
    # g_stem.add_vertices(labels.size(0))
    # for i in range(g_stem.vcount()):
    #     g_stem.vs[i]['id'] = i
    #     g_stem.vs[i]['feature'] = features[i].tolist()
    #     g_stem.vs[i]['type'] = labels[i].tolist()
    #
    # pbar = tqdm(edges_list)
    #
    # edges_double = []
    #
    # for i in range(len(edges_list) - 1):
    #     for j in range(i+1, len(edges_list)):
    #         if edges_list[i][0] == edges_list[j][1] and edges_list[i][1] == edges_list[j][0]:
    #             edges_double.append(edges_list[j])
    #
    # for e in edges_double:
    #     if e in edges_list:
    #         edges_list.remove(e)
    #
    # pbar = tqdm(edges_list)
    #
    # for i, e in enumerate(pbar):
    #     if e[0] == e[1]:
    #         edges_list.remove(e)
    #     else:
    #         g_stem.add_edge(e[0], e[1])
    #
    # save_object(g_stem, './data/{}/graph_oriented.pk'.format(dataset_path))
    # exit()

    max_size = 10
    ####################################################
    # some inside funcs                                #
    ####################################################
    def get_vert_attrs(g):
        vid = []
        lb = []
        ft = []
        for i in range(g.vcount()):
            vid.append(g.vs[i]['id'])
            lb.append(g.vs[i]['type'])
            ft.append(g.vs[i]['feature'])
        adj = g.get_adjacency().data

        return torch.tensor(vid), torch.tensor(lb).unsqueeze(dim=0), torch.tensor(ft).unsqueeze(dim=0), adj#torch.tensor(adj).unsqueeze(dim=0)

    def get_inci(g):
        vert_n = g.vcount()

        edge_n = vert_n * (vert_n - 1)
        e_list = g.get_edgelist()

        e_idx = []
        for i in range(vert_n):
            for j in range(vert_n):
                if i != j:
                    e_idx.append([i, j])

        inci_mat_T = torch.zeros(vert_n, edge_n)
        inci_mat_H = torch.zeros(vert_n, edge_n)

        inci_lb_T = torch.zeros(vert_n, edge_n)
        inci_lb_H = torch.zeros(vert_n, edge_n)

        edge_set = []
        e_list_select = []

        edges_double = []

        for i in range(len(e_list) - 1):
            for j in range(i + 1, len(e_list)):
                if e_list[i][0] == e_list[j][1] and e_list[i][1] == e_list[j][0]:
                    edges_double.append(e_list[j])

        for e in edges_double:
            if e in e_list:
                e_list.remove(e)

        # ###########################################
        # # add mask                                #
        # ###########################################
        # idx = list(range(len(e_list)))
        # random.shuffle(idx)
        # idx = idx[:int(len(e_list) * .15)]
        # e_list_select = [e_list[i] for i in idx]
        # ###########################################

        edge_test_set = []
        edge_test_label = []


        for i in range(vert_n):
            for j in range(vert_n):
                if i != j:
                    edge_set.append([i, j])

                if (i, j) in e_list:
                    ind = e_idx.index([i, j])
                    inci_lb_T[i, ind] = 1
                    inci_lb_H[j, ind] = 1

                    edge_test_set.append(ind)
                    edge_test_label.append(1)
                    edge_test_set.append(e_idx.index([j, i]))
                    edge_test_label.append(0)

                    if (i, j) not in e_list_select:
                        inci_mat_T[i, ind] = 1
                        inci_mat_H[j, ind] = 1

        inci_mat_T = inci_mat_T.t()
        inci_mat_H = inci_mat_H.t()
        inci_lb_T = inci_lb_T.t()
        inci_lb_H = inci_lb_H.t()

        if inci_mat_T.size(1) != 10:
            print(inci_mat_T.size())
            exit()

        pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
        pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

        weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
        weight_tensor_T = torch.ones(weight_mask_T.size(0))
        weight_tensor_T[weight_mask_T] = pos_weight_T

        weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
        weight_tensor_H = torch.ones(weight_mask_H.size(0))
        weight_tensor_H[weight_mask_H] = pos_weight_H

        return inci_mat_T.unsqueeze(dim=0), inci_mat_H.unsqueeze(dim=0), weight_tensor_T, weight_tensor_H, inci_lb_T.unsqueeze(dim=0), inci_lb_H.unsqueeze(dim=0), torch.tensor(edge_set).unsqueeze(dim=0), torch.tensor(edge_test_set).long(), torch.tensor(edge_test_label).long()

    def get_subgraph_verts(v):
        v_list = list(set(g_stem.neighbors(v)))

        v_list.append(v)

        burning = 0

        if len(v_list) == 0 or len(v_list) == 1:
            return []

        while len(v_list) > max_size or len(v_list) < max_size:
            burning = burning + 1
            if burning == 100:
                return []

            if len(v_list) > max_size:
                v_list = random.sample(v_list, max_size)
            else:
                v = random.sample(v_list, 1)[0]
                v_neig = list(set(g_stem.neighbors(v)))
                v_list = list(set(v_list + v_neig))

        return v_list

    train_data_set = []
    valid_data_set = []
    test_data_set = []

    print('Processing sub graphs ...')
    for itt in range(10):
        train_data = []
        test_data = []

        node_idx = list(range(g_stem.vcount()))
        random.shuffle(node_idx)
        random.shuffle(node_idx)
        NODE_TEST = int(g_stem.vcount() * .9)
        node_train = node_idx[:-NODE_TEST]
        node_test = node_idx[-NODE_TEST:]

        for i in node_train:
            sub_vert_list = get_subgraph_verts(i)

            if len(sub_vert_list) == 0 or len(sub_vert_list) == 1:
                continue

            g_sub = g_stem.subgraph(sub_vert_list)
            vid, label, feature, adj = get_vert_attrs(g_sub)
            inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label = get_inci(g_sub)

            train_data.append((vid, label, feature, adj, inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label, g_sub))


        for i in node_test:
            sub_vert_list = get_subgraph_verts(i)

            if len(sub_vert_list) == 0 or len(sub_vert_list) == 1:
                continue

            g_sub = g_stem.subgraph(sub_vert_list)
            vid, label, feature, adj = get_vert_attrs(g_sub)
            inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label = get_inci(g_sub)

            test_data.append((vid, label, feature, adj, inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label, g_sub))

        print(len(train_data))
        print(len(test_data))

        train_data_set.append(train_data)
        valid_data_set.append(test_data)
        test_data_set.append(test_data)


    NO_SUBGRAPHS = len(train_data)
    MAX_NODES = max_size
    MAX_EDGES = MAX_NODES * (MAX_NODES - 1)
    TEST = int(NO_SUBGRAPHS * .15)

    graph_args.num_vertex_type = 10  # original types + start/end types
    graph_args.max_n = MAX_NODES  # maximum number of nodes
    graph_args.max_n_eg = MAX_EDGES
    graph_args.test_eg = TEST
    graph_args.max_true_ng = MAX_EDGES
    graph_args.feature_dimension = features.size(-1)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))
    print('Test edges: %d' % (TEST))

    # return train_data[:-TEST], train_data[-TEST:], test_data, graph_args
    return train_data_set, valid_data_set, test_data_set, graph_args


def load_cora_node(dataset="cora"):
    print('Loading {} dataset...'.format(dataset))
    path = "./data/{}/".format(dataset)
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    features = normalize_features(features)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    g_stem = igraph.Graph(directed=True)
    g_stem.add_vertices(adj.shape[0])
    for i in range(g_stem.vcount()):
        g_stem.vs[i]['id'] = i
        g_stem.vs[i]['type'] = labels[i].tolist()
        g_stem.vs[i]['feature'] = features[i].tolist()

    for e in edges:
        if e[0] != e[1]:
            g_stem.add_edge(e[0], e[1])

    max_size = 3
    ####################################################
    # some inside funcs                                #
    ####################################################
    def get_vert_attrs(g):
        vid = []
        lb = []
        ft = []
        for i in range(g.vcount()):
            vid.append(g.vs[i]['id'])
            lb.append(g.vs[i]['type'])
            ft.append(g.vs[i]['feature'])
        adj = g.get_adjacency().data

        return torch.tensor(vid), torch.tensor(lb).unsqueeze(dim=0), torch.tensor(ft).unsqueeze(dim=0), adj#torch.tensor(adj).unsqueeze(dim=0)

    def get_inci(g):
        vert_n = g.vcount()

        edge_n = vert_n * (vert_n - 1)
        e_list = g.get_edgelist()

        e_idx = []
        for i in range(vert_n):
            for j in range(vert_n):
                if i != j:
                    e_idx.append([i, j])

        inci_mat_T = torch.zeros(vert_n, edge_n)
        inci_mat_H = torch.zeros(vert_n, edge_n)

        inci_lb_T = torch.zeros(vert_n, edge_n)
        inci_lb_H = torch.zeros(vert_n, edge_n)

        edge_set = []
        e_list_select = []

        # edges_double = []
        #
        # for i in range(len(e_list) - 1):
        #     for j in range(i + 1, len(e_list)):
        #         if e_list[i][0] == e_list[j][1] and e_list[i][1] == e_list[j][0]:
        #             edges_double.append(e_list[j])
        #
        # for e in edges_double:
        #     if e in e_list:
        #         e_list.remove(e)

        # ###########################################
        # # add mask                                #
        # ###########################################
        # idx = list(range(len(e_list)))
        # random.shuffle(idx)
        # idx = idx[:int(len(e_list) * .15)]
        # e_list_select = [e_list[i] for i in idx]
        # ###########################################

        edge_test_set = []
        edge_test_label = []

        for i in range(vert_n):
            for j in range(vert_n):
                if i != j:
                    edge_set.append([i, j])

                if (i, j) in e_list:
                    ind = e_idx.index([i, j])
                    inci_lb_T[i, ind] = 1
                    inci_lb_H[j, ind] = 1

                    edge_test_set.append(ind)
                    edge_test_label.append(1)
                    edge_test_set.append(e_idx.index([j, i]))
                    edge_test_label.append(0)

                    if (i, j) not in e_list_select:
                        inci_mat_T[i, ind] = 1
                        inci_mat_H[j, ind] = 1

        inci_mat_T = inci_mat_T.t()
        inci_mat_H = inci_mat_H.t()
        inci_lb_T = inci_lb_T.t()
        inci_lb_H = inci_lb_H.t()

        if inci_mat_T.size(1) != max_size:
            print(inci_mat_T.size())
            exit()

        pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
        pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

        weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
        weight_tensor_T = torch.ones(weight_mask_T.size(0))
        weight_tensor_T[weight_mask_T] = pos_weight_T

        weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
        weight_tensor_H = torch.ones(weight_mask_H.size(0))
        weight_tensor_H[weight_mask_H] = pos_weight_H

        return inci_mat_T.unsqueeze(dim=0), inci_mat_H.unsqueeze(dim=0), weight_tensor_T, weight_tensor_H, inci_lb_T.unsqueeze(dim=0), inci_lb_H.unsqueeze(dim=0), torch.tensor(edge_set).unsqueeze(dim=0), torch.tensor(edge_test_set).long(), torch.tensor(edge_test_label).long()

    def get_subgraph_verts(v, check_set):
        v_list = list(set(g_stem.neighbors(v)))

        v_list.append(v)

        burning = 0

        if len(v_list) == 0 or len(v_list) == 1:
            return []

        for item in v_list:
            if item in check_set:
                v_list.remove(item)

        while len(v_list) > max_size or len(v_list) < max_size:
            burning = burning + 1
            if burning == 100:
                return []

            if len(v_list) > max_size:
                v_list = random.sample(v_list, max_size)
            else:
                v = random.sample(v_list, 1)[0]
                v_neig = list(set(g_stem.neighbors(v)))
                v_list = list(set(v_list + v_neig))

        return v_list

    train_data_set = []
    valid_data_set = []
    test_data_set = []

    print('Processing sub graphs ...')
    for itt in range(10):
        train_data = []
        test_data = []

        node_idx = list(range(g_stem.vcount()))
        random.shuffle(node_idx)
        NODE_TEST = int(g_stem.vcount() * .3)
        node_train = node_idx[:-NODE_TEST]
        node_test = node_idx[-NODE_TEST:]

        for i in node_train:
            sub_vert_list = get_subgraph_verts(i, node_test)

            if len(sub_vert_list) == 0 or len(sub_vert_list) == 1:
                continue

            g_sub = g_stem.subgraph(sub_vert_list)
            vid, label, feature, adj = get_vert_attrs(g_sub)
            inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label = get_inci(g_sub)

            train_data.append((vid, label, feature, adj, inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label, g_sub))


        for i in node_test:
            sub_vert_list = get_subgraph_verts(i, node_train)

            if len(sub_vert_list) == 0 or len(sub_vert_list) == 1:
                continue

            g_sub = g_stem.subgraph(sub_vert_list)
            vid, label, feature, adj = get_vert_attrs(g_sub)
            inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label = get_inci(g_sub)

            test_data.append((vid, label, feature, adj, inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label, g_sub))

        train_data_set.append(train_data)
        valid_data_set.append(test_data)
        test_data_set.append(test_data)

    NO_SUBGRAPHS = len(train_data)
    MAX_NODES = 3
    MAX_EDGES = 6
    TEST = int(NO_SUBGRAPHS * .15)

    graph_args.num_vertex_type = 7  # original types + start/end types
    graph_args.max_n = MAX_NODES  # maximum number of nodes
    graph_args.max_n_eg = MAX_EDGES
    graph_args.test_eg = TEST
    graph_args.max_true_ng = MAX_EDGES
    graph_args.feature_dimension = features.size(-1)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))
    print('Test edges: %d' % (TEST))

    return train_data_set, valid_data_set, test_data_set, graph_args


def load_citeseer_node(dataset="citeseer"):
    print('Loading {} dataset...'.format(dataset))

    cs_content = pd.read_csv('./data/citeseer/citeseer.content', sep='\t', header=None)
    cs_cite = pd.read_csv('./data/citeseer/citeseer.cites', sep='\t', header=None)
    ct_idx = list(cs_content.index)
    paper_id = list(cs_content.iloc[:, 0])
    paper_id = [str(i) for i in paper_id]
    mp = dict(zip(paper_id, ct_idx))

    label = cs_content.iloc[:, -1]
    label = pd.get_dummies(label)

    feature = cs_content.iloc[:, 1:-1]

    mlen = cs_content.shape[0]
    adj = np.zeros((mlen, mlen))

    features = torch.FloatTensor(np.array(feature))
    labels = torch.LongTensor(np.where(label)[1])

    g_stem = igraph.Graph(directed=True)
    g_stem.add_vertices(adj.shape[0])
    for i in range(g_stem.vcount()):
        g_stem.vs[i]['id'] = i
        g_stem.vs[i]['type'] = labels[i].tolist()
        g_stem.vs[i]['feature'] = features[i].tolist()

    for i, j in zip(cs_cite[0], cs_cite[1]):
        if str(i) in mp.keys() and str(j) in mp.keys():
            x = mp[str(i)]
            y = mp[str(j)]
            if x != y:
                g_stem.add_edge(x, y)

    max_size = 3
    ####################################################
    # some inside funcs                                #
    ####################################################
    def get_vert_attrs(g):
        vid = []
        lb = []
        ft = []
        for i in range(g.vcount()):
            vid.append(g.vs[i]['id'])
            lb.append(g.vs[i]['type'])
            ft.append(g.vs[i]['feature'])
        adj = g.get_adjacency().data

        return torch.tensor(vid), torch.tensor(lb).unsqueeze(dim=0), torch.tensor(ft).unsqueeze(dim=0), adj#torch.tensor(adj).unsqueeze(dim=0)

    def get_inci(g):
        vert_n = g.vcount()

        edge_n = vert_n * (vert_n - 1)
        e_list = g.get_edgelist()

        e_idx = []
        for i in range(vert_n):
            for j in range(vert_n):
                if i != j:
                    e_idx.append([i, j])

        inci_mat_T = torch.zeros(vert_n, edge_n)
        inci_mat_H = torch.zeros(vert_n, edge_n)

        inci_lb_T = torch.zeros(vert_n, edge_n)
        inci_lb_H = torch.zeros(vert_n, edge_n)

        edge_set = []
        e_list_select = []

        edge_test_set = []
        edge_test_label = []

        for i in range(vert_n):
            for j in range(vert_n):
                if i != j:
                    edge_set.append([i, j])

                if (i, j) in e_list:
                    ind = e_idx.index([i, j])
                    inci_lb_T[i, ind] = 1
                    inci_lb_H[j, ind] = 1

                    edge_test_set.append(ind)
                    edge_test_label.append(1)
                    edge_test_set.append(e_idx.index([j, i]))
                    edge_test_label.append(0)

                    if (i, j) not in e_list_select:
                        inci_mat_T[i, ind] = 1
                        inci_mat_H[j, ind] = 1

        inci_mat_T = inci_mat_T.t()
        inci_mat_H = inci_mat_H.t()
        inci_lb_T = inci_lb_T.t()
        inci_lb_H = inci_lb_H.t()

        if inci_mat_T.size(1) != max_size:
            print(inci_mat_T.size())
            exit()

        pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
        pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

        weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
        weight_tensor_T = torch.ones(weight_mask_T.size(0))
        weight_tensor_T[weight_mask_T] = pos_weight_T

        weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
        weight_tensor_H = torch.ones(weight_mask_H.size(0))
        weight_tensor_H[weight_mask_H] = pos_weight_H

        return inci_mat_T.unsqueeze(dim=0), inci_mat_H.unsqueeze(dim=0), weight_tensor_T, weight_tensor_H, inci_lb_T.unsqueeze(dim=0), inci_lb_H.unsqueeze(dim=0), torch.tensor(edge_set).unsqueeze(dim=0), torch.tensor(edge_test_set).long(), torch.tensor(edge_test_label).long()

    def get_subgraph_verts(v, check_set):

        v_list = list(set(g_stem.neighbors(v)))

        v_list.append(v)

        burning = 0

        if len(v_list) == 0 or len(v_list) == 1:
            return []

        for item in v_list:
            if item in check_set:
                v_list.remove(item)

        while len(v_list) > max_size or len(v_list) < max_size:
            burning = burning + 1
            if burning == 100:
                return []

            if len(v_list) > max_size:
                v_list = random.sample(v_list, max_size)
            else:
                v = random.sample(v_list, 1)[0]
                v_neig = list(set(g_stem.neighbors(v)))
                v_list = list(set(v_list + v_neig))

        return v_list

    train_data_set = []
    valid_data_set = []
    test_data_set = []

    for itt in range(10):
        train_data = []
        test_data = []

        print('Processing sub graphs ...')
        node_idx = list(range(g_stem.vcount()))
        random.shuffle(node_idx)
        NODE_TEST = int(g_stem.vcount() * .3)
        node_train = node_idx[:-NODE_TEST]
        node_test = node_idx[-NODE_TEST:]

        for i in node_train:
            sub_vert_list = get_subgraph_verts(i, node_test)

            if len(sub_vert_list) == 0 or len(sub_vert_list) == 1:
                continue

            g_sub = g_stem.subgraph(sub_vert_list)
            vid, label, feature, adj = get_vert_attrs(g_sub)
            inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label = get_inci(g_sub)

            train_data.append((vid, label, feature, adj, inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label, g_sub))


        for i in node_test:
            sub_vert_list = get_subgraph_verts(i, node_train)

            if len(sub_vert_list) == 0 or len(sub_vert_list) == 1:
                continue

            g_sub = g_stem.subgraph(sub_vert_list)
            vid, label, feature, adj = get_vert_attrs(g_sub)
            inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label = get_inci(g_sub)

            test_data.append((vid, label, feature, adj, inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label, g_sub))

        train_data_set.append(train_data)
        valid_data_set.append(test_data)
        test_data_set.append(test_data)


    NO_SUBGRAPHS = len(train_data)
    MAX_NODES = 3
    MAX_EDGES = 6
    TEST = int(NO_SUBGRAPHS * .15)

    graph_args.num_vertex_type = 6  # original types + start/end types
    graph_args.max_n = MAX_NODES  # maximum number of nodes
    graph_args.max_n_eg = MAX_EDGES
    graph_args.test_eg = TEST
    graph_args.max_true_ng = MAX_EDGES
    graph_args.feature_dimension = features.size(-1)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))
    print('Test edges: %d' % (TEST))

    return train_data_set, valid_data_set, test_data_set, graph_args


def load_WebKB_node(dataset):
    print('Loading {} dataset...'.format(dataset))
    path = './data/{}/'.format(dataset)

    dataset = WebKB(root=path, name=dataset)[0]

    features = dataset.x
    labels = dataset.y
    edges = dataset.edge_index

    g_stem = igraph.Graph(directed=True)
    g_stem.add_vertices(labels.size(0))

    for i in range(g_stem.vcount()):
        g_stem.vs[i]['id'] = i
        g_stem.vs[i]['type'] = labels[i].tolist()
        g_stem.vs[i]['feature'] = features[i].tolist()

    for e in edges.t():
        if e[0] != e[1]:
            g_stem.add_edge(e[0], e[1])

    max_size = 3

    ####################################################
    # some inside funcs                                #
    ####################################################
    def get_vert_attrs(g):
        vid = []
        lb = []
        ft = []
        for i in range(g.vcount()):
            vid.append(g.vs[i]['id'])
            lb.append(g.vs[i]['type'])
            ft.append(g.vs[i]['feature'])
        adj = g.get_adjacency().data

        return torch.tensor(vid), torch.tensor(lb).unsqueeze(dim=0), torch.tensor(ft).unsqueeze(dim=0), adj  # torch.tensor(adj).unsqueeze(dim=0)

    def get_inci(g):
        vert_n = g.vcount()

        edge_n = vert_n * (vert_n - 1)
        e_list = g.get_edgelist()

        e_idx = []
        for i in range(vert_n):
            for j in range(vert_n):
                if i != j:
                    e_idx.append([i, j])

        inci_mat_T = torch.zeros(vert_n, edge_n)
        inci_mat_H = torch.zeros(vert_n, edge_n)

        inci_lb_T = torch.zeros(vert_n, edge_n)
        inci_lb_H = torch.zeros(vert_n, edge_n)

        edge_set = []
        e_list_select = []

        edge_test_set = []
        edge_test_label = []

        for i in range(vert_n):
            for j in range(vert_n):
                if i != j:
                    edge_set.append([i, j])

                if (i, j) in e_list:
                    ind = e_idx.index([i, j])
                    inci_lb_T[i, ind] = 1
                    inci_lb_H[j, ind] = 1

                    edge_test_set.append(ind)
                    edge_test_label.append(1)
                    edge_test_set.append(e_idx.index([j, i]))
                    edge_test_label.append(0)

                    if (i, j) not in e_list_select:
                        inci_mat_T[i, ind] = 1
                        inci_mat_H[j, ind] = 1

        inci_mat_T = inci_mat_T.t()
        inci_mat_H = inci_mat_H.t()
        inci_lb_T = inci_lb_T.t()
        inci_lb_H = inci_lb_H.t()

        if inci_mat_T.size(1) != max_size:
            print(inci_mat_T.size())
            exit()

        pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
        pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

        weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
        weight_tensor_T = torch.ones(weight_mask_T.size(0))
        weight_tensor_T[weight_mask_T] = pos_weight_T

        weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
        weight_tensor_H = torch.ones(weight_mask_H.size(0))
        weight_tensor_H[weight_mask_H] = pos_weight_H

        return inci_mat_T.unsqueeze(dim=0), inci_mat_H.unsqueeze(dim=0), weight_tensor_T, weight_tensor_H, inci_lb_T.unsqueeze(dim=0), inci_lb_H.unsqueeze(dim=0), torch.tensor(edge_set).unsqueeze(dim=0), torch.tensor(edge_test_set).long(), torch.tensor(edge_test_label).long()

    def get_subgraph_verts(v, check_set):

        v_list = list(set(g_stem.neighbors(v)))

        v_list.append(v)

        burning = 0

        if len(v_list) == 0 or len(v_list) == 1:
            return []

        for item in v_list:
            if item in check_set:
                v_list.remove(item)

        while len(v_list) > max_size or len(v_list) < max_size:
            burning = burning + 1
            if burning == 100:
                return []

            if len(v_list) > max_size:
                v_list = random.sample(v_list, max_size)
            else:
                v = random.sample(v_list, 1)[0]
                v_neig = list(set(g_stem.neighbors(v)))
                v_list = list(set(v_list + v_neig))

        return v_list

    train_data_set = []
    valid_data_set = []
    test_data_set = []

    print('Processing sub graphs ...')
    for itt in range(10):
        train_data = []
        test_data = []

        node_idx = list(range(g_stem.vcount()))
        random.shuffle(node_idx)
        NODE_TEST = int(g_stem.vcount() * .3)
        node_train = node_idx[:-NODE_TEST]
        node_test = node_idx[-NODE_TEST:]

        for i in node_train:
            sub_vert_list = get_subgraph_verts(i, node_test)

            if len(sub_vert_list) == 0 or len(sub_vert_list) == 1:
                continue

            g_sub = g_stem.subgraph(sub_vert_list)
            vid, label, feature, adj = get_vert_attrs(g_sub)
            inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label = get_inci(g_sub)

            train_data.append((vid, label, feature, adj, inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label, g_sub))

        for i in node_test:
            sub_vert_list = get_subgraph_verts(i, node_train)

            if len(sub_vert_list) == 0 or len(sub_vert_list) == 1:
                continue

            g_sub = g_stem.subgraph(sub_vert_list)
            vid, label, feature, adj = get_vert_attrs(g_sub)
            inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label = get_inci(g_sub)

            test_data.append((vid, label, feature, adj, inci_mat_T, inci_mat_H, weight_T, weight_H, inci_lb_T, inci_lb_H, edge_set, edge_test_set, edge_test_label, g_sub))

        train_data_set.append(train_data)
        valid_data_set.append(test_data)
        test_data_set.append(test_data)

    NO_SUBGRAPHS = len(train_data)
    MAX_NODES = 3
    MAX_EDGES = 6
    TEST = int(NO_SUBGRAPHS * .15)

    graph_args.num_vertex_type = 5  # original types + start/end types
    graph_args.max_n = MAX_NODES  # maximum number of nodes
    graph_args.max_n_eg = MAX_EDGES
    graph_args.test_eg = TEST
    graph_args.max_true_ng = MAX_EDGES
    graph_args.feature_dimension = features.size(-1)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))
    print('Test edges: %d' % (TEST))

    return train_data_set, valid_data_set, test_data_set, graph_args


##################################################################

def load_WebKB_edges(dataset="cora", directed=False):
    print('Loading {} dataset...'.format(dataset))
    path = './data/{}/'.format(dataset)

    dataset = WebKB(root=path, name=dataset)[0]
    # dataset = WikiCS(root=path)[0]

    features = dataset.x
    labels = dataset.y
    edges =  dataset.edge_index

    MAX_NODES = labels.size(0)

    g_stem = igraph.Graph(directed=True)
    g_stem.add_vertices(labels.size(0))

    edges_list = edges.t().tolist()

    if directed:
        edges_double = []

        for i in range(len(edges_list) - 1):
            for j in range(i + 1, len(edges_list)):
                if edges_list[i][0] == edges_list[j][1] and edges_list[i][1] == edges_list[j][0]:
                    edges_double.append(edges_list[j])

        for e in edges_double:
            if e in edges_list:
                edges_list.remove(e)


    edge_idx = list(range(len(edges_list)))
    random.shuffle(edge_idx)

    edge_true = [edges_list[i] for i in edge_idx]

    ###############################################################
    # add some negative edges, which are not existed in the graph #
    ###############################################################
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    edge_false = []

    if directed:
        for e in edges_list:
            edge_false.append([e[1], e[0]])
    else:
        while len(edge_false) < len(edge_true):
            idx_i = np.random.randint(0, MAX_NODES)
            idx_j = np.random.randint(0, MAX_NODES)

            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], np.array(edges_list)):
                continue
            if edge_false:
                if ismember([idx_i, idx_j], np.array(edge_false)):
                    continue
                if ismember([idx_j, idx_i], np.array(edge_false)):
                    continue
            edge_false.append((idx_i, idx_j))

    assert ~ismember(edge_false, np.array(edges_list))
    assert len(edge_true) == len(edge_false)

    edge_list = edge_true + edge_false
    edge_label = torch.tensor([1] * len(edge_true) + [0] * len(edge_false))

    MAX_EDGES = len(edge_true)

    ################################################################

    inci_mat_T = torch.zeros(MAX_EDGES * 2, MAX_NODES)
    inci_mat_H = torch.zeros(MAX_EDGES * 2, MAX_NODES)

    inci_lb_T = torch.zeros(MAX_EDGES * 2, MAX_NODES)
    inci_lb_H = torch.zeros(MAX_EDGES * 2, MAX_NODES)

    for i, e in enumerate(edge_true):
        inci_lb_T[i][e[0]] = 1
        inci_lb_H[i][e[1]] = 1
        inci_mat_T[i][e[0]] = 1
        inci_mat_H[i][e[1]] = 1


    # adj = adj + adj.t() + torch.eye(adj.shape[0])
    ft_ex = torch.zeros(features.size(0), 1)
    features = torch.cat([features, ft_ex], dim=-1)

    pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
    pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

    weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T
    weight_tensor_T = weight_tensor_T.view(inci_lb_T.size())

    weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H
    weight_tensor_H = weight_tensor_H.view(inci_lb_H.size())

    edge_idx = list(range(len(edge_list)))
    random.shuffle(edge_idx)

    TEST_NO = int(len(edge_idx) * .15)
    train_idx = edge_idx[:-TEST_NO]
    test_idx = edge_idx[-TEST_NO:]

    edge_list = torch.tensor(edge_list)
    train_idx = torch.tensor(train_idx)
    test_idx = torch.tensor(test_idx)

    inci_mat_T = inci_mat_T.index_fill(dim=0, index=test_idx, value=0)
    inci_mat_H = inci_mat_H.index_fill(dim=0, index=test_idx, value=0)

    graph_args.num_vertex_type = 5  # original types + start/end types
    graph_args.max_n = MAX_NODES  # maximum number of nodes
    graph_args.max_n_eg = MAX_EDGES * 2
    graph_args.test_eg = TEST_NO
    graph_args.max_true_ng = MAX_EDGES
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))
    print('Test edges: %d' % (TEST_NO))

    return edge_label, features, labels, edge_list, train_idx, test_idx, inci_mat_T, inci_mat_H, inci_lb_T, inci_lb_H, weight_tensor_T, weight_tensor_H, graph_args


def load_cora_edges(dataset="cora", directed=False):
    print('Loading {} dataset...'.format(dataset))
    path = './data/{}/'.format(dataset)

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    features = normalize_features(features)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    MAX_NODES = labels.size(0)

    edges_list = edges.tolist()

    if directed:
        edges_double = []

        for i in range(len(edges_list) - 1):
            for j in range(i + 1, len(edges_list)):
                if edges_list[i][0] == edges_list[j][1] and edges_list[i][1] == edges_list[j][0]:
                    edges_double.append(edges_list[j])

        for e in edges_double:
            if e in edges_list:
                edges_list.remove(e)

    edge_idx = list(range(len(edges_list)))
    random.shuffle(edge_idx)

    edge_true = [edges_list[i] for i in edge_idx]

    ###############################################################
    # add some negative edges, which are not existed in the graph #
    ###############################################################
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    edge_false = []

    if directed:
        for e in edges_list:
            edge_false.append([e[1], e[0]])
    else:
        while len(edge_false) < len(edge_true):
            idx_i = np.random.randint(0, MAX_NODES)
            idx_j = np.random.randint(0, MAX_NODES)

            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], np.array(edges_list)):
                continue
            if edge_false:
                if ismember([idx_i, idx_j], np.array(edge_false)):
                    continue
                if ismember([idx_j, idx_i], np.array(edge_false)):
                    continue
            edge_false.append((idx_i, idx_j))

    assert ~ismember(edge_false, np.array(edges_list))
    assert len(edge_true) == len(edge_false)

    edge_list = edge_true + edge_false
    edge_label = torch.tensor([1] * len(edge_true) + [0] * len(edge_false))

    MAX_EDGES = len(edge_true)

    ################################################################

    inci_mat_T = torch.zeros(MAX_EDGES * 2, MAX_NODES)
    inci_mat_H = torch.zeros(MAX_EDGES * 2, MAX_NODES)

    inci_lb_T = torch.zeros(MAX_EDGES * 2, MAX_NODES)
    inci_lb_H = torch.zeros(MAX_EDGES * 2, MAX_NODES)

    for i, e in enumerate(edge_true):
        inci_lb_T[i][e[0]] = 1
        inci_lb_H[i][e[1]] = 1
        inci_mat_T[i][e[0]] = 1
        inci_mat_H[i][e[1]] = 1


    # adj = adj + adj.t() + torch.eye(adj.shape[0])
    ft_ex = torch.zeros(features.size(0), 1)
    features = torch.cat([features, ft_ex], dim=-1)

    pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
    pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

    weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T
    weight_tensor_T = weight_tensor_T.view(inci_lb_T.size())

    weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H
    weight_tensor_H = weight_tensor_H.view(inci_lb_H.size())

    edge_idx = list(range(len(edge_list)))
    random.shuffle(edge_idx)

    TEST_NO = int(len(edge_idx) * .15)
    train_idx = edge_idx[:-TEST_NO]
    test_idx = edge_idx[-TEST_NO:]

    edge_list = torch.tensor(edge_list)
    train_idx = torch.tensor(train_idx)
    test_idx = torch.tensor(test_idx)

    inci_mat_T = inci_mat_T.index_fill(dim=0, index=test_idx, value=0)
    inci_mat_H = inci_mat_H.index_fill(dim=0, index=test_idx, value=0)

    graph_args.num_vertex_type = 7  # original types + start/end types
    graph_args.max_n = MAX_NODES  # maximum number of nodes
    graph_args.max_n_eg = MAX_EDGES * 2
    graph_args.test_eg = TEST_NO
    graph_args.max_true_ng = MAX_EDGES
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))
    print('Test edges: %d' % (TEST_NO))

    return edge_label, features, labels, edge_list, train_idx, test_idx, inci_mat_T, inci_mat_H, inci_lb_T, inci_lb_H, weight_tensor_T, weight_tensor_H, graph_args


def load_citeseer_edges(dataset="cora", directed=False):
    print('Loading {} dataset...'.format(dataset))

    cs_content = pd.read_csv('./data/citeseer/citeseer.content', sep='\t', header=None)
    cs_cite = pd.read_csv('./data/citeseer/citeseer.cites', sep='\t', header=None)
    ct_idx = list(cs_content.index)
    paper_id = list(cs_content.iloc[:, 0])
    paper_id = [str(i) for i in paper_id]
    mp = dict(zip(paper_id, ct_idx))

    label = cs_content.iloc[:, -1]
    label = pd.get_dummies(label)

    feature = cs_content.iloc[:, 1:-1]

    mlen = cs_content.shape[0]
    adj = np.zeros((mlen, mlen))

    features = torch.FloatTensor(np.array(feature))
    labels = torch.LongTensor(np.where(label)[1])

    MAX_NODES = labels.size(0)

    edges_list = []

    for i, j in zip(cs_cite[0], cs_cite[1]):
        if str(i) in mp.keys() and str(j) in mp.keys():
                        x = mp[str(i)]
                        y = mp[str(j)]
                        if x != y:
                            edges_list.append([x, y])

    if directed:
        edges_double = []

        for i in range(len(edges_list) - 1):
            for j in range(i + 1, len(edges_list)):
                if edges_list[i][0] == edges_list[j][1] and edges_list[i][1] == edges_list[j][0]:
                    edges_double.append(edges_list[j])

        for e in edges_double:
            if e in edges_list:
                edges_list.remove(e)

    edge_idx = list(range(len(edges_list)))
    random.shuffle(edge_idx)

    edge_true = [edges_list[i] for i in edge_idx]

    ###############################################################
    # add some negative edges, which are not existed in the graph #
    ###############################################################
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    edge_false = []

    if directed:
        for e in edges_list:
            edge_false.append([e[1], e[0]])
    else:
        while len(edge_false) < len(edge_true):
            idx_i = np.random.randint(0, MAX_NODES)
            idx_j = np.random.randint(0, MAX_NODES)

            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], np.array(edges_list)):
                continue
            if edge_false:
                if ismember([idx_i, idx_j], np.array(edge_false)):
                    continue
                if ismember([idx_j, idx_i], np.array(edge_false)):
                    continue
            edge_false.append((idx_i, idx_j))

    assert ~ismember(edge_false, np.array(edges_list))
    assert len(edge_true) == len(edge_false)

    edge_list = edge_true + edge_false
    edge_label = torch.tensor([1] * len(edge_true) + [0] * len(edge_false))

    MAX_EDGES = len(edge_true)

    ################################################################

    inci_mat_T = torch.zeros(MAX_EDGES * 2, MAX_NODES)
    inci_mat_H = torch.zeros(MAX_EDGES * 2, MAX_NODES)

    inci_lb_T = torch.zeros(MAX_EDGES * 2, MAX_NODES)
    inci_lb_H = torch.zeros(MAX_EDGES * 2, MAX_NODES)

    for i, e in enumerate(edge_true):
        inci_lb_T[i][e[0]] = 1
        inci_lb_H[i][e[1]] = 1
        inci_mat_T[i][e[0]] = 1
        inci_mat_H[i][e[1]] = 1

    ft_ex = torch.zeros(features.size(0), 1)
    features = torch.cat([features, ft_ex], dim=-1)

    pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
    pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

    weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T
    weight_tensor_T = weight_tensor_T.view(inci_lb_T.size())

    weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H
    weight_tensor_H = weight_tensor_H.view(inci_lb_H.size())

    edge_idx = list(range(len(edge_list)))
    random.shuffle(edge_idx)

    TEST_NO = int(len(edge_idx) * .15)
    train_idx = edge_idx[:-TEST_NO]
    test_idx = edge_idx[-TEST_NO:]

    edge_list = torch.tensor(edge_list)
    train_idx = torch.tensor(train_idx)
    test_idx = torch.tensor(test_idx)

    inci_mat_T = inci_mat_T.index_fill(dim=0, index=test_idx, value=0)
    inci_mat_H = inci_mat_H.index_fill(dim=0, index=test_idx, value=0)

    graph_args.num_vertex_type = 6  # original types + start/end types
    graph_args.max_n = MAX_NODES  # maximum number of nodes
    graph_args.max_n_eg = MAX_EDGES * 2
    graph_args.test_eg = TEST_NO
    graph_args.max_true_ng = MAX_EDGES
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))
    print('Test edges: %d' % (TEST_NO))

    return edge_label, features, labels, edge_list, train_idx, test_idx, inci_mat_T, inci_mat_H, inci_lb_T, inci_lb_H, weight_tensor_T, weight_tensor_H, graph_args
