#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import numpy as np
from numba import jit
from numba import njit

import torch
import networkx as nx
from scipy import sparse
import torch.optim as optim
import torch.nn.functional as F
from torch.multiprocessing import Pool
from torch_geometric.utils import subgraph
from torch_geometric.utils import k_hop_subgraph

from utils import *
from models import GNN_MP
from dataset_utils import DataLoader

import warnings
warnings.filterwarnings('ignore')


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for {func.__name__}: {elapsed_time} seconds")
        return result
    return wrapper


def get_overlap_subgraph(u, v, edge_index):
    u_subset, _, _, _ = k_hop_subgraph(u, 1, edge_index, relabel_nodes=False)
    v_subset, _, _, _ = k_hop_subgraph(v, 1, edge_index, relabel_nodes=False)
    overlap_subset = np.intersect1d(u_subset, v_subset, assume_unique=True)
    overlap_edge_index = subgraph(torch.tensor(overlap_subset), edge_index)[0]
    return len(overlap_subset), overlap_edge_index.shape[1] / 2


@timer
def get_sc_matrix_pyg(num_node, edge_index, lambda_):
    structural_coeff = torch.zeros(num_node, num_node)
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        s_num_node, s_num_edge = get_overlap_subgraph(u, v, edge_index)
        structural_coeff[u, v] = s_num_edge * \
            (s_num_node ** lambda_) / (s_num_node * (s_num_node - 1))
    return structural_coeff


@timer
def get_sc_matrix_nx(G):
    A = nx.adjacency_matrix(G).todense()
    sub_graphs = []

    for i in np.arange(len(A)):
        s_indexes = []
        for j in np.arange(len(A)):
            s_indexes.append(i)
            if (A[i][j] == 1):
                s_indexes.append(j)
        sub_graphs.append(G.subgraph(s_indexes))

    subgraph_nodes_list = []

    for i in np.arange(len(sub_graphs)):
        subgraph_nodes_list.append(list(sub_graphs[i].nodes))

    sub_graphs_adj = []
    for index in np.arange(len(sub_graphs)):
        sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())

    structural_coeff = torch.zeros(A.shape[0], A.shape[0])

    for node in np.arange(len(subgraph_nodes_list)):
        sub_adj = sub_graphs_adj[node]
        for neighbors in np.arange(len(subgraph_nodes_list[node])):
            index = subgraph_nodes_list[node][neighbors]
            count = torch.tensor(0).float()
            if (index == node):
                continue
            else:
                c_neighbors = set(
                    subgraph_nodes_list[node]).intersection(
                    subgraph_nodes_list[index])
                if index in c_neighbors:
                    nodes_list = subgraph_nodes_list[node]
                    sub_graph_index = nodes_list.index(index)
                    c_neighbors_list = list(c_neighbors)
                    for i, item1 in enumerate(nodes_list):
                        if (item1 in c_neighbors):
                            for item2 in c_neighbors_list:
                                j = nodes_list.index(item2)
                                count += sub_adj[i][j]

                structural_coeff[node][index] = count / 2
                structural_coeff[node][index] = structural_coeff[node][index] / \
                    (len(c_neighbors) * (len(c_neighbors) - 1))
                structural_coeff[node][index] = structural_coeff[node][index] * \
                    (len(c_neighbors)**1)
    return structural_coeff


def sc_benchmarking(data, epsilon=1e-9):
    A_norm, A, X, labels, idx_train, idx_val, idx_test = load_citation_data(
        data)
    G = nx.from_numpy_array(A)
    num_node = data['x'].shape[0]
    edge_index = data['edge_index']
    sc_pyg = get_sc_matrix_pyg(num_node, edge_index, 1)
    sc_nx = get_sc_matrix_nx(G)
    print(sc_pyg)
    print(sc_nx)
    error = torch.mean(torch.abs(sc_nx - sc_pyg))
    assert error < epsilon, error


def sc_normalization(sc, adj):
    normalized_sc = torch.tensor(sc)
    normalized_sc = normalized_sc / normalized_sc.sum(1, keepdim=True)
    normalized_sc = normalized_sc + torch.tensor(adj)
    adjust = normalized_sc.sum(1, keepdim=True)
    adjust = torch.diag(adjust.t()[0])
    normalized_sc = normalized_sc + adjust
    normalized_sc = torch.nan_to_num(torch.tensor(normalized_sc), nan=0)
    return normalized_sc


def degree_normalization(sc):
    row_sum = torch.sum(sc, dim=1)
    degree_matrix = torch.diag(row_sum + 1)
    D = degree_matrix.float().inverse().sqrt()
    return torch.matmul(torch.matmul(D, sc), D)


def preprocess(data):
    A_norm, A, X, labels, idx_train, idx_val, idx_test = load_citation_data(
        data)

    G = nx.from_numpy_array(A)
    feature_dictionary = {}

    for i in np.arange(len(labels)):
        feature_dictionary[i] = labels[i]

    nx.set_node_attributes(G, feature_dictionary, "attr_name")

    features = torch.FloatTensor(X)
    labels = torch.LongTensor(labels)

    # sc = get_sc_matrix_nx(G)
    num_node = data['x'].shape[0]
    edge_index = data['edge_index']
    sc = get_sc_matrix_pyg(num_node, edge_index, 1)
    normalized_sc = sc_normalization(sc, A)
    normalized_sc = degree_normalization(normalized_sc)
    edge_weight = []
    for i in range(edge_index.shape[1]):
        u, v = edge_index[1, i], edge_index[0, i]
        edge_weight.append(normalized_sc[u, v])
    edge_weight = torch.tensor(edge_weight)
    # print(f"edge_weight.shape = {edge_weight.shape}")
    return features, labels, idx_train, idx_val, idx_test, edge_weight


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_index, edge_weight)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately, deactivates dropout
        # during validation run.
        model.eval()
        output = model(features, edge_index, edge_weight)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, edge_index, edge_weight)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


def get_G1():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4)
    ])
    return G


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=8e-3,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', default='cora', help='Dataset name.')
    args = parser.parse_args("")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dname = args.dataset
    dataset = DataLoader(dname)
    data = dataset[0]

    features, labels, idx_train, idx_val, idx_test, edge_weight = preprocess(data)
    edge_index = data['edge_index']
    # Model and optimizer
    model = GNN_MP(nfeat=features.shape[1],
                   nhid=args.hidden,
                   nclass=labels.max().item() + 1,
                   dropout=args.dropout)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()
