#!/usr/bin/env python
# coding: utf-8

from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from scipy import sparse
from scipy.linalg import fractional_matrix_power
from torch_geometric.utils import subgraph
from torch_geometric.utils import k_hop_subgraph


from utils import *
from param import get_args
from models import Graphsn_GIN
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

def get_overlap_subgraph(u_subset, v_subset, edge_index):
    overlap_subset = np.intersect1d(u_subset, v_subset, assume_unique=True)
    overlap_edge_index = subgraph(torch.tensor(overlap_subset), edge_index)[0]
    return len(overlap_subset), overlap_edge_index.shape[1] / 2

@timer
def get_sc_matrix_pyg(num_node, edge_index, lambda_):
    """
    Calculates the structural coefficients matrix for a given graph.

    Parameters:
        num_node (int): The number of nodes in the graph.
        edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges] and type torch.long.
        lambda_ (float): A parameter used in the calculation of structural coefficients.

    Returns:
        torch.Tensor: The structural coefficients matrix with shape [num_node, num_node].
    """
    subgraph_list = []
    for u in range(num_node):
        subgraph_list.append(k_hop_subgraph(u, 1, edge_index, relabel_nodes=False)[0])
    structural_coeff = torch.zeros(num_node, num_node)
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        s_num_node, s_num_edge = get_overlap_subgraph(
            subgraph_list[u], subgraph_list[v], edge_index)
        assert (s_num_node * (s_num_node - 1)) != 0, f"s_num_node = {s_num_node}"
        structural_coeff[u, v] = s_num_edge * (s_num_node ** lambda_) / (s_num_node * (s_num_node - 1))
    return structural_coeff


def sc_normalization(sc, adj):
    normalized_sc = torch.tensor(sc)
    normalized_sc = normalized_sc / normalized_sc.sum(1, keepdim=True)
    normalized_sc = normalized_sc + torch.tensor(adj)
    adjust = normalized_sc.sum(1, keepdim=True)
    adjust = torch.diag(adjust.t()[0])
    normalized_sc = normalized_sc + adjust
    normalized_sc = torch.nan_to_num(torch.tensor(normalized_sc), nan=0)
    return normalized_sc


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

    num_node = data['x'].shape[0]
    edge_index = data['edge_index']
    sc = get_sc_matrix_pyg(num_node, edge_index, 1)
    normalized_sc = sc_normalization(sc, A)

    return features, labels, idx_train, idx_val, idx_test, normalized_sc


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, normalized_sc)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, normalized_sc)

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
    output = model(features, normalized_sc)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == "__main__":
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dname = args.dataset
    dataset = DataLoader(dname)
    data = dataset[0]

    features, labels, idx_train, idx_val, idx_test, normalized_sc = preprocess(data)
    model = Graphsn_GIN(nfeat=features.shape[1],
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
