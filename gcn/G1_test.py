#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import numpy as np
from typing import List, Tuple, Union, Optional

import torch
import networkx as nx
import torch.nn.functional as F
from torch.multiprocessing import Pool
from torch_geometric.utils import subgraph
from torch_geometric.utils import k_hop_subgraph

from numba import cuda

from utils import *
from param import get_args
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
        subgraph_list.append(
            k_hop_subgraph(
                u,
                1,
                edge_index,
                relabel_nodes=False)[0])
    structural_coeff = torch.zeros(num_node, num_node)
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        s_num_node, s_num_edge = get_overlap_subgraph(
            subgraph_list[u], subgraph_list[v], edge_index)
        print(f"({u}, {v})\ts_num_node={s_num_node}\ts_num_edge={s_num_edge}")
        assert (s_num_node * (s_num_node - 1)
                ) != 0, f"s_num_node = {s_num_node}"
        structural_coeff[u, v] = s_num_edge * \
            (s_num_node ** lambda_) / (s_num_node * (s_num_node - 1))
    return structural_coeff


@timer
def get_sc_matrix_nx(G):
    """
    Calculates the structural coefficients matrix for a given networkx graph.

    Parameters:
        G (nx.Graph): The networkx graph.

    Returns:
        torch.Tensor: The structural coefficients matrix with shape [num_node, num_node].
    """
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


def subgraph_numpy(subset: Union[np.ndarray,
                                 List[int]],
                   edge_index: np.ndarray,
                   edge_attr: Optional[np.ndarray] = None,
                   relabel_nodes: bool = False,
                   num_nodes: Optional[int] = None,
                   return_edge_mask: bool = False) -> Union[Tuple[np.ndarray,
                                                                  Optional[np.ndarray]],
                                                            Tuple[np.ndarray,
                                                                  Optional[np.ndarray],
                                                                  Optional[np.ndarray]]]:
    subset = np.asarray(subset)
    edge_index = np.asarray(edge_index)
    edge_attr = np.asarray(edge_attr) if edge_attr is not None else None

    if num_nodes is None:
        num_nodes = np.max(edge_index) + 1
    mask = np.zeros(num_nodes, dtype=bool)
    mask[subset] = True

    filtered_edge_index = edge_index[:, np.logical_and(
        mask[edge_index[0]], mask[edge_index[1]])]

    if relabel_nodes:
        node_map = np.zeros(num_nodes, dtype=int)
        node_map[mask] = np.arange(np.sum(mask))
        filtered_edge_index = node_map[filtered_edge_index]

    filtered_edge_attr = None
    if edge_attr is not None:
        filtered_edge_attr = edge_attr[np.logical_and(
            mask[edge_index[0]], mask[edge_index[1]])]

    if return_edge_mask:
        edge_mask = np.logical_and(mask[edge_index[0]], mask[edge_index[1]])
        return filtered_edge_index, filtered_edge_attr, edge_mask
    else:
        return filtered_edge_index, filtered_edge_attr


def get_overlap_subgraph_numpy(u_subset, v_subset, edge_index):
    overlap_subset = np.intersect1d(u_subset, v_subset, assume_unique=True)
    overlap_edge_index = subgraph_numpy(
        torch.tensor(overlap_subset), edge_index)[0]
    return len(overlap_subset), overlap_edge_index.shape[1] / 2


def k_hop_subgraph_numpy(
        node_idx,
        num_hops,
        edge_index,
        relabel_nodes=False,
        num_nodes=None,
        flow='source_to_target',
        directed=False):
    node_idx = np.asarray(node_idx)
    edge_index = np.asarray(edge_index)

    if num_nodes is None:
        num_nodes = np.max(edge_index) + 1

    visited = np.zeros(num_nodes, dtype=bool)
    visited[node_idx] = True

    frontier = np.copy(node_idx)

    subgraph_nodes = [node_idx]
    subgraph_edges = []

    for _ in range(num_hops):
        neighbors = np.unique(edge_index[:, np.isin(edge_index[0], frontier)])
        new_nodes = neighbors[~visited[neighbors]]
        visited[new_nodes] = True
        frontier = new_nodes
        subgraph_nodes.append(frontier)
        edges = edge_index[:, np.isin(edge_index[0], frontier)]
        if not directed:
            mask = np.logical_or(visited[edges[0]], visited[edges[1]])
            edges = edges[:, ~mask]
        subgraph_edges.append(edges)
    subgraph_nodes = np.concatenate(subgraph_nodes)
    subgraph_edges = np.concatenate(subgraph_edges, axis=1)

    if relabel_nodes:
        node_map = np.zeros(num_nodes, dtype=int)
        node_map[visited] = np.arange(np.sum(visited))
        subgraph_edges = node_map[subgraph_edges]
    subgraph_mask = np.ones(subgraph_edges.shape[1], dtype=bool)
    return subgraph_nodes, subgraph_edges, subgraph_mask, visited


@timer
def get_sc_matrix_numpy(num_node, edge_index, lambda_):
    subgraph_list = []
    for u in range(num_node):
        subgraph_list.append(k_hop_subgraph_numpy([u], 1, edge_index)[0])
    # with open("out/subgraph.txt", 'w') as fp:
    #     for s_nodes in subgraph_list:
    #         fp.write(f"{s_nodes}\n")
    structural_coeff = torch.zeros(num_node, num_node)
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        s_num_node, s_num_edge = get_overlap_subgraph_numpy(
            subgraph_list[u], subgraph_list[v], edge_index)
        structural_coeff[u, v] = s_num_edge * \
            (s_num_node ** lambda_) / (s_num_node * (s_num_node - 1))
    return structural_coeff


def sc_benchmarking(data, epsilon=1e-9):
    A_norm, A, X, labels, idx_train, idx_val, idx_test = load_citation_data(data)
    # A_norm, A, X, labels, idx_train, idx_val, idx_test = load_nell_data(data)
    G = nx.from_numpy_array(A)
    num_node = data.num_nodes
    edge_index = data['edge_index']
    np.savetxt("out/cora_edge_index.csv", edge_index, delimiter=',', fmt="%d")
    sc_pyg = get_sc_matrix_pyg(num_node, edge_index, 1)
    sc_nx = get_sc_matrix_nx(G)
    error = torch.mean(torch.abs(sc_nx - sc_pyg))
    assert error < epsilon, error

    edge_index = edge_index.numpy()
    sc_numpy = get_sc_matrix_numpy(num_node, edge_index, 1)
    np.savetxt("out/sc.txt", np.round(sc_numpy, 2), delimiter=',', fmt="%.2f")
    error = torch.mean(torch.abs(sc_nx - sc_numpy))
    assert error < epsilon, error


def get_G1():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4)
    ])
    return G


if __name__ == "__main__":
    args = get_args()
    print(args)
    
    num_nodes = 5
    edge_index = torch.tensor([[0,0,0,0,1,2,3,1,2,3,4,2,3,4], 
                               [1,2,3,4,2,3,4,0,0,0,0,1,2,3]])
    
    sc_nx = get_sc_matrix_pyg(num_nodes, edge_index, 1)
    print(sc_nx)

