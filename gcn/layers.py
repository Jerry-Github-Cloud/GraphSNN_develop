import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import MessagePassing


class Graphsn_GCN_MM(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Graphsn_GCN_MM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.eps = nn.Parameter(torch.FloatTensor(1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.9 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
        stdv_eps = 0.21 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, sc):
        v = (self.eps) * torch.diag(sc)
        mask = torch.diag(torch.ones_like(v))
        # print(f"v = {v}")
        # print(f"\tmask = {mask}")
        sc = mask * torch.diag(v) + (1. - mask) * sc
        support = torch.mm(input, self.weight)
        output = torch.spmm(sc, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               

class Graphsn_GCN_MP(MessagePassing):
    def __init__(self, in_features, out_features, bias=True):
        super(Graphsn_GCN_MP, self).__init__(aggr='add')
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.eps = nn.Parameter(torch.FloatTensor(1))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.9 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        stdv_eps = 0.21 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index, edge_weight):
        # print(f"\tedge_index.shape = {edge_index.shape}")
        v = (self.eps) * torch.diag(edge_weight)
        mask = torch.diag(torch.ones_like(v))
        edge_weight = mask * torch.diag(v) + (1. - mask) * edge_weight

        support = torch.matmul(x, self.weight)
        output = self.propagate(edge_index, x=support, edge_weight=edge_weight)

        if self.bias is not None:
            output = output + self.bias

        return output

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    # def update(self, aggr_out):
    #     print(f"\taggr_out.shape = {aggr_out.shape}")
    #     return aggr_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
