import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import Graphsn_GCN_MM
from layers import Graphsn_GCN_MP

class GNN_MM(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN_MM, self).__init__()
        self.gc1 = Graphsn_GCN_MM(nfeat, nhid)
        self.gc2 = Graphsn_GCN_MM(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, sc):
        """
        Forward pass of the GNN model.

        Parameters:
            x (torch.Tensor): The input feature of the GNN. Shape: (num_node, feature dimension)
            sc (torch.Tensor): The structural coefficients matrix. Shape: [num_node, num_node]

        Returns:
            torch.Tensor: The output tensor of the forward pass.
        """
        x = F.relu(self.gc1(x, sc))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, sc)
        return F.log_softmax(x, dim=-1)


class GNN_MP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN_MP, self).__init__()
        
        self.gc1 = Graphsn_GCN_MP(nfeat, nhid)
        self.gc2 = Graphsn_GCN_MP(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        
        x = F.relu(self.gc1(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight)
        
        return F.log_softmax(x, dim=-1)

