## GraphSNN Document

## Usage
train on cora dataset
``` bash=
python3 graphsnn_gcn_mm.py
# Total time elapsed: 14.9772s
# Test set results: loss= 1.0738 accuracy= 0.8310
```

``` bash=
python3 graphsnn_gin_mm.py
# Total time elapsed: 23.3898s
# Test set results: loss= 0.7250 accuracy= 0.7990
```

## structural coefficient matrix
Two methods of finding the structural coefficient matrix are implemented
* `get_sc_matrix_pyg(num_node, edge_index, lambda_)`
``` python=
    """
    Calculates the structural coefficients matrix for a given graph.

    Parameters:
        num_node (int): The number of nodes in the graph.
        edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges] and type torch.long.
        lambda_ (float): A parameter used in the calculation of structural coefficients.

    Returns:
        torch.Tensor: The structural coefficients matrix with shape [num_node, num_node].
    """
```


* `get_sc_matrix_nx(G)`
``` python=
    """
    Calculates the structural coefficients matrix for a given networkx graph.

    Parameters:
        G (nx.Graph): The networkx graph.

    Returns:
        np.ndarray: The structural coefficients matrix as a numpy array.
    """
```

### Benchmark
``` python=
def sc_benchmarking(data, epsilon=1e-9):
    A_norm, A, X, labels, idx_train, idx_val, idx_test = load_citation_data(
        data)
    G = nx.from_numpy_array(A)
    num_node = data['x'].shape[0]
    edge_index = data['edge_index']
    sc_pyg = get_sc_matrix_pyg(num_node, edge_index, 1)
    sc_nx = get_sc_matrix_nx(G)
    # print(sc_pyg)
    # print(sc_nx)
    error = torch.mean(torch.abs(sc_nx - sc_pyg))
    assert error < epsilon, error
```
> Elapsed time for get_sc_matrix_pyg: 4.3355255126953125 seconds
> Elapsed time for get_sc_matrix_nx: 5.996627330780029 seconds

The default is to use the  function `get_sc_matrix_pyg`, because it is faster

## GNN layer
``` python=
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
```