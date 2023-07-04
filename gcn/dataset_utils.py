import torch
import math

import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import NELL

def DataLoader(name):
    name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, transform=None)
    elif name == 'nell':
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = NELL(path, transform=None)
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset